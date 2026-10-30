"""One dense covariance build for several stacks, sliced back into the per-stack matrices.

Every covariance entry ``(i, j)`` comes from row ``i``'s data, column ``j``'s data and the kernel
constants only. The stacks' kernel inputs are concatenated (the i side along dim 0, the j side along
dim 1), so the diagonal blocks of the one ``(N, N)`` matrix are exactly the per-stack matrices; the
off-diagonal blocks are computed and never read. The terms placed by position (nugget diagonals, fault
drift rows) are placed block by block (``KernelInput.block_layout``, read by ``_covariance_assembler``).

Stacks share a build only with the same kernel options (the kernel constants enter as Python floats)
and the same number of faults (the width of the fault drift vectors).
"""
import re
from dataclasses import dataclass

from ._kernels_assembler import create_cov_kernel
from ._structs import KernelInput, SurfacePointNuggets
from ._vectors_preparation import cov_vectors_preparation
from ...core.backend_tensor import BackendTensor
from ...core.data.matrices_sizes import MatricesSizes

# Rows of one stacked build; its (N, N, 3) intermediates stay at a few tens of MB.
MAX_STACKED_SIZE = 1024


@dataclass(frozen=True)
class BlockLayout:
    sizes: tuple  # covariance rows of each block
    nugget_grad: tuple  # each block's gradient nuggets (their length is the block's gradient rows)
    nugget_scalar: tuple  # each block's SurfacePointNuggets


def covariance_size(solver_input, kernel_options) -> int:
    sp_, ori_ = solver_input.sp_internal, solver_input.ori_internal
    return MatricesSizes(ori_size=ori_.n_orientations_tiled, sp_size=sp_.n_points, uni_drift_size=kernel_options.n_uni_eq,
                         faults_size=solver_input.fault_internal.n_faults, dim=kernel_options.number_dimensions,
                         n_dips=ori_.n_orientations).cov_size


def stacked_covariances(solver_inputs, kernel_options) -> list:
    """The dense covariance matrix of each stack in ``solver_inputs``, from one build."""
    kernel_inputs = [cov_vectors_preparation(solver_input, kernel_options) for solver_input in solver_inputs]
    matrix = create_cov_kernel(stack_kernel_inputs(kernel_inputs), kernel_options)
    blocks, start = [], 0
    for ki in kernel_inputs:
        size = ki.drift_matrix_selector.sel_ui.shape[0]
        blocks.append(matrix[start:start + size, start:start + size].contiguous())
        start += size
    return blocks


def stack_kernel_inputs(kernel_inputs: list[KernelInput]) -> KernelInput:
    """One covariance ``KernelInput`` for several stacks: every i-side field concatenated along dim 0,
    every j-side field along dim 1."""
    first = kernel_inputs[0]
    faults = first.ref_fault is not None
    if any((ki.ref_fault is not None) != faults or (faults and ki.ref_fault.n_faults_i != first.ref_fault.n_faults_i)
           for ki in kernel_inputs):
        raise ValueError("stacked covariance: every stack needs the same number of faults")
    cat = BackendTensor.t.cat
    stacked = KernelInput(
        ori_sp_matrices=_joined([ki.ori_sp_matrices for ki in kernel_inputs]),
        cartesian_selector=_joined([ki.cartesian_selector for ki in kernel_inputs]),
        nugget_scalar=SurfacePointNuggets(**{
            name: None if any(getattr(ki.nugget_scalar, name) is None for ki in kernel_inputs)
            else cat([getattr(ki.nugget_scalar, name) for ki in kernel_inputs])
            for name in ("rest", "reference", "surface_ids")}),
        nugget_grad=cat([ki.nugget_grad for ki in kernel_inputs]),
        ori_drift=_joined([ki.ori_drift for ki in kernel_inputs]),
        ref_drift=_joined([ki.ref_drift for ki in kernel_inputs]),
        rest_drift=_joined([ki.rest_drift for ki in kernel_inputs]),
        drift_matrix_selector=_joined([ki.drift_matrix_selector for ki in kernel_inputs]),
        ref_fault=_joined([ki.ref_fault for ki in kernel_inputs]) if faults else None,
        rest_fault=_joined([ki.rest_fault for ki in kernel_inputs]) if faults else None,
    )
    stacked.block_layout = BlockLayout(
        sizes=tuple(ki.drift_matrix_selector.sel_ui.shape[0] for ki in kernel_inputs),
        nugget_grad=tuple(ki.nugget_grad for ki in kernel_inputs),
        nugget_scalar=tuple(ki.nugget_scalar for ki in kernel_inputs),
    )
    return stacked


def _joined(structs):
    """``structs[0]``'s type with each field concatenated over ``structs``: names ending in ``i`` (before
    any digits) are row-side, ``(n, 1, d)``; names ending in ``j`` are column-side, ``(1, n, d)``."""
    cls = type(structs[0])
    joined = cls.__new__(cls)
    for name, value in vars(structs[0]).items():
        if name == "n_faults_i":
            joined.n_faults_i = value
            continue
        side = re.sub(r"\d+$", "", name)[-1]
        if side not in "ij":
            raise ValueError(f"stacked covariance: {cls.__name__}.{name} is neither a row nor a column field")
        joined.__dict__[name] = BackendTensor.t.cat([vars(s)[name] for s in structs], dim=0 if side == "i" else 1)
    return joined
