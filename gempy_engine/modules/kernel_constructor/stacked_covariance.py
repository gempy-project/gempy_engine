"""One dense covariance build for several stacks, sliced back into the per-stack matrices.

Every covariance entry ``(i, j)`` comes from row ``i``'s data, column ``j``'s data and the kernel
constants only. The stacks' kernel inputs are concatenated (the i side along dim 0, the j side along
dim 1), so the diagonal blocks of the one ``(N, N)`` matrix are exactly the per-stack matrices; the
off-diagonal blocks are computed and never read. The terms placed by position (nugget diagonals, fault
drift rows) are placed block by block (``KernelInput.block_layout``, read by ``_covariance_assembler``).

Stacks share a build only with the same kernel options (the kernel constants enter as Python floats)
and the same number of faults (the width of the fault drift vectors).

The concatenated kernel input itself is built for the whole chunk at once (``chunk_kernel_input``): every
row vector of ``cov_vectors_preparation`` is a copy of input rows, a pattern that depends only on sizes,
or an elementwise ``where`` of the two, so one gather from the stacks' arrays concatenated and one upload
per pattern give the same values as building each stack's vectors and concatenating them.
"""
import re
from dataclasses import dataclass

import numpy as np

from ._kernel_constructors import ug_host_patterns, universal_host_rows
from ._kernel_selectors import cartesian_host_patterns
from ._kernels_assembler import create_cov_kernel
from ._structs import (CartesianSelector, DriftMatrixSelector, FaultDrift, KernelInput, OrientationSurfacePointsCoords,
                       OrientationsDrift, PointsDrift, SurfacePointNuggets)
from ._vectors_preparation import StackAssemblies, assemblies_key, cov_vectors_preparation
from ...core.backend_tensor import BackendTensor
from ...core.compute_scope import stack_cache_of
from ...core.data.matrices_sizes import MatricesSizes
from ...core.host_arrays import upload

# Rows of one stacked build; its (N, N, 3) intermediates stay at a few tens of MB.
MAX_STACKED_SIZE = 1024


@dataclass(frozen=True)
class BlockLayout:
    sizes: tuple  # covariance rows of each block
    nugget_grad: tuple  # each block's gradient nuggets (their length is the block's gradient rows)
    nugget_scalar: tuple  # each block's SurfacePointNuggets


def covariance_size(solver_input, kernel_options) -> int:
    return _matrices_sizes(solver_input, kernel_options).cov_size


def _matrices_sizes(solver_input, kernel_options) -> MatricesSizes:
    sp_, ori_ = solver_input.sp_internal, solver_input.ori_internal
    return MatricesSizes(ori_size=ori_.n_orientations_tiled, sp_size=sp_.n_points, uni_drift_size=kernel_options.n_uni_eq,
                         faults_size=solver_input.fault_internal.n_faults, dim=kernel_options.number_dimensions,
                         n_dips=ori_.n_orientations)


def stacked_covariances(solver_inputs, kernel_options) -> list:
    """The dense covariance matrix of each stack in ``solver_inputs``, from one build."""
    stacked = chunk_kernel_input(solver_inputs, kernel_options)
    if stacked is None:  # mixed dtypes: concatenate the per-stack inputs, which keeps their promotions
        stacked = stack_kernel_inputs([cov_vectors_preparation(solver_input, kernel_options)
                                       for solver_input in solver_inputs])
    matrix = create_cov_kernel(stacked, kernel_options)
    blocks, start = [], 0
    for size in stacked.block_layout.sizes:
        blocks.append(matrix[start:start + size, start:start + size].contiguous())
        start += size
    return blocks


def chunk_kernel_input(solver_inputs, kernel_options) -> KernelInput | None:
    """``stack_kernel_inputs`` of the stacks' ``cov_vectors_preparation``, value for value, built at once.

    The coordinate and drift vectors are one gather from a source of every stack's tiled dip positions,
    reference and rest points (plus the universal-drift rows and a zero row); the fault vectors one gather
    from the stacks' fault values; the selectors one upload of the stacks' host patterns. Each stack's
    ``StackAssemblies`` (shared with its evaluation data side) is left in its cache as views of the chunk
    vectors. None when the arrays do not all have the engine dtype.
    """
    t, dtype, options = BackendTensor.t, BackendTensor.dtype_obj, kernel_options
    sps = [solver_input.sp_internal for solver_input in solver_inputs]
    oris = [solver_input.ori_internal for solver_input in solver_inputs]
    faults = [solver_input.fault_internal for solver_input in solver_inputs]
    n_faults = faults[0].n_faults
    if any(f.n_faults != n_faults for f in faults):
        raise ValueError("stacked covariance: every stack needs the same number of faults")
    arrays = [a for sp, ori in zip(sps, oris) for a in (ori.dip_positions_tiled, sp.ref_surface_points, sp.rest_surface_points)]
    arrays += [a for f in faults for a in (f.fault_values_ref, f.fault_values_rest)] if n_faults else []
    if (any(getattr(a, "dtype", None) != dtype for a in arrays)
            or t.empty(0, dtype=dtype, device="cpu").numpy().dtype != np.dtype(BackendTensor.dtype)):
        return None

    sizes = [_matrices_sizes(solver_input, options) for solver_input in solver_inputs]
    n_dim, drift, degree = options.number_dimensions, sizes[0].drifts_size, options.uni_degree
    ori_n, sp_n = [z.ori_size for z in sizes], [z.sp_size for z in sizes]
    pos_at, ref_at = _offsets(ori_n, 0), _offsets(sp_n, sum(ori_n))
    rest_at, universal_at = _offsets(sp_n, sum(ori_n) + sum(sp_n)), sum(ori_n) + 2 * sum(sp_n)
    zero = universal_at + 3 * drift

    def rows(pieces_of):  # one index per chunk row, block after block
        return np.concatenate([np.concatenate(pieces_of(k)) for k in range(len(sizes))]).astype(np.int64)

    def run(start, n):
        return np.arange(start, start + n)

    def zeros(n):
        return np.full(n, zero)

    point_fields = {
        "dips_ref_coord": rows(lambda k: (run(pos_at[k], ori_n[k]), run(ref_at[k], sp_n[k]), zeros(drift))),
        "dips_rest_coord": rows(lambda k: (run(pos_at[k], ori_n[k]), run(rest_at[k], sp_n[k]), zeros(drift))),
    }
    for side, at in (("ref", ref_at), ("rest", rest_at)):
        for term in range(3):  # degree 1, degree 2a, degree 2b
            point_fields[f"{side}_{term}"] = rows(lambda k, at=at, term=term: (
                zeros(ori_n[k]), run(at[k], sp_n[k]), run(universal_at + term * drift, drift)))
    if degree == 2:  # dip positions aligned with the covariance rows (assembly_dips_ug_coords)
        point_fields["positions"] = rows(lambda k: (run(pos_at[k], ori_n[k]), zeros(sizes[k].cov_size - ori_n[k])))
    index_parts = list(point_fields.values())
    if n_faults:
        refs_at = _offsets(sp_n, 0)
        rests_at, eye_at = _offsets(sp_n, sum(sp_n)), 2 * sum(sp_n) + 1
        fault_zero = 2 * sum(sp_n)
        for at in (refs_at, rests_at):
            index_parts.append(rows(lambda k, at=at: (np.full(ori_n[k], fault_zero), run(at[k], sp_n[k]),
                                                      np.full(options.n_uni_eq, fault_zero), run(eye_at, n_faults))))
    index = t.split(upload(np.concatenate(index_parts)), [len(part) for part in index_parts])

    host_rows = np.concatenate((universal_host_rows(options, drift).reshape(-1, n_dim), np.zeros((1, n_dim))))
    source = t.cat([ori.dip_positions_tiled for ori in oris] + [sp.ref_surface_points for sp in sps]
                   + [sp.rest_surface_points for sp in sps] + [upload(host_rows.astype(BackendTensor.dtype))])
    gathered = source[t.cat(index[:len(point_fields)])].reshape(len(point_fields), -1, n_dim)
    fields = dict(zip(point_fields, gathered))

    ug = upload(np.concatenate([ug_host_patterns(ori.n_orientations, size, degree) for ori, size in zip(oris, sizes)], axis=1))
    dips_a, dips_b, dips_c, second_degree_selector = ug[0], ug[1], ug[3], ug[5]
    if degree == 2:
        dips_b = t.where(ug[2] > 0, fields["positions"], dips_b)
        dips_c = t.where(ug[4] > 0, fields["positions"], dips_c)
    sel_hu, sel_hv, sel_huv = upload(np.concatenate([cartesian_host_patterns(size) for size in sizes], axis=1))
    drift_selection = DriftMatrixSelector.stacked([(z.cov_size, z.cov_size, options.n_uni_eq, z.ori_size + z.sp_size,
                                                    z.ori_size + z.sp_size) for z in sizes])
    fault_vectors = None
    if n_faults:
        host_eye = np.concatenate((np.zeros((1, n_faults)), np.eye(n_faults))).astype(BackendTensor.dtype)
        fault_source = t.cat([f.fault_values_ref.T for f in faults] + [f.fault_values_rest.T for f in faults]
                             + [upload(host_eye)])
        fault_vectors = fault_source[t.cat(index[len(point_fields):])].reshape(2, -1, n_faults)

    stacked = KernelInput(
        ori_sp_matrices=OrientationSurfacePointsCoords(fields["dips_ref_coord"], fields["dips_ref_coord"],
                                                       fields["dips_rest_coord"], fields["dips_rest_coord"]),
        cartesian_selector=CartesianSelector(x_sel_hu=sel_hu, y_sel_hu=sel_hv, x_sel_hv=sel_hv, y_sel_hv=sel_hu,
                                             x_sel_h_ref=sel_huv, y_sel_h_ref=sel_huv,
                                             x_sel_h_rest=sel_huv, y_sel_h_rest=sel_huv),
        nugget_scalar=SurfacePointNuggets(**{
            name: None if any(getattr(sp, attribute) is None for sp in sps) else t.cat([getattr(sp, attribute) for sp in sps])
            for name, attribute in (("rest", "nugget_effect_rest"), ("reference", "nugget_effect_ref"),
                                    ("surface_ids", "surface_ids"))}),
        nugget_grad=t.cat([ori.nugget_effect_grad for ori in oris]),
        ori_drift=OrientationsDrift(dips_a, dips_a, dips_b, dips_b, dips_c, dips_c, second_degree_selector),
        ref_drift=PointsDrift(*(fields[f"ref_{term}"] for term in (0, 0, 1, 1, 2, 2))),
        rest_drift=PointsDrift(*(fields[f"rest_{term}"] for term in (0, 0, 1, 1, 2, 2))),
        drift_matrix_selector=drift_selection,
        ref_fault=FaultDrift(fault_vectors[0], fault_vectors[0]) if n_faults else None,
        rest_fault=FaultDrift(fault_vectors[1], fault_vectors[1]) if n_faults else None,
    )
    stacked.block_layout = BlockLayout(
        sizes=tuple(z.cov_size for z in sizes),
        nugget_grad=tuple(ori.nugget_effect_grad for ori in oris),
        nugget_scalar=tuple(SurfacePointNuggets(rest=sp.nugget_effect_rest, reference=sp.nugget_effect_ref,
                                                surface_ids=sp.surface_ids) for sp in sps),
    )

    key, start = assemblies_key(options), 0
    for solver_input, size in zip(solver_inputs, stacked.block_layout.sizes):
        cache, block = stack_cache_of(solver_input), slice(start, start + size)
        start += size
        if cache is None or key in cache.data_side:
            continue
        cache.data_side[key] = StackAssemblies(
            dips_ref_coord=fields["dips_ref_coord"][block], dips_rest_coord=fields["dips_rest_coord"][block],
            dips_ug=(dips_a[block], dips_b[block], dips_c[block], second_degree_selector[block]),
            dips_ref=tuple(fields[f"ref_{term}"][block] for term in range(3)),
            dips_rest=tuple(fields[f"rest_{term}"][block] for term in range(3)),
            faults=(fault_vectors[0][block], fault_vectors[1][block]) if n_faults else None,
        )
    return stacked


def _offsets(counts, base):
    return base + np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)


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
