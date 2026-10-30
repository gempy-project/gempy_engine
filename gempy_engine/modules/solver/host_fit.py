"""Dense fits of small interpolation systems on the host, through the native CPU covariance assembler.

Prototype, off by default: ``GEMPY_HOST_FIT`` set to ``float64`` or ``float32`` selects the native module
``_native_cpu`` or ``_native_cpu_float32`` (the experimental native backend's pybind module, imported from
``sys.path``) and the host precision. The systems are tiny (Perth: 11 to 99 rows), so the whole chunk is
read back once, assembled with ``macro_matrix``, stabilized and solved in NumPy, and its weights uploaded
in one copy, instead of about a thousand small device ops per compute.

Mirrors the torch path (``interpolation_solver``): the same rows (component-major orientations, rest
points against their surface's first point, drift, faults), nuggets scaled by ``c_o``, fault block,
fault-drift stabilization or Ruiz equilibration and negative fault loading. torch's ``median`` (the lower
middle value) is kept for the stabilization. Results agree with it within float tolerance, not bitwise.
"""
import os
from importlib import import_module

import numpy as np

from ...config import AvailableBackends
from ...core.backend_tensor import BackendTensor
from ...core.host_arrays import upload
from ...core.data.options import NuggetImplementation

_DTYPES = {"float64": np.float64, "float32": np.float32}
_KERNELS = ("cubic", "exponential", "matern_5_2")
MAX_HOST_FIT_SIZE = 512
_contexts = {}


def host_fit_dtype():
    """The host precision when the host fit is enabled and its native module imports, else None."""
    name = os.getenv("GEMPY_HOST_FIT", "off").lower()
    if name not in _DTYPES or BackendTensor.engine_backend is not AvailableBackends.PYTORCH:
        return None
    return _DTYPES[name] if _context(name) is not None else None


def _context(name):
    if name not in _contexts:
        try:
            module = import_module("_native_cpu" + ("_float32" if name == "float32" else ""))
            _contexts[name] = module.Context(0)
        except ImportError:
            _contexts[name] = None
    return _contexts[name]


def eligible(solver_input, kernel_options, tensor_struct) -> bool:
    import gempy_engine.config as config
    from .interpolation_solver import pykeops_solver_requested
    if (BackendTensor.COMPUTE_GRADS or config.DEBUG_MODE or pykeops_solver_requested()
            or kernel_options.optimizing_condition_number or kernel_options.number_dimensions != 3
            or kernel_options.kernel_function.name not in _KERNELS or kernel_options.uni_degree not in (0, 1, 2)
            or getattr(tensor_struct, "host_counts", None) is None):
        return False
    return _system_size(solver_input, kernel_options) <= MAX_HOST_FIT_SIZE


def _system_size(solver_input, kernel_options):
    drift = {0: 0, 1: 3, 2: 9}[kernel_options.uni_degree]
    return (3 * solver_input.ori_internal.orientations.dip_positions.shape[0]
            + solver_input.sp_internal.rest_surface_points.shape[0] + drift + solver_input.fault_internal.n_faults)


def fit_chunk(members, dtype) -> list:
    """Weights ``(size,)`` in the engine dtype on its device, per member ``(solver_input, interpolation_input,
    kernel_options, tensor_struct)``: one read of all inputs, one upload of all weights."""
    import torch
    parts = []
    for solver_input, interpolation_input, _, _ in members:
        sp, ori = interpolation_input.surface_points, solver_input.ori_internal.orientations
        faults = solver_input.fault_internal.fault_values_on_sp
        parts += [sp.sp_coords, sp.nugget_effect_scalar, ori.dip_positions, ori.dip_gradients, ori.nugget_effect_grad]
        if solver_input.fault_internal.n_faults:
            parts.append(faults)
    host = torch.cat([part.detach().reshape(-1).to(BackendTensor.dtype_obj) for part in parts]).cpu().numpy()
    host = host.astype(dtype)
    offset, weights = 0, []

    def take(shape):
        nonlocal offset
        size = int(np.prod(shape))
        offset += size
        return host[offset - size:offset].reshape(shape)

    context = _context("float32" if dtype == np.float32 else "float64")
    for solver_input, interpolation_input, kernel_options, tensor_struct in members:
        n, m = interpolation_input.surface_points.sp_coords.shape[0], solver_input.ori_internal.orientations.dip_positions.shape[0]
        points, point_nuggets, positions, gradients = take((n, 3)), take((n,)), take((m, 3)), take((m, 3))
        orientation_nuggets = take((m,))
        n_faults = solver_input.fault_internal.n_faults
        fault_values = take((n_faults, n)) if n_faults else np.zeros((0, n), dtype)
        weights.append(_fit(context, kernel_options, tensor_struct.host_counts.astype(np.int64), points, point_nuggets,
                            positions, gradients, orientation_nuggets, fault_values, dtype))
    sizes = [len(w) for w in weights]
    device = upload(np.concatenate(weights).astype(np.float32 if BackendTensor.dtype_obj == torch.float32 else np.float64))
    return list(device.split(sizes))


def _fit(context, options, counts, points, point_nuggets, positions, gradients, orientation_nuggets, fault_values, dtype):
    m, n_faults = len(positions), len(fault_values)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    rest = np.setdiff1d(np.arange(counts.sum()), starts)
    reference = np.repeat(starts, counts - 1)
    surface_ids = np.repeat(np.arange(len(counts)), counts - 1)
    drift = {0: [], 1: [4, 5, 6], 2: list(range(4, 13))}[options.uni_degree]
    kinds = np.array([0] * m + [1] * m + [2] * m + [3] * len(rest) + drift + [13] * n_faults, dtype=np.int32)
    size, rows = len(kinds), slice(3 * m, 3 * m + len(rest))
    fault_start = size - n_faults
    coordinates = np.zeros((size, 6), dtype=dtype)
    coordinates[:3 * m, :3] = np.tile(positions, (3, 1))
    coordinates[rows, :3], coordinates[rows, 3:] = points[reference], points[rest]
    parameters = np.array([options.range, options.c_o, options.i_res, options.gi_res], dtype=dtype)
    kernel = options.kernel_function.name
    matrix = context.macro_matrix(coordinates, kinds, coordinates, kinds, parameters, kernel)

    nuggets = np.zeros(size, dtype=dtype)
    implementation = options.nugget_implementation
    nuggets[:3 * m] = orientation_nuggets[0] if implementation is NuggetImplementation.LEGACY else np.tile(orientation_nuggets, 3)
    nuggets[rows] = point_nuggets[rest]
    if implementation is NuggetImplementation.DIAGONAL_REF_REST:
        nuggets[rows] += point_nuggets[reference]
    matrix[np.diag_indices(size)] += options.c_o * nuggets
    if implementation is NuggetImplementation.FULL_POINT_COVARIANCE:
        matrix[rows, rows] += options.c_o * point_nuggets[reference, None] * (surface_ids[:, None] == surface_ids[None, :])
    if n_faults:
        block = np.full((fault_start, n_faults), 1e-8, dtype=dtype)
        block[rows] += (fault_values[:, reference] - fault_values[:, rest]).T
        matrix[:fault_start, fault_start:], matrix[fault_start:, :fault_start] = block, block.T

    rhs = np.zeros(size, dtype=dtype)
    rhs[:3 * m] = gradients.T.ravel()
    effective, scale = _stabilized(matrix, options, fault_start, n_faults)
    if options.compute_condition_number:
        options.condition_number_before = float(np.linalg.cond(matrix))
        options.condition_number_after = options.condition_number = float(np.linalg.cond(effective))
    return scale * np.linalg.solve(effective, scale * rhs)


def _lower_median(values):
    return np.sort(values)[(len(values) - 1) // 2]


def _stabilized(matrix, options, fault_start, n_faults):
    """The solved matrix and the symmetric scaling, as ``stabilize_fault_drift_system`` or
    ``equilibrate_symmetric_system`` plus ``add_fault_regularization`` build them."""
    scale = np.ones(len(matrix), dtype=matrix.dtype)
    effective = matrix.copy()
    loading = 1.0
    if options.symmetric_equilibration_method == "ruiz":
        for _ in range(options.symmetric_equilibration_max_iterations):
            norms = np.max(np.abs(effective), axis=1)
            step = 1 / np.sqrt(np.where(norms > 0, norms, 1))
            scale *= step
            effective *= step[:, None] * step[None, :]
            norms = np.max(np.abs(effective), axis=1)
            if np.max(np.abs(norms[norms > 0] - 1), initial=0) <= options.symmetric_equilibration_tolerance:
                break
        loading = options.fault_drift_regularization
    elif n_faults:
        non_fault = matrix[:fault_start, :fault_start]
        norms = np.linalg.norm(non_fault, axis=0)
        reference = _lower_median(norms[norms > 0]) if np.any(norms > 0) else 1.0
        if options.fault_drift_equilibration:
            fault_norms = np.linalg.norm(matrix[:fault_start, fault_start:], axis=0)
            scale[fault_start:] = np.where(fault_norms > 0, reference / np.where(fault_norms > 0, fault_norms, 1), 1)
        diagonal = np.abs(np.diag(non_fault))
        loading = options.fault_drift_regularization * (
            _lower_median(diagonal[diagonal > 0]) if np.any(diagonal > 0) else reference)
        effective *= scale[:, None] * scale[None, :]
    if n_faults:
        effective[np.arange(fault_start, len(matrix)), np.arange(fault_start, len(matrix))] -= loading
    return effective, scale
