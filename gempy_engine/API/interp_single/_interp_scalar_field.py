from typing import Any, Union
from copy import copy

import numpy as np
from numpy import dtype, ndarray

import gempy_engine.config
from ...core.backend_tensor import BackendTensor
from ...core.data.engine_grid import EngineGrid
from ...core.data.exported_fields import ExportedFields
from ...core.data.internal_structs import SolverInput, SolverInput_v2, EvaluatorInput
from ...core.data.options import KernelOptions, InterpolationOptions
from ...modules.evaluator.generic_evaluator import generic_evaluator
from ...modules.evaluator.symbolic_evaluator import symbolic_evaluator
from ...modules.kernel_constructor.drift_design import (
    analyze_drift_design,
    build_drift_design,
    enforce_rank_policy,
)
from ...modules.solver.interpolation_solver import (
    InterpolationSolveRoute,
    assemble_solve_debug,
    pykeops_solver_requested,
    select_interpolation_solve_route,
    solve_dense_fault_stabilized,
    solve_dense_ruiz,
    solve_dense_untransformed,
    solve_pykeops_with_dense_fallback, InterpolationSolveResult,
)
from ...modules.weights_cache.weight_cache_policy import (
    WeightCacheRoute,
    resolve_weight_cache,
    store_weight_result,
)


def compute_weights(solver_input: Union[SolverInput, SolverInput_v2], stack_number: int, options: InterpolationOptions) \
        -> ndarray[tuple[Any, ...], dtype[Any]]:
    if options.kernel_options.drift_diagnostics:
        drift_design = build_drift_design(solver_input, options.kernel_options)
        drift_report = analyze_drift_design(drift_design, options.kernel_options.drift_rank_rcond)
        solver_input.debug["drift_diagnostics"] = drift_report
        enforce_rank_policy(
            drift_report,
            options.kernel_options.drift_rank_policy,
            stack_number,
            options.kernel_options.drift_warning_rcond,
        )
    pykeops_requested = pykeops_solver_requested()
    cache_decision = resolve_weight_cache(
        options=options,
        stack_number=stack_number,
        pykeops_requested=pykeops_requested,
        solver_input=solver_input,
    )
    match cache_decision.route:
        case WeightCacheRoute.CACHED:
            return cache_decision.weights
        case WeightCacheRoute.SOLVE:
            return _solve_interpolation(solver_input, options.kernel_options, pykeops_requested)
        case WeightCacheRoute.SOLVE_AND_STORE:
            result: InterpolationSolveResult = _solve_interpolation_result(solver_input, options.kernel_options, pykeops_requested)
            if not result.used_fallback:
                store_weight_result(cache_decision, result.weights)
            return result.weights


def _solve_interpolation(
        interp_input: Union[SolverInput, SolverInput_v2],
        kernel_options: KernelOptions,
        pykeops_requested: bool | None = None,
) -> np.ndarray:
    result: InterpolationSolveResult = _solve_interpolation_result(interp_input, kernel_options, pykeops_requested)
    return result.weights


def _solve_interpolation_result(
        interp_input: Union[SolverInput, SolverInput_v2],
        kernel_options: KernelOptions,
        pykeops_requested: bool | None = None,
) -> InterpolationSolveResult:
    pykeops_requested = pykeops_solver_requested() if pykeops_requested is None else pykeops_requested
    route = select_interpolation_solve_route(
        kernel_options=kernel_options,
        n_faults=interp_input.fault_internal.n_faults,
        pykeops_requested=pykeops_requested,
    )
    match route:
        case InterpolationSolveRoute.DENSE_UNTRANSFORMED:
            result = solve_dense_untransformed(interp_input, kernel_options)
        case InterpolationSolveRoute.DENSE_FAULT_STABILIZED:
            result = solve_dense_fault_stabilized(interp_input, kernel_options)
        case InterpolationSolveRoute.DENSE_RUIZ:
            result = solve_dense_ruiz(interp_input, kernel_options)
        case InterpolationSolveRoute.PYKEOPS_WITH_DENSE_FALLBACK:
            result = solve_pykeops_with_dense_fallback(interp_input, kernel_options)

    if gempy_engine.config.DEBUG_MODE:
        from gempy_engine.core.data.solutions import Solutions
        Solutions.debug_input_data.update(assemble_solve_debug(result, kernel_options))
    return result


def _evaluate_sys_eq(eval_input: Union[SolverInput, EvaluatorInput], weights: np.ndarray, options: InterpolationOptions,
                     grid: EngineGrid | None = None) -> ExportedFields:
    inverse = None
    if options.evaluation_options.deduplicate_octree_corners:
        eval_input, inverse = _deduplicate_corners(eval_input, grid)
    if BackendTensor.use_pykeops:
        exported_fields = symbolic_evaluator(eval_input, weights, options)
    else:
        exported_fields = generic_evaluator(eval_input, weights, options)

    return _restore_corner_fields(exported_fields, inverse)


def _restore_corner_fields(exported_fields: ExportedFields, inverse) -> ExportedFields:
    """Expand a reduced evaluation before attaching original grid metadata."""
    if inverse is not None:
        for name in ('_scalar_field', '_gx_field', '_gy_field', '_gz_field'):
            values = getattr(exported_fields, name)
            if values is not None:
                index = BackendTensor.t.to_numpy(inverse) if isinstance(values, np.ndarray) else inverse
                setattr(exported_fields, name, values[index])
    return exported_fields


def _deduplicate_corners(eval_input: SolverInput | EvaluatorInput, grid: EngineGrid | None):
    """Call-local evaluation view; never change grid layout or surface-point metadata."""
    if grid is None or grid.octree_grid is None or grid.corners_grid is None:
        return eval_input, None
    corners = grid.corners_grid.values
    # Separate coordinate/fault row derivatives must not be redirected to a representative.
    if len(corners) == 0 or getattr(corners, 'requires_grad', False):
        return eval_input, None
    faults = eval_input.fault_internal
    fault_values = faults.fault_values_everywhere if faults.n_faults else None
    if getattr(fault_values, 'requires_grad', False):
        return eval_input, None

    t = BackendTensor.t
    offsets = t.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype='int64')
    coordinates = (grid.octree_grid.integer_coordinates[:, None, :] + offsets).reshape(-1, 3)
    if len(coordinates) != len(corners):
        return eval_input, None
    if BackendTensor.engine_backend == gempy_engine.config.AvailableBackends.PYTORCH:
        import torch
        unique, inverse = torch.unique(coordinates, dim=0, return_inverse=True)
        first = torch.full((len(unique),), len(corners), dtype=torch.int64, device=coordinates.device)
        first.scatter_reduce_(0, inverse, torch.arange(len(corners), device=coordinates.device), reduce='amin')
    else:
        _, first, inverse = np.unique(coordinates, axis=0, return_index=True, return_inverse=True)
    if len(first) == len(corners):
        return eval_input, None

    start, stop = grid.corners_grid_slice.start, grid.corners_grid_slice.stop
    xyz = eval_input.xyz_to_interpolate
    # Use original physical rows, not extent + lattice * spacing: refined extents
    # may carry a different origin shift. Reject non-lattice/custom corner layouts.
    tolerance = 32 * np.finfo(BackendTensor.dtype).eps
    if not t.allclose(xyz[start:stop], xyz[start + first][inverse], rtol=tolerance, atol=tolerance):
        return eval_input, None
    if fault_values is not None:
        if not t.all(fault_values[:, start:stop] == fault_values[:, start + first][:, inverse]):
            return eval_input, None

    before = BackendTensor.arange(start, dtype='int64')
    after = stop + BackendTensor.arange(len(xyz) - stop, dtype='int64')
    keep = t.concatenate((before, start + first, after))
    restore = t.concatenate((before, start + inverse,
                             start + len(first) + BackendTensor.arange(len(xyz) - stop, dtype='int64')))
    reduced = copy(eval_input)
    reduced.xyz_to_interpolate = xyz[keep]
    if fault_values is not None:
        reduced_faults = copy(faults)
        reduced_faults.fault_values_everywhere = fault_values[:, keep]
        if isinstance(reduced, EvaluatorInput):
            reduced.solver_input = copy(reduced.solver_input)
            reduced.solver_input.fault_internal = reduced_faults
        else:
            reduced.fault_internal = reduced_faults
    return reduced, restore
