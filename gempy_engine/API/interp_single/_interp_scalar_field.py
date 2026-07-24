from typing import Any, Union

import numpy as np
from numpy import dtype, ndarray

import gempy_engine.config
from ...core.backend_tensor import BackendTensor
from ...core.data.exported_fields import ExportedFields
from ...core.data.internal_structs import SolverInput, SolverInput_v2, EvaluatorInput
from ...core.data.options import KernelOptions, InterpolationOptions
from ...modules.evaluator.generic_evaluator import generic_evaluator
from ...modules.evaluator.symbolic_evaluator import symbolic_evaluator
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


def _evaluate_sys_eq(eval_input: Union[SolverInput, EvaluatorInput], weights: np.ndarray, options: InterpolationOptions) -> ExportedFields:
    if BackendTensor.use_pykeops:
        exported_fields = symbolic_evaluator(eval_input, weights, options)
    else:
        exported_fields = generic_evaluator(eval_input, weights, options)

    return exported_fields
