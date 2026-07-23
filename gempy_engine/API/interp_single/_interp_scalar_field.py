import os
import warnings
from dataclasses import asdict
from typing import Optional, Any, Union

import numpy as np
from numpy import dtype, ndarray

import gempy_engine.config
from ...core.backend_tensor import BackendTensor
from ...core.data.exported_fields import ExportedFields
from ...core.data.internal_structs import SolverInput, SolverInput_v2, EvaluatorInput
from ...core.data.options import KernelOptions, InterpolationOptions
from ...modules.evaluator.generic_evaluator import generic_evaluator
from ...modules.evaluator.symbolic_evaluator import symbolic_evaluator
from ...modules.kernel_constructor import kernel_constructor_interface as kernel_constructor
from ...modules.kernel_constructor._kernels_assembler import create_cov_kernel
from ...modules.kernel_constructor._structs import KernelInput
from ...modules.kernel_constructor._vectors_preparation import cov_vectors_preparation
from ...modules.solver import solver_interface
from ...modules.solver.fault_drift_stabilization import stabilize_fault_drift_system
from ...modules.solver.linear_system_transform import (
    add_fault_regularization,
    equilibrate_symmetric_system,
    normalized_residual,
)
from ...modules.weights_cache.weights_cache_interface import WeightCache, generate_cache_key


def compute_weights(solver_input: Union[SolverInput, SolverInput_v2], stack_number: int, options: InterpolationOptions) -> ndarray[tuple[Any, ...], dtype[Any]]:
    weights_key = f"{options.cache_model_name}.{stack_number}"
    weights_hash = None
    cache_enabled = (
        options.cache_mode in (InterpolationOptions.CacheMode.CACHE, InterpolationOptions.CacheMode.IN_MEMORY_CACHE)
        and not BackendTensor.COMPUTE_GRADS
    )
    effective_cache_mode = (
        options.cache_mode
        if cache_enabled or options.cache_mode == InterpolationOptions.CacheMode.CLEAR_CACHE
        else InterpolationOptions.CacheMode.NO_CACHE
    )
    match effective_cache_mode:
        case InterpolationOptions.CacheMode.NO_CACHE:
            weights_cached = None
        case InterpolationOptions.CacheMode.CACHE | InterpolationOptions.CacheMode.IN_MEMORY_CACHE:
            weights_cached: Optional[dict] = WeightCache.load_weights(
                key=weights_key,
                look_in_disk=not options.cache_mode == InterpolationOptions.CacheMode.IN_MEMORY_CACHE
            )
            ts = options.temp_interpolation_values.start_computation_ts
            if ts == -1:
                warnings.warn("No start computation timestamp found. No caching.")
                weights_cached = None
                cache_enabled = False
            else:
                weights_hash = generate_cache_key(
                    name="",
                    parameters={
                            "schema": 2,
                            "ts": ts,
                            "kernel_options": _cacheable_kernel_options(options.kernel_options),
                            "solver": repr(options.kernel_options.kernel_solver),
                            "backend": repr(BackendTensor.engine_backend),
                            "dtype": str(BackendTensor.dtype),
                            "pykeops": os.getenv("PYKEOPS_SOLVER", "False") == "True",
                    }
                )
        case InterpolationOptions.CacheMode.CLEAR_CACHE:
            WeightCache.initialize_cache_dir()
            weights_cached = None
        case _:
            raise ValueError("Cache mode not recognized")

    previous_pykeops_state = BackendTensor.pykeops_enabled
    BackendTensor.pykeops_enabled = os.getenv("PYKEOPS_SOLVER", "False") == "True"
    try:
        match weights_cached:
            case None:
                weights = _solve_and_store_weights(
                    solver_input=solver_input,
                    kernel_options=options.kernel_options,
                    weights_key=weights_key,
                    weights_hash=weights_hash,
                    store=cache_enabled,
                )
            case _ if weights_cached["hash"] != weights_hash:
                weights = _solve_and_store_weights(
                    solver_input=solver_input,
                    kernel_options=options.kernel_options,
                    weights_key=weights_key,
                    weights_hash=weights_hash,
                    store=cache_enabled,
                )
            case _ if weights_cached["hash"] == weights_hash:
                weights = weights_cached["weights"]
            case _:
                raise ValueError("Something went wrong with the cache")
        return weights
    finally:
        BackendTensor.pykeops_enabled = previous_pykeops_state


def _cacheable_kernel_options(kernel_options):
    parameters = asdict(kernel_options)
    for runtime_field in ("condition_number", "condition_number_before", "condition_number_after"):
        parameters.pop(runtime_field, None)
    return parameters


def _solve_and_store_weights(solver_input, kernel_options, weights_key, weights_hash, store):
    weights = _solve_interpolation(solver_input, kernel_options)
    if store:
        WeightCache.store_weights(file_name=weights_key, hash=weights_hash, weights=weights)
    return weights


def _solve_interpolation(interp_input: SolverInput, kernel_options: KernelOptions) -> np.ndarray:
    n_faults = interp_input.fault_internal.n_faults
    requires_dense = n_faults > 0 or kernel_options.symmetric_equilibration_method != "none"
    previous_pykeops_state = BackendTensor.pykeops_enabled
    if requires_dense and previous_pykeops_state:
        warnings.warn(
            "Fault stabilization and symmetric equilibration require a dense system; disabling PyKeOps for this solve.",
            RuntimeWarning,
        )
        BackendTensor.pykeops_enabled = False

    try:
        kernel_data_tensor: KernelInput = cov_vectors_preparation(interp_input, kernel_options)
        kernel_data = kernel_data_tensor.upgrade_tensors() if BackendTensor.pykeops_enabled else kernel_data_tensor
        physical_matrix, physical_rhs, transform = _build_transformed_system(
            kernel_data, interp_input, kernel_options, n_faults
        )
        solve_matrix = transform.matrix if transform is not None else physical_matrix
        solve_rhs = transform.rhs if transform is not None else physical_rhs

        if kernel_options.optimizing_condition_number:
            _optimize_nuggets_against_condition_number(solve_matrix, interp_input, kernel_options)

        _record_condition_numbers(kernel_options, physical_matrix, solve_matrix)
        scaled_initial_guess = (
            transform.scale_initial_guess(interp_input.weights_x0)
            if transform is not None
            else interp_input.weights_x0
        )
        scaled_weights = solver_interface.kernel_reduction(
            cov=solve_matrix,
            b=solve_rhs,
            kernel_options=kernel_options,
            x0=scaled_initial_guess,
        )

        if scaled_weights is None:
            BackendTensor.pykeops_enabled = False
            physical_matrix, physical_rhs, transform = _build_transformed_system(
                kernel_data_tensor, interp_input, kernel_options, n_faults
            )
            solve_matrix = transform.matrix if transform is not None else physical_matrix
            solve_rhs = transform.rhs if transform is not None else physical_rhs
            scaled_weights = solver_interface.kernel_reduction(
                cov=solve_matrix,
                b=solve_rhs,
                kernel_options=kernel_options,
                x0=(
                    transform.scale_initial_guess(interp_input.weights_x0)
                    if transform is not None
                    else interp_input.weights_x0
                ),
            )

        weights = transform.restore_weights(scaled_weights) if transform is not None else scaled_weights
        residual = (
            normalized_residual(physical_matrix, physical_rhs, weights)
            if not BackendTensor.pykeops_enabled
            else None
        )

        if gempy_engine.config.DEBUG_MODE:
            from gempy_engine.core.data.solutions import Solutions
            Solutions.debug_input_data.update({
                "weights": weights,
                "A_matrix": physical_matrix,
                "b_vector": physical_rhs,
                "A_matrix_physical": physical_matrix,
                "b_vector_physical": physical_rhs,
                "A_matrix_scaled": solve_matrix,
                "b_vector_scaled": solve_rhs,
                "scaling_factors": transform.factors if transform is not None else None,
                "weights_scaled": scaled_weights,
                "weights_physical": weights,
                "condition_number_before": kernel_options.condition_number_before,
                "condition_number_after": kernel_options.condition_number_after,
                "normalized_residual": residual,
                "equilibration_iterations": transform.iterations if transform is not None else 0,
                "equilibration_converged": transform.converged if transform is not None else True,
            })

        return weights
    finally:
        BackendTensor.pykeops_enabled = previous_pykeops_state


def _build_transformed_system(kernel_data, interp_input, kernel_options, n_faults):
    physical_matrix = create_cov_kernel(kernel_data, kernel_options)
    physical_rhs = kernel_constructor.yield_b_vector(interp_input.ori_internal, physical_matrix.shape[0])

    if kernel_options.symmetric_equilibration_method == "ruiz":
        transform = equilibrate_symmetric_system(
            physical_matrix,
            physical_rhs,
            max_iterations=kernel_options.symmetric_equilibration_max_iterations,
            tolerance=kernel_options.symmetric_equilibration_tolerance,
        )
        transform = add_fault_regularization(
            transform,
            n_faults=n_faults,
            regularization=kernel_options.fault_drift_regularization,
        )
    elif n_faults:
        transform = stabilize_fault_drift_system(
            covariance=physical_matrix,
            b_vector=physical_rhs,
            n_faults=n_faults,
            relative_regularization=kernel_options.fault_drift_regularization,
            equilibrate=kernel_options.fault_drift_equilibration,
        )
    else:
        transform = None
    return physical_matrix, physical_rhs, transform


def _record_condition_numbers(kernel_options, physical_matrix, scaled_matrix):
    if not kernel_options.compute_condition_number or BackendTensor.pykeops_enabled:
        return
    before = BackendTensor.t.linalg.cond(physical_matrix)
    after = BackendTensor.t.linalg.cond(scaled_matrix)
    kernel_options.condition_number_before = before
    kernel_options.condition_number_after = after
    kernel_options.condition_number = after


def _optimize_nuggets_against_condition_number(A_matrix, interp_input, kernel_options):
    from ...core.data.continue_epoch import ContinueEpoch
    import torch
    cond_number = BackendTensor.t.linalg.cond(A_matrix)
    nuggets = interp_input.sp_internal.nugget_effect_ref_rest
    l1_reg = torch.norm(nuggets, 2) ** 2
    lambda_l1 = 100_000_000
    loss = cond_number - lambda_l1 * l1_reg
    loss.backward()
    kernel_options.condition_number = cond_number
    print(f'Condition number: {cond_number}.')
    raise ContinueEpoch()


def _evaluate_sys_eq(eval_input: Union[SolverInput, EvaluatorInput], weights: np.ndarray, options: InterpolationOptions) -> ExportedFields:
    BackendTensor.pykeops_enabled = BackendTensor.use_pykeops
    if BackendTensor.pykeops_enabled:
        exported_fields = symbolic_evaluator(eval_input, weights, options)
    else:
        exported_fields = generic_evaluator(eval_input, weights, options)

    return exported_fields
