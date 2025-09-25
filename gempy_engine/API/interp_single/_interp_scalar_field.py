import warnings

from typing import Tuple, Optional

import numpy as np

import gempy_engine.config
from ...core.backend_tensor import BackendTensor
from ...core.data.exported_fields import ExportedFields
from ...core.data.internal_structs import SolverInput
from ...core.data.options import KernelOptions, InterpolationOptions
from ...modules.evaluator.generic_evaluator import generic_evaluator
from ...modules.evaluator.symbolic_evaluator import symbolic_evaluator

from ...modules.kernel_constructor import kernel_constructor_interface as kernel_constructor
from ...modules.solver import solver_interface
from ...modules.weights_cache.weights_cache_interface import WeightCache, generate_cache_key



def interpolate_scalar_field(solver_input: SolverInput, options: InterpolationOptions, stack_number: int) -> Tuple[np.ndarray, ExportedFields]:
    # region Solver

    weights_key = f"{options.cache_model_name}.{stack_number}"
    weights_hash = None
    match options.cache_mode:
        case InterpolationOptions.CacheMode.NO_CACHE:
            weights_cached = None
        case InterpolationOptions.CacheMode.CACHE | InterpolationOptions.CacheMode.IN_MEMORY_CACHE:
            weights_cached: Optional[dict] = WeightCache.load_weights(
                key=weights_key,
                look_in_disk= not options.cache_mode == InterpolationOptions.CacheMode.IN_MEMORY_CACHE
            )
            ts = options.temp_interpolation_values.start_computation_ts
            if ts == -1:
                warnings.warn("No start computation timestamp found. No caching.")
                weights_cached = None
            else:
                weights_hash = generate_cache_key(
                    name="",
                    parameters={
                            "ts": ts
                    }
                )
        case  InterpolationOptions.CacheMode.CLEAR_CACHE:
            WeightCache.initialize_cache_dir()
            weights_cached = None
        case _:
            raise ValueError("Cache mode not recognized")

    
    BackendTensor.pykeops_enabled = False
    match weights_cached:
        case None :
            weights = _solve_and_store_weights(
                solver_input=solver_input,
                kernel_options=options.kernel_options,
                weights_key=weights_key,
                weights_hash=weights_hash
            )
        case _ if weights_cached["hash"] != weights_hash:
            weights = _solve_and_store_weights(
                solver_input=solver_input,
                kernel_options=options.kernel_options,
                weights_key=weights_key,
                weights_hash=weights_hash
            )
        case _ if weights_cached["hash"] == weights_hash:
            weights = weights_cached["weights"]
        case _:
            raise ValueError("Something went wrong with the cache")

    # endregion

    BackendTensor.pykeops_enabled = BackendTensor.use_pykeops
    exported_fields: ExportedFields = _evaluate_sys_eq(solver_input, weights, options)

    return weights, exported_fields


def _solve_and_store_weights(solver_input, kernel_options, weights_key, weights_hash):
    weights = _solve_interpolation(solver_input, kernel_options)
    WeightCache.store_weights(file_name=weights_key, hash=weights_hash, weights=weights)
    return weights


def _solve_interpolation(interp_input: SolverInput, kernel_options: KernelOptions) -> np.ndarray:
    A_matrix = kernel_constructor.yield_covariance(interp_input, kernel_options)
    b_vector = kernel_constructor.yield_b_vector(interp_input.ori_internal, A_matrix.shape[0])

    if kernel_options.optimizing_condition_number:
        _optimize_nuggets_against_condition_number(A_matrix, interp_input, kernel_options)

    weights = solver_interface.kernel_reduction(
        cov=A_matrix,
        b=b_vector,
        kernel_options=kernel_options,
        x0=interp_input.weights_x0
    )

    if gempy_engine.config.DEBUG_MODE:
        # Save debug data for later
        from gempy_engine.core.data.solutions import Solutions
        Solutions.debug_input_data["weights"] = weights
        Solutions.debug_input_data["A_matrix"] = A_matrix
        Solutions.debug_input_data["b_vector"] = b_vector

        # Check matrices have the right dtype:
        if False:
            assert A_matrix.dtype == BackendTensor.dtype_obj, f"Wrong dtype for A_matrix: {A_matrix.dtype}. should be {BackendTensor.dtype_obj}"
            assert b_vector.dtype == BackendTensor.dtype_obj, f"Wrong dtype for b_vector: {b_vector.dtype}. should be {BackendTensor.dtype_obj}"
            assert weights.dtype == BackendTensor.dtype_obj, f"Wrong dtype for weights: {weights.dtype}. should be {BackendTensor.dtype_obj}"

    return weights


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


def _evaluate_sys_eq(solver_input: SolverInput, weights: np.ndarray, options: InterpolationOptions) -> ExportedFields:
    if BackendTensor.pykeops_enabled is True:
        exported_fields = symbolic_evaluator(solver_input, weights, options)
    else:
        exported_fields = generic_evaluator(solver_input, weights, options)

    return exported_fields
