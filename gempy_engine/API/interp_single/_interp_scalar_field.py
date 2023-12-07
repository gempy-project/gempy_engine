from typing import Tuple

import numpy as np

import gempy_engine.config
from ...core.backend_tensor import BackendTensor
from ...core.data.exported_fields import ExportedFields
from ...core.data.internal_structs import SolverInput
from ...core.data.options import KernelOptions, InterpolationOptions

from ...modules.kernel_constructor import kernel_constructor_interface as kernel_constructor
from ...modules.solver import solver_interface
from ...modules.weights_cache.weights_cache_interface import WeightCache, generate_cache_key


class WeightsBuffer:
    weights: dict[hash, np.ndarray] = {}

    @classmethod
    def add(cls, value: np.ndarray, solver_input: SolverInput, kernel_options: KernelOptions):
        input_hash = hash((solver_input, kernel_options))  # ! This seems more expensive that compute the weights!
        cls.weights[input_hash] = value

    @classmethod
    def clean(cls):
        cls.weights = {}

    @classmethod
    def get(cls, solver_input: SolverInput, kernel_options: KernelOptions):
        input_hash = hash((solver_input, kernel_options))
        current_weights = cls.weights.get(input_hash, None)
        return current_weights


def interpolate_scalar_field(solver_input: SolverInput, options: InterpolationOptions, stack_number: int) -> Tuple[np.ndarray, ExportedFields]:
    # region Solver

    model_name = options._model_name
    weights_key = f"{model_name}.{stack_number}"
    
    weights_hash = generate_cache_key(
        name="",
        parameters={
            "surface_points": solver_input.sp_internal,
            "orientations": solver_input.ori_internal,
            "fault_internal": solver_input._fault_internal.fault_values_on_sp,
            "kernel_options": options.kernel_options
        }
    )
    
    # TODO: If the weights_key are exactly the same we can skip the interpolation all along. Otherwise if the shape is
    #  the same we can use the weights as initial guess
    # TODO: Pass model name + group at least number in solver_input?
    
    weights_from_cache = WeightCache.load_weights(weights_key)

    # TODO: Check if the weights are the same and if the shape is the same
    hash_is_equals = weights_from_cache["hash"] == weights_hash
    if weights_from_cache is None:
        weights = _solve_interpolation(solver_input, options.kernel_options)
        
        WeightCache.store_weights(
            file_name=weights_key,
            hash=weights_hash,
            weights= weights
        )
    elif hash_is_equals:
        weights = weights_from_cache["weights"]
    else:
        solver_input.weights_x0 = weights_from_cache["weights"]
        weights = _solve_interpolation(solver_input, options.kernel_options)

    # endregion

    exported_fields: ExportedFields = _evaluate_sys_eq(solver_input, weights, options)

    return weights, exported_fields


def _solve_interpolation(interp_input: SolverInput, kernel_options: KernelOptions) -> np.ndarray:
    A_matrix = kernel_constructor.yield_covariance(interp_input, kernel_options)
    b_vector = kernel_constructor.yield_b_vector(interp_input.ori_internal, A_matrix.shape[0])

    if kernel_options.optimizing_condition_number:  
        _optimize_nuggets_against_condition_number(A_matrix, interp_input, kernel_options)

    # TODO: Smooth should be taken from options
    weights = solver_interface.kernel_reduction(
        cov=A_matrix,
        b=b_vector,
        kernel_options=kernel_options
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
    lambda_l1 = 100000000
    loss = cond_number - lambda_l1 * l1_reg
    loss.backward()
    kernel_options.condition_number = cond_number
    print(f'Condition number: {cond_number}.')
    raise ContinueEpoch()


def _evaluate_sys_eq(solver_input: SolverInput, weights: np.ndarray, options: InterpolationOptions) -> ExportedFields:
    compute_gradient: bool = options.compute_scalar_gradient

    eval_kernel = kernel_constructor.yield_evaluation_kernel(solver_input, options.kernel_options)

    if BackendTensor.pykeops_enabled is True and BackendTensor.engine_backend == gempy_engine.config.AvailableBackends.numpy:
        if solver_input.xyz_to_interpolate.flags['C_CONTIGUOUS'] is False:  # ! This is not working with TF yet
            print("xyz is not C_CONTIGUOUS")

        from pykeops.numpy import LazyTensor
        # ! Seems not to make any difference but we need this if we want to change the backend
        # ! We need to benchmark GPU vs CPU with more input
        backend_string = BackendTensor.get_backend_string()
        scalar_field = (eval_kernel.T * LazyTensor(np.asfortranarray(weights), axis=1)).sum(axis=1, backend=backend_string).reshape(-1)

        if compute_gradient is True:
            eval_gx_kernel = kernel_constructor.yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=0)
            eval_gy_kernel = kernel_constructor.yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=1)

            gx_field = (eval_gx_kernel.T * LazyTensor(weights, axis=1)).sum(axis=1, backend=backend_string).reshape(-1)
            gy_field = (eval_gy_kernel.T * LazyTensor(weights, axis=1)).sum(axis=1, backend=backend_string).reshape(-1)

            if options.number_dimensions == 3:
                eval_gz_kernel = kernel_constructor.yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=2)
                gz_field = (eval_gz_kernel.T * LazyTensor(weights, axis=1)).sum(axis=1, backend=backend_string).reshape(-1)
            elif options.number_dimensions == 2:
                gz_field = None
            else:
                raise ValueError("Number of dimensions have to be 2 or 3")

            exported_fields = ExportedFields(scalar_field, gx_field, gy_field, gz_field)
        else:
            exported_fields = ExportedFields(scalar_field, None, None, None)
    else:
        scalar_field = (eval_kernel.T @ weights).reshape(-1)

        if compute_gradient is True:
            eval_gx_kernel = kernel_constructor.yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=0)
            eval_gy_kernel = kernel_constructor.yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=1)
            gx_field = (eval_gx_kernel.T @ weights).reshape(-1)
            gy_field = (eval_gy_kernel.T @ weights).reshape(-1)

            if options.number_dimensions == 3:
                eval_gz_kernel = kernel_constructor.yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=2)
                gz_field = (eval_gz_kernel.T @ weights).reshape(-1)
            elif options.number_dimensions == 2:
                gz_field = None
            else:
                raise ValueError("Number of dimensions have to be 2 or 3")
            exported_fields = ExportedFields(scalar_field, gx_field, gy_field, gz_field)
        else:
            exported_fields = ExportedFields(scalar_field, None, None, None)

    return exported_fields
