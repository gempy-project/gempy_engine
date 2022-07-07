from typing import Tuple

import numpy as np

from ...core.backend_tensor import BackendTensor
from ...core.data.exported_fields import ExportedFields
from ...core.data.internal_structs import SolverInput
from ...core.data.options import KernelOptions, InterpolationOptions

from ...modules.kernel_constructor import kernel_constructor_interface as kernel_constructor
from ...modules.solver import solver_interface


class Buffer:
    weights = None

    @classmethod
    def clean(cls):
        cls.weights = None


def interpolate_scalar_field(solver_input: SolverInput,
                             options: InterpolationOptions) -> Tuple[np.ndarray, ExportedFields]:
    # region Solver
    if Buffer.weights is None:
        weights = _solve_interpolation(solver_input, options.kernel_options)
        Buffer.weights = weights
    else:
        weights = Buffer.weights

    # endregion

    exported_fields: ExportedFields = _evaluate_sys_eq(solver_input, weights, options)

    Buffer.clean()
    return weights, exported_fields


def _solve_interpolation(interp_input: SolverInput, kernel_options: KernelOptions) -> np.ndarray:
    A_matrix = kernel_constructor.yield_covariance(interp_input, kernel_options)
    b_vector = kernel_constructor.yield_b_vector(interp_input.ori_internal, A_matrix.shape[0])
    # TODO: Smooth should be taken from options
    weights = solver_interface.kernel_reduction(A_matrix, b_vector, smooth=0.01)
    return weights


def _evaluate_sys_eq(solver_input: SolverInput, weights: np.ndarray, options: InterpolationOptions) -> ExportedFields:
    if solver_input.xyz_to_interpolate.flags['C_CONTIGUOUS'] is False:
        print("xyz is not C_CONTIGUOUS")

    compute_gradient: bool = options.compute_scalar_gradient

    eval_kernel = kernel_constructor.yield_evaluation_kernel(solver_input, options.kernel_options)

    if BackendTensor.pykeops_enabled is True:
        from pykeops.numpy import LazyTensor
        # ! Seems not to make any difference but we need this if we want to change the backend
        # ! We need to benchmark GPU vs CPU with more input
        scalar_field = (eval_kernel.T * LazyTensor(np.asfortranarray(weights), axis=1)).sum(axis=1, backend="GPU").reshape(-1)

        if compute_gradient is True:
            eval_gx_kernel = kernel_constructor.yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=0)
            eval_gy_kernel = kernel_constructor.yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=1)

            gx_field = (eval_gx_kernel.T * LazyTensor(weights, axis=1)).sum(axis=1, backend="GPU").reshape(-1)
            gy_field = (eval_gy_kernel.T * LazyTensor(weights, axis=1)).sum(axis=1, backend="GPU").reshape(-1)

            if options.number_dimensions == 3:
                eval_gz_kernel = kernel_constructor.yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=2)
                gz_field = (eval_gz_kernel.T * LazyTensor(weights, axis=1)).sum(axis=1, backend="GPU").reshape(-1)
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
