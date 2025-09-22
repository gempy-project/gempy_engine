from typing import Optional

import numpy as np

import gempy_engine
from ...core.backend_tensor import BackendTensor
from ...core.data import InterpolationOptions
from ...core.data.exported_fields import ExportedFields
from ...core.data.internal_structs import SolverInput
from ..kernel_constructor.kernel_constructor_interface import yield_evaluation_grad_kernel, yield_evaluation_kernel


def symbolic_evaluator(solver_input: SolverInput, weights: np.ndarray, options: InterpolationOptions):
    
    if BackendTensor.engine_backend == gempy_engine.config.AvailableBackends.numpy and solver_input.xyz_to_interpolate.flags['C_CONTIGUOUS'] is False:  # ! This is not working with TF yet
        print("xyz is not C_CONTIGUOUS")
    # ! Seems not to make any difference but we need this if we want to change the backend
    # ! We need to benchmark GPU vs CPU with more input
    backend_string = BackendTensor.get_backend_string()

    eval_kernel = yield_evaluation_kernel(solver_input, options.kernel_options)
    if BackendTensor.engine_backend == gempy_engine.config.AvailableBackends.numpy:
        from pykeops.numpy import LazyTensor
        # Create lazy_weights with correct dimensions: we want (16, 1) to match eval_kernel's nj dimension
        lazy_weights = LazyTensor(np.asfortranarray(weights.reshape(-1, 1)), axis=0)  # axis=0 means this is the 'i' dimension
        scalar_field: np.ndarray = (eval_kernel * lazy_weights).sum(axis=0, backend=backend_string).reshape(-1)
    else:
        from pykeops.torch import LazyTensor
        lazy_weights = LazyTensor(weights.view((-1, 1)), axis=0)  # axis=0 for 'i' dimension
        # Use element-wise multiplication and sum over the correct axis
        scalar_field: np.ndarray = (eval_kernel * lazy_weights).sum(axis=0, backend=backend_string).reshape(-1)

    gx_field: Optional[np.ndarray] = None
    gy_field: Optional[np.ndarray] = None
    gz_field: Optional[np.ndarray] = None
    
    if options.compute_scalar_gradient is True:
        eval_gx_kernel = yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=0)
        eval_gy_kernel = yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=1)

        gx_field = (eval_gx_kernel * lazy_weights).sum(axis=0, backend=backend_string).reshape(-1)
        gy_field = (eval_gy_kernel * lazy_weights).sum(axis=0, backend=backend_string).reshape(-1)

        if options.number_dimensions == 3:
            eval_gz_kernel = yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=2)
            gz_field = (eval_gz_kernel * lazy_weights).sum(axis=0, backend=backend_string).reshape(-1)
        elif options.number_dimensions == 2:
            gz_field = None
        else:
            raise ValueError("Number of dimensions have to be 2 or 3")

    return ExportedFields(scalar_field, gx_field, gy_field, gz_field)
