from typing import Optional

import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_evaluation_grad_kernel


def symbolic_evaluator(compute_gradient, eval_kernel, options, solver_input, weights):
    if solver_input.xyz_to_interpolate.flags['C_CONTIGUOUS'] is False:  # ! This is not working with TF yet
        print("xyz is not C_CONTIGUOUS")
    from pykeops.numpy import LazyTensor
    # ! Seems not to make any difference but we need this if we want to change the backend
    # ! We need to benchmark GPU vs CPU with more input
    backend_string = BackendTensor.get_backend_string()
    scalar_field: np.ndarray = (eval_kernel.T * LazyTensor(np.asfortranarray(weights), axis=1)).sum(axis=1, backend=backend_string).reshape(-1)
    gx_field: Optional[np.ndarray] = None
    gy_field: Optional[np.ndarray] = None
    gz_field: Optional[np.ndarray] = None
    
    if compute_gradient is True:
        eval_gx_kernel = yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=0)
        eval_gy_kernel = yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=1)

        gx_field = (eval_gx_kernel.T * LazyTensor(weights, axis=1)).sum(axis=1, backend=backend_string).reshape(-1)
        gy_field = (eval_gy_kernel.T * LazyTensor(weights, axis=1)).sum(axis=1, backend=backend_string).reshape(-1)

        if options.number_dimensions == 3:
            eval_gz_kernel = yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=2)
            gz_field = (eval_gz_kernel.T * LazyTensor(weights, axis=1)).sum(axis=1, backend=backend_string).reshape(-1)
        elif options.number_dimensions == 2:
            gz_field = None
        else:
            raise ValueError("Number of dimensions have to be 2 or 3")

    return ExportedFields(scalar_field, gx_field, gy_field, gz_field)
