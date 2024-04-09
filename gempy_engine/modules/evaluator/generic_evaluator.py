import numpy as np
from typing import Optional

from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_evaluation_grad_kernel


def generic_evaluator(compute_gradient, eval_kernel, options, solver_input, weights):
    scalar_field: np.ndarray = (eval_kernel.T @ weights).reshape(-1)
    gx_field: Optional[np.ndarray] = None
    gy_field: Optional[np.ndarray] = None
    gz_field: Optional[np.ndarray] = None

    if compute_gradient is True:
        eval_gx_kernel = yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=0)
        eval_gy_kernel = yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=1)
        gx_field = (eval_gx_kernel.T @ weights).reshape(-1)
        gy_field = (eval_gy_kernel.T @ weights).reshape(-1)

        if options.number_dimensions == 3:
            eval_gz_kernel = yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=2)
            gz_field = (eval_gz_kernel.T @ weights).reshape(-1)
        elif options.number_dimensions == 2:
            gz_field = None
        else:
            raise ValueError("Number of dimensions have to be 2 or 3")

    return ExportedFields(scalar_field, gx_field, gy_field, gz_field)
