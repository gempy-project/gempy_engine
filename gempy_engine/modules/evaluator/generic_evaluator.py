import numpy as np
from typing import Optional

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_evaluation_grad_kernel, yield_evaluation_kernel


def generic_evaluator(solver_input: SolverInput, weights: np.ndarray, options: InterpolationOptions) -> ExportedFields:
    grid_size = solver_input.xyz_to_interpolate.shape[0]
    matrix_size = grid_size * weights.shape[0]
    scalar_field: np.ndarray = BackendTensor.t.zeros(grid_size, dtype=weights.dtype)
    gx_field: Optional[np.ndarray] = None
    gy_field: Optional[np.ndarray] = None
    gz_field: Optional[np.ndarray] = None
    gradient = options.compute_scalar_gradient

    # * Chunking the evaluation
    max_size = options.evaluation_chunk_size
    n_chunks = int(np.ceil(matrix_size / max_size))
    chunk_size = int(np.ceil(grid_size / n_chunks))
    for i in range(n_chunks): # TODO: It seems the chunking is not properly implemented
        slice_array = slice(i * chunk_size, (i + 1) * chunk_size)
        scalar_field_chunk, gx_field_chunk, gy_field_chunk, gz_field_chunk = _eval_on(
            solver_input=solver_input,
            weights=weights,
            options=options,
            slice_array=slice_array
        )

        scalar_field[slice_array] = scalar_field_chunk
        if gradient is True:
            if i == 0:
                gx_field = BackendTensor.t.zeros(grid_size, dtype=weights.dtype)
                gy_field = BackendTensor.t.zeros(grid_size, dtype=weights.dtype)
                gz_field = BackendTensor.t.zeros(grid_size, dtype=weights.dtype)

            gx_field[slice_array] = gx_field_chunk
            gy_field[slice_array] = gy_field_chunk
            gz_field[slice_array] = gz_field_chunk

    if n_chunks > 5:
        print(f"Chunking done: {n_chunks} chunks")

    return ExportedFields(scalar_field, gx_field, gy_field, gz_field)


def _eval_on(solver_input, weights, options, slice_array: slice = None):
    eval_kernel = yield_evaluation_kernel(solver_input, options.kernel_options, slice_array=slice_array)
    scalar_field: np.ndarray = (eval_kernel.T @ weights).reshape(-1)
    scalar_field[-50:]
    gx_field: Optional[np.ndarray] = None
    gy_field: Optional[np.ndarray] = None
    gz_field: Optional[np.ndarray] = None
    if options.compute_scalar_gradient is True:
        eval_gx_kernel = yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=0, slice_array=slice_array)
        eval_gy_kernel = yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=1, slice_array=slice_array)
        gx_field = (eval_gx_kernel.T @ weights).reshape(-1)
        gy_field = (eval_gy_kernel.T @ weights).reshape(-1)

        if options.number_dimensions == 3:
            eval_gz_kernel = yield_evaluation_grad_kernel(solver_input, options.kernel_options, axis=2, slice_array=slice_array)
            gz_field = (eval_gz_kernel.T @ weights).reshape(-1)
        elif options.number_dimensions == 2:
            gz_field = None
        else:
            raise ValueError("Number of dimensions have to be 2 or 3")
    return scalar_field, gx_field, gy_field, gz_field
