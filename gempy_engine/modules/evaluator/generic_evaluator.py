import numpy as np
import gc
from typing import Optional

import gempy_engine.config
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_evaluation_grad_kernel, yield_evaluation_kernel
from gempy_engine.modules.kernel_constructor._internalDistancesMatrices import DistancesBuffer


def generic_evaluator(
    solver_input: SolverInput,
    weights: np.ndarray,
    options: InterpolationOptions
) -> ExportedFields:
    grid_size = solver_input.xyz_to_interpolate.shape[0]
    max_op_size = options.evaluation_chunk_size
    num_weights = weights.shape[0]
    

    chunk_size_grid = max(1, int(max_op_size / num_weights))  # Ensure at least 1 point per chunk
    n_chunks = int(np.ceil(grid_size / chunk_size_grid)) 

    # Pre‑allocate outputs
    scalar_field = BackendTensor.t.zeros(grid_size, dtype=weights.dtype)
    gx_field: Optional[np.ndarray] = None
    gy_field: Optional[np.ndarray] = None
    gz_field: Optional[np.ndarray] = None
    if options.compute_scalar_gradient:
        gx_field = BackendTensor.t.zeros(grid_size, dtype=weights.dtype)
        gy_field = BackendTensor.t.zeros(grid_size, dtype=weights.dtype)
        if options.number_dimensions == 3:
            gz_field = BackendTensor.t.zeros(grid_size, dtype=weights.dtype)

    # Chunked evaluation over grid indices
    for i in range(n_chunks):

        start = i * chunk_size_grid
        end = min(grid_size, start + chunk_size_grid)  # Ensure 'end' doesn't exceed grid_size
        slice_array = slice(start, end)

        # Avoid processing empty slices if start == end
        if start >= end:
            continue

        sf_chunk, gx_chunk, gy_chunk, gz_chunk = _eval_on(
            solver_input=solver_input,
            weights=weights,
            options=options,
            slice_array=slice_array
        )

        scalar_field[slice_array] = sf_chunk
        if options.compute_scalar_gradient:
            gx_field[slice_array] = gx_chunk  # type: ignore
            gy_field[slice_array] = gy_chunk  # type: ignore
            if gz_field is not None:
                gz_field[slice_array] = gz_chunk  # type: ignore

    # Force garbage collection every few chunks to prevent memory buildup
    if (i + 1) % 5 == 0 or i == n_chunks - 1:
        gc.collect()
        
    if n_chunks > 5:
        print(f"Chunking done: {n_chunks} chunks")

    micro = options.evaluation_options.micro_anisotropic
    if micro.enabled and micro.weights is not None and micro.points is not None and micro.anisotropy_matrices is not None:
        from gempy_engine.modules.evaluator.micro_anisotropic_evaluator import evaluate_micro_correction
        if BackendTensor.engine_backend != gempy_engine.config.AvailableBackends.numpy:
            scalar_field = BackendTensor.t.to_numpy(scalar_field)
        correction = evaluate_micro_correction(
            xyz_to_interpolate=solver_input.xyz_to_interpolate,
            micro_points=micro.points,
            micro_weights=micro.weights,
            anisotropy_matrices=micro.anisotropy_matrices,
            kernel_range=micro.kernel_range,
        )
        scalar_field = scalar_field + correction

    return ExportedFields(scalar_field, gx_field, gy_field, gz_field)


def _eval_on(
    solver_input: SolverInput,
    weights: np.ndarray,
    options: InterpolationOptions,
    slice_array: slice
):
    distance_buffer = DistancesBuffer() if options.compute_scalar_gradient else None
    eval_kernel = yield_evaluation_kernel(
        solver_input,
        options.kernel_options,
        slice_array=slice_array,
        distance_buffer=distance_buffer,
    )
    try:
        scalar_field = (eval_kernel.T @ weights).reshape(-1)
    except ValueError:
        scalar_field = None
    
    del eval_kernel

    gx_field: Optional[np.ndarray] = None
    gy_field: Optional[np.ndarray] = None
    gz_field: Optional[np.ndarray] = None

    if options.compute_scalar_gradient:
        eval_gx = yield_evaluation_grad_kernel(
            solver_input,
            options.kernel_options,
            axis=0,
            slice_array=slice_array,
            distance_buffer=distance_buffer,
        )
        gx_field = (eval_gx.T @ weights).reshape(-1)  # Use BEFORE deleting
        del eval_gx  # Clean up immediately after use
        
        eval_gy = yield_evaluation_grad_kernel(
            solver_input,
            options.kernel_options,
            axis=1,
            slice_array=slice_array,
            distance_buffer=distance_buffer,
        )
        gy_field = (eval_gy.T @ weights).reshape(-1)  # Use BEFORE deleting
        del eval_gy  # Clean up immediately after use

        if options.number_dimensions == 3:
            eval_gz = yield_evaluation_grad_kernel(
                solver_input,
                options.kernel_options,
                axis=2,
                slice_array=slice_array,
                distance_buffer=distance_buffer,
            )
            gz_field = (eval_gz.T @ weights).reshape(-1)  # Use BEFORE deleting
            del eval_gz  # Clean up immediately after use
        elif options.number_dimensions != 2:
            raise ValueError("`number_dimensions` must be 2 or 3")

    return scalar_field, gx_field, gy_field, gz_field
