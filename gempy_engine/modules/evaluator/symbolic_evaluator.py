from typing import Optional

import numpy as np
import torch

import gempy_engine
from ..kernel_constructor._kernels_assembler import create_scalar_kernel, create_grad_kernel
from ..kernel_constructor._structs import KernelInput
from ..kernel_constructor._vectors_preparation import evaluation_vectors_preparations
from ...core.backend_tensor import BackendTensor
from ...core.data import InterpolationOptions
from ...core.data.exported_fields import ExportedFields
from ...core.data.internal_structs import SolverInput, EvaluatorInput
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


def _build_block_sparse_ranges(M_sizes: list[int], N_sizes: list[int]):
    """Build PyKeOps block-sparse ranges tuple for block-diagonal evaluation."""
    keep_i = np.cumsum([0] + N_sizes)
    keep_j = np.cumsum([0] + M_sizes)

    n_fields = len(M_sizes)
    ranges_i = np.array([[keep_i[k], keep_i[k + 1]] for k in range(n_fields)], dtype=np.int32)
    ranges_j = np.array([[keep_j[k], keep_j[k + 1]] for k in range(n_fields)], dtype=np.int32)

    # FIXED: Slices must be the cumulative sum of blocks (1-indexed sequence)
    slices_i = np.arange(1, n_fields + 1, dtype=np.int32)
    slices_j = np.arange(1, n_fields + 1, dtype=np.int32)

    # Order: (ranges_i, slices_i, redranges_j, ranges_j, slices_j, redranges_i)

    numpy_ranges = (ranges_i, slices_i, ranges_j, ranges_j, slices_j, ranges_i)
    tensor_ranges = (BackendTensor.t.array(range_) for range_ in numpy_ranges)
    return tensor_ranges


def symbolic_evaluator_optimized_stacked(
        eval_inputs: list[EvaluatorInput],
        weights_list: list[np.ndarray],
        options: InterpolationOptions
) -> list[ExportedFields]:
    """Evaluate multiple fields in a single PyKeOps call using block-sparse ranges.
    
    Concatenates all evaluation inputs across stacks, builds block-diagonal ranges,
    performs a single PyKeOps reduction, then splits results back per field.
    """

    n_fields = len(eval_inputs)

    # Collect sizes: M = grid/eval points (j-dim), N = weights/cov_size (i-dim)
    M_sizes = [ei.xyz_to_interpolate.shape[0] for ei in eval_inputs]
    N_sizes = [w.shape[0] for w in weights_list]

    if options.compute_scalar_gradient is True:
        # We will stack scalar, then gx, then gy, and optionally gz
        # All of them have the same M_sizes and N_sizes
        M_sizes = M_sizes * 4
        N_sizes = N_sizes * 4

    # Build block-sparse ranges
    ranges = _build_block_sparse_ranges(M_sizes, N_sizes)

    # Concatenate weights
    all_weights = BackendTensor.t.concatenate(weights_list, axis=0)
    if options.compute_scalar_gradient is True:
        all_weights = BackendTensor.t.tile(all_weights, 4)

    kernel_data_list = []
    # 1. Build a sequentially ordered list of task arguments: (eval_input, axis)
    prep_tasks = [(ei, None) for ei in eval_inputs]

    if options.compute_scalar_gradient is True:
        # X gradient
        prep_tasks.extend([(ei, 0) for ei in eval_inputs])
        # Y gradient
        prep_tasks.extend([(ei, 1) for ei in eval_inputs])
        # Z gradient
        if options.number_dimensions == 3:
            prep_tasks.extend([(ei, 2) for ei in eval_inputs])

    # 2. Define a small wrapper function for the executor to map over
    def _run_prep(args):
        ei, axis = args
        # noinspection PyTypeChecker
        return evaluation_vectors_preparations(
            ei,
            options.kernel_options,
            axis=axis,
            slice_array=None
        )

    # 3. Execute in parallel (preserving order)
    # Note: You can pass a specific max_workers or let Python decide based on your CPU cores
    with concurrent.futures.ThreadPoolExecutor() as executor:
        kernel_data_list = list(executor.map(_run_prep, prep_tasks))

    concat_kernel_data: KernelInput = _build_stacked_kernel_data(kernel_data_list)
    if options.compute_scalar_gradient is True:
        eval_kernel = create_grad_kernel(concat_kernel_data, options.kernel_options)
    if options.compute_scalar is True:
        eval_kernel = create_scalar_kernel(concat_kernel_data, options.kernel_options)

    if BackendTensor.engine_backend == gempy_engine.config.AvailableBackends.numpy:
        from pykeops.numpy import LazyTensor
        lazy_weights = LazyTensor(np.asfortranarray(all_weights.reshape(-1, 1)), axis=0)
        all_results_concat: np.ndarray = (eval_kernel * lazy_weights).sum(
            axis=0,
            backend=BackendTensor.get_backend_string(),
            ranges=ranges
        ).reshape(-1)
    else:
        from pykeops.torch import LazyTensor
        try:
            all_weights = all_weights.pin_memory().to("cuda", non_blocking=True)
            lazy_weights = LazyTensor(all_weights.view((-1, 1)), axis=0)
            all_results_concat = (eval_kernel * lazy_weights).sum(
                axis=0,
                backend=BackendTensor.get_backend_string(),
                ranges=ranges
            ).reshape(-1)
        except TypeError:
            raise ValueError("Failed to compute symbolic evaluation with PyKeOps. Ensure that all_weights and eval_kernel are compatible for lazy tensor operations.")

    # For torch
    all_results_concat = all_results_concat.to("cpu")
    all_results_split = BackendTensor.t.split(all_results_concat, M_sizes)

    # For numpy
    # split_indices = np.cumsum(M_sizes)[:-1]
    # all_results_split = np.split(all_results_concat, split_indices)

    original_n_fields = len(eval_inputs)
    scalar_fields = all_results_split[:original_n_fields]
    gx_fields = [None] * original_n_fields
    gy_fields = [None] * original_n_fields
    gz_fields = [None] * original_n_fields

    if options.compute_scalar_gradient is True:
        gx_fields = all_results_split[original_n_fields: 2 * original_n_fields]
        gy_fields = all_results_split[2 * original_n_fields: 3 * original_n_fields]
        if options.number_dimensions == 3:
            gz_fields = all_results_split[3 * original_n_fields: 4 * original_n_fields]

    # Build ExportedFields per stack
    results = []
    for idx in range(n_fields):
        results.append(ExportedFields(scalar_fields[idx], gx_fields[idx], gy_fields[idx], gz_fields[idx]))

    return results


import concurrent.futures


def _build_stacked_kernel_data(kernel_data_list: list[KernelInput], max_workers: int = 8) -> KernelInput:
    """Build a stacked KernelInput by concatenating the internal arrays in parallel."""
    from ..kernel_constructor._structs import (
        OrientationSurfacePointsCoords, CartesianSelector, OrientationsDrift,
        PointsDrift, FaultDrift, DriftMatrixSelector, _cast_tensors
    )
    BackendTensor.pykeops_enabled = True

    def _stack_sub_struct(items, cls):
        """Concatenate fields of a sub-dataclass and move to GPU."""
        if items[0] is None:
            return None

        result = cls.__new__(cls)
        first_item = items[0]

        for field_name in first_item.__dict__:
            vals = [getattr(item, field_name) for item in items]
            first_val = vals[0]

            if isinstance(first_val, int):
                setattr(result, field_name, first_val)
                continue

            # 1. Concatenate
            axis = 0 if first_val.shape[0] > first_val.shape[1] else 1
            concat_val = BackendTensor.t.concatenate(vals, axis=axis)

            # 2. Enforce Contiguity & 3. Move to GPU
            if BackendTensor.engine_backend == gempy_engine.config.AvailableBackends.numpy:
                concat_val = concat_val.copy()
            else:
                concat_val = concat_val.contiguous()

                # Move to GPU asynchronously if using PyTorch
                concat_val = concat_val.to('cuda', non_blocking=True)

            setattr(result, field_name, concat_val)

        # Ensure _cast_tensors safely handles tensors that are already on the GPU!
        _cast_tensors(result)
        return result

    stacked = KernelInput.__new__(KernelInput)

    # Define the base fields to extract and their corresponding classes
    tasks = [
            ("ori_sp_matrices", OrientationSurfacePointsCoords),
            ("cartesian_selector", CartesianSelector),
            ("ori_drift", OrientationsDrift),
            ("ref_drift", PointsDrift),
            ("rest_drift", PointsDrift),
            ("drift_matrix_selector", DriftMatrixSelector),
    ]

    # Safely check for ref_fault using getattr
    if getattr(kernel_data_list[0], 'ref_fault', None) is not None:
        tasks.extend([
                ("ref_fault", FaultDrift),
                ("rest_fault", FaultDrift)
        ])
    else:
        # Explicitly set them to None on the stacked object to match your original structure
        stacked.ref_fault = None
        stacked.rest_fault = None
    # Execute concatenation in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks: extract the list of sub-structs once before passing to the thread
        future_to_field = {
                executor.submit(_stack_sub_struct, [getattr(kd, field) for kd in kernel_data_list], cls): field
                for field, cls in tasks
        }

        # Collect results as they finish
        for future in concurrent.futures.as_completed(future_to_field):
            field_name = future_to_field[future]
            try:
                setattr(stacked, field_name, future.result())
            except Exception as exc:
                raise RuntimeError(f"Stacking field {field_name} generated an exception: {exc}")

    # Handle scalars
    stacked.nugget_scalar = kernel_data_list[0].nugget_scalar
    stacked.nugget_grad = kernel_data_list[0].nugget_grad

    return stacked
