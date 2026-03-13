from typing import Optional

import numpy as np

import gempy_engine
from ..kernel_constructor._kernels_assembler import create_scalar_kernel
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


def symbolic_evaluator_optimized(eval_input: EvaluatorInput, weights: np.ndarray, options: InterpolationOptions):
    if BackendTensor.engine_backend == gempy_engine.config.AvailableBackends.numpy and eval_input.xyz_to_interpolate.flags['C_CONTIGUOUS'] is False:  # ! This is not working with TF yet
        print("xyz is not C_CONTIGUOUS")
    # ! Seems not to make any difference but we need this if we want to change the backend
    # ! We need to benchmark GPU vs CPU with more input
    backend_string = BackendTensor.get_backend_string()

    eval_kernel = yield_evaluation_kernel(eval_input, options.kernel_options)
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
        eval_gx_kernel = yield_evaluation_grad_kernel(eval_input, options.kernel_options, axis=0)
        eval_gy_kernel = yield_evaluation_grad_kernel(eval_input, options.kernel_options, axis=1)

        gx_field = (eval_gx_kernel * lazy_weights).sum(axis=0, backend=backend_string).reshape(-1)
        gy_field = (eval_gy_kernel * lazy_weights).sum(axis=0, backend=backend_string).reshape(-1)

        if options.number_dimensions == 3:
            eval_gz_kernel = yield_evaluation_grad_kernel(eval_input, options.kernel_options, axis=2)
            gz_field = (eval_gz_kernel * lazy_weights).sum(axis=0, backend=backend_string).reshape(-1)
        elif options.number_dimensions == 2:
            gz_field = None
        else:
            raise ValueError("Number of dimensions have to be 2 or 3")

    return ExportedFields(scalar_field, gx_field, gy_field, gz_field)


def _build_block_sparse_ranges_(M_sizes: list[int], N_sizes: list[int]):
    """Build PyKeOps block-sparse ranges tuple for block-diagonal evaluation.
    
    Args:
        M_sizes: number of evaluation points (j-dimension / grid) per field
        N_sizes: number of weights/centers (i-dimension / cov_size) per field
    
    Returns:
        ranges: 6-tuple for PyKeOps block-sparse reductions
    """
    keep_i = np.cumsum([0] + N_sizes)
    keep_j = np.cumsum([0] + M_sizes)

    n_fields = len(M_sizes)
    ranges_i = np.array([[keep_i[k], keep_i[k + 1]] for k in range(n_fields)], dtype=np.int32)
    slices_i = np.arange(n_fields, dtype=np.int32)
    ranges_j = np.array([[keep_j[k], keep_j[k + 1]] for k in range(n_fields)], dtype=np.int32)
    slices_j = np.arange(n_fields, dtype=np.int32)

    return (ranges_i, slices_i, ranges_j, slices_j, ranges_j, slices_j)


def _build_block_sparse_ranges_(M_sizes: list[int], N_sizes: list[int]):
    """Build PyKeOps block-sparse ranges tuple for block-diagonal evaluation."""
    keep_i = np.cumsum([0] + N_sizes)
    keep_j = np.cumsum([0] + M_sizes)

    n_fields = len(M_sizes)
    ranges_i = np.array([[keep_i[k], keep_i[k + 1]] for k in range(n_fields)], dtype=np.int32)
    slices_i = np.arange(n_fields, dtype=np.int32)
    ranges_j = np.array([[keep_j[k], keep_j[k + 1]] for k in range(n_fields)], dtype=np.int32)
    slices_j = np.arange(n_fields, dtype=np.int32)

    # FIXED: Reordered to match PyKeOps expectations
    # (ranges_i, slices_i, redranges_j, ranges_j, slices_j, redranges_i)
    return (ranges_i, slices_i, ranges_j, ranges_j, slices_j, ranges_i)

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
    return (ranges_i, slices_i, ranges_j, ranges_j, slices_j, ranges_i)


def symbolic_evaluator_optimized_stacked(
        eval_inputs: list[EvaluatorInput],
        weights_list: list[np.ndarray],
        options: InterpolationOptions
) -> list[ExportedFields]:
    """Evaluate multiple fields in a single PyKeOps call using block-sparse ranges.
    
    Concatenates all evaluation inputs across stacks, builds block-diagonal ranges,
    performs a single PyKeOps reduction, then splits results back per field.
    """
    from ...modules.kernel_constructor.kernel_constructor_interface import yield_evaluation_kernel, yield_evaluation_grad_kernel

    n_fields = len(eval_inputs)

    # If only one field, fall back to the non-stacked version
    # if n_fields == 1:
    #     result = symbolic_evaluator_optimized(eval_inputs[0], weights_list[0], options)
    #     return [result]

    backend_string = BackendTensor.get_backend_string()

    # Collect sizes: M = grid/eval points (j-dim), N = weights/cov_size (i-dim)
    # TODO: Reuse grid
    M_sizes = [ei.xyz_to_interpolate.shape[0] for ei in eval_inputs]
    N_sizes = [w.shape[0] for w in weights_list]

    # Build block-sparse ranges
    ranges = _build_block_sparse_ranges(M_sizes, N_sizes)

    # Concatenate weights
    all_weights = np.concatenate(weights_list, axis=0)

    kernel_data_list = []

    if n_fields == 1:
        BackendTensor.pykeops_enabled = True
        concat_kernel_data: KernelInput = evaluation_vectors_preparations(eval_inputs[0], options.kernel_options, axis=None, slice_array=None)
        # BackendTensor.pykeops_enabled = False
    else:
        for i in range(n_fields):
            kernel_data: KernelInput = evaluation_vectors_preparations(eval_inputs[i], options.kernel_options, axis=None, slice_array=None)
            kernel_data_list.append(kernel_data)

        concat_kernel_data: KernelInput = _build_stacked_kernel_data(kernel_data_list)
    eval_kernel = create_scalar_kernel(concat_kernel_data, options.kernel_options)

    if BackendTensor.engine_backend == gempy_engine.config.AvailableBackends.numpy:
        from pykeops.numpy import LazyTensor
        lazy_weights = LazyTensor(np.asfortranarray(all_weights.reshape(-1, 1)), axis=0)
        scalar_field_concat: np.ndarray = (eval_kernel * lazy_weights).sum(
            axis=0,
            backend=backend_string,
            ranges=ranges
        ).reshape(-1)
    else:
        from pykeops.torch import LazyTensor
        lazy_weights = LazyTensor(all_weights.view((-1, 1)), axis=0)
        scalar_field_concat: np.ndarray = (eval_kernel * lazy_weights).sum(
            axis=0, backend=backend_string, ranges=ranges
        ).reshape(-1)

    # Split scalar field results back per field
    split_indices_M = np.cumsum(M_sizes)[:-1]
    scalar_fields = np.split(scalar_field_concat, split_indices_M)

    # Handle gradient fields
    gx_fields = [None] * n_fields
    gy_fields = [None] * n_fields
    gz_fields = [None] * n_fields

    if options.compute_scalar_gradient is True:
        eval_gx_kernel = yield_evaluation_grad_kernel(stacked_eval_input, options.kernel_options, axis=0)
        eval_gy_kernel = yield_evaluation_grad_kernel(stacked_eval_input, options.kernel_options, axis=1)

        gx_concat = (eval_gx_kernel * lazy_weights).sum(axis=0, backend=backend_string, ranges=ranges).reshape(-1)
        gy_concat = (eval_gy_kernel * lazy_weights).sum(axis=0, backend=backend_string, ranges=ranges).reshape(-1)

        gx_split = np.split(gx_concat, split_indices_M)
        gy_split = np.split(gy_concat, split_indices_M)
        gx_fields = list(gx_split)
        gy_fields = list(gy_split)

        if options.number_dimensions == 3:
            eval_gz_kernel = yield_evaluation_grad_kernel(stacked_eval_input, options.kernel_options, axis=2)
            gz_concat = (eval_gz_kernel * lazy_weights).sum(axis=0, backend=backend_string, ranges=ranges).reshape(-1)
            gz_fields = list(np.split(gz_concat, split_indices_M))
        elif options.number_dimensions == 2:
            gz_fields = [None] * n_fields
        else:
            raise ValueError("Number of dimensions have to be 2 or 3")

    # Build ExportedFields per stack
    results = []
    for idx in range(n_fields):
        results.append(ExportedFields(scalar_fields[idx], gx_fields[idx], gy_fields[idx], gz_fields[idx]))

    return results


def _build_stacked_kernel_data(kernel_data_list: list[KernelInput]) -> KernelInput:
    """Build a stacked KernelInput by concatenating the internal arrays from multiple KernelInput instances.
    
    For each sub-dataclass field, arrays with suffix '_i' (shape (M, 1, D)) are concatenated along axis=0,
    and arrays with suffix '_j' (shape (1, N, D)) are concatenated along axis=1.
    Scalar fields (nuggets) are taken from the first element (assumed identical across inputs).
    """
    from ..kernel_constructor._structs import (
        OrientationSurfacePointsCoords, CartesianSelector, OrientationsDrift,
        PointsDrift, FaultDrift, DriftMatrixSelector, _cast_tensors
    )
    BackendTensor.pykeops_enabled = True

    def _stack_sub_struct(items, cls):
        """Concatenate fields of a sub-dataclass: _i fields along axis=0, _j fields along axis=1.
        
        Handles both raw numpy arrays and PyKeOps LazyTensor fields by extracting
        the underlying numpy data before concatenation, then re-applying _cast_tensors.
        """
        result = cls.__new__(cls)
        for field_name in items[0].__dict__:
            vals_raw = [getattr(item, field_name) for item in items]
            vals = vals_raw
            if not isinstance(vals[0], np.ndarray):
                setattr(result, field_name, vals_raw[0])
                continue
            if vals[0].shape[0] > vals[0].shape[1]:  # _i field: (M, 1, D)
                setattr(result, field_name, np.concatenate(vals, axis=0))
            else:  # _j field: (1, N, D)
                setattr(result, field_name, np.concatenate(vals, axis=1))
        _cast_tensors(result)
        return result

    stacked = KernelInput.__new__(KernelInput)
    stacked.ori_sp_matrices = _stack_sub_struct([kd.ori_sp_matrices for kd in kernel_data_list], OrientationSurfacePointsCoords)
    stacked.cartesian_selector = _stack_sub_struct([kd.cartesian_selector for kd in kernel_data_list], CartesianSelector)
    stacked.ori_drift = _stack_sub_struct([kd.ori_drift for kd in kernel_data_list], OrientationsDrift)
    stacked.ref_drift = _stack_sub_struct([kd.ref_drift for kd in kernel_data_list], PointsDrift)
    stacked.rest_drift = _stack_sub_struct([kd.rest_drift for kd in kernel_data_list], PointsDrift)
    stacked.drift_matrix_selector = _stack_sub_struct([kd.drift_matrix_selector for kd in kernel_data_list], DriftMatrixSelector)

    if kernel_data_list[0].ref_fault is not None:
        stacked.ref_fault = _stack_sub_struct([kd.ref_fault for kd in kernel_data_list], FaultDrift)
        stacked.rest_fault = _stack_sub_struct([kd.rest_fault for kd in kernel_data_list], FaultDrift)
    else:
        stacked.ref_fault = None
        stacked.rest_fault = None

    stacked.nugget_scalar = kernel_data_list[0].nugget_scalar
    stacked.nugget_grad = kernel_data_list[0].nugget_grad

    return stacked
