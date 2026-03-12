from typing import Optional

import numpy as np

import gempy_engine
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


def _build_block_sparse_ranges(M_sizes: list[int], N_sizes: list[int]):
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
    if n_fields == 1:
        result = symbolic_evaluator_optimized(eval_inputs[0], weights_list[0], options)
        return [result]

    backend_string = BackendTensor.get_backend_string()

    # Collect sizes: M = grid/eval points (j-dim), N = weights/cov_size (i-dim)
    M_sizes = [ei.xyz_to_interpolate.shape[0] for ei in eval_inputs]
    N_sizes = [w.shape[0] for w in weights_list]

    # Build block-sparse ranges
    ranges = _build_block_sparse_ranges(M_sizes, N_sizes)

    # Concatenate weights
    all_weights = np.concatenate(weights_list, axis=0)

    # Build concatenated kernel by evaluating each field's kernel individually
    # and relying on PyKeOps block-sparse ranges to skip cross-field interactions.
    # We need to build a single "stacked" eval_input with concatenated arrays.
    stacked_eval_input: EvaluatorInput = _build_stacked_eval_input(eval_inputs)

    eval_kernel = yield_evaluation_kernel(stacked_eval_input, options.kernel_options)

    if BackendTensor.engine_backend == gempy_engine.config.AvailableBackends.numpy:
        from pykeops.numpy import LazyTensor
        lazy_weights = LazyTensor(np.asfortranarray(all_weights.reshape(-1, 1)), axis=0)
        scalar_field_concat: np.ndarray = (eval_kernel * lazy_weights).sum(
            axis=0, backend=backend_string, ranges=ranges
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


def _build_stacked_eval_input(eval_inputs: list[EvaluatorInput]) -> EvaluatorInput:
    """Build a stacked EvaluatorInput by concatenating arrays from multiple eval_inputs.
    
    Creates a synthetic EvaluatorInput whose internal arrays (xyz_to_interpolate,
    sp_internal coords, ori_internal coords, fault data) are concatenated across
    all provided eval_inputs, suitable for block-sparse PyKeOps evaluation.
    """
    from ...core.data import SurfacePointsInternals, OrientationsInternals
    from ...core.data.kernel_classes.orientations import Orientations
    from ...core.data.kernel_classes.faults import FaultsData
    from ...core.data.internal_structs import SolverInput_v2
    
    # Concatenate xyz_to_interpolate (evaluation grid points)
    all_xyz = np.concatenate([ei.xyz_to_interpolate for ei in eval_inputs], axis=0)
    
    # Concatenate surface points internals (frozen dataclass - use constructor)
    all_ref_sp = np.concatenate([ei.sp_internal.ref_surface_points for ei in eval_inputs], axis=0)
    all_rest_sp = np.concatenate([ei.sp_internal.rest_surface_points for ei in eval_inputs], axis=0)
    all_nugget_ref_rest = np.concatenate([ei.sp_internal.nugget_effect_ref_rest for ei in eval_inputs], axis=0)
    
    stacked_sp = SurfacePointsInternals(
        ref_surface_points=all_ref_sp,
        rest_surface_points=all_rest_sp,
        nugget_effect_ref_rest=all_nugget_ref_rest
    )
    
    # Concatenate orientations internals
    all_dip_positions = np.concatenate([ei.ori_internal.dip_positions_tiled for ei in eval_inputs], axis=0)
    all_gradients = np.concatenate([ei.ori_internal.gradients_tiled for ei in eval_inputs], axis=0)
    all_nugget_grad = np.concatenate([ei.ori_internal.nugget_effect_grad for ei in eval_inputs], axis=0)
    
    # OrientationsInternals requires an Orientations object; create a dummy one
    # with concatenated dip_positions (used only for n_orientations property)
    all_ori_dip_pos = np.concatenate([ei.ori_internal.orientations.dip_positions for ei in eval_inputs], axis=0)
    all_ori_dip_grad = np.concatenate([ei.ori_internal.orientations.dip_gradients for ei in eval_inputs], axis=0)
    all_ori_nugget = np.concatenate([ei.ori_internal.orientations.nugget_effect_grad for ei in eval_inputs], axis=0)
    stacked_orientations = Orientations(
        dip_positions=all_ori_dip_pos,
        dip_gradients=all_ori_dip_grad,
        nugget_effect_grad=all_ori_nugget
    )
    
    stacked_ori = OrientationsInternals(
        orientations=stacked_orientations,
        dip_positions_tiled=all_dip_positions,
        gradients_tiled=all_gradients,
        nugget_effect_grad=all_nugget_grad
    )
    
    # Concatenate fault data
    has_faults = any(ei.fault_internal.n_faults > 0 for ei in eval_inputs)
    if has_faults:
        all_fault_everywhere = np.concatenate(
            [ei.fault_internal.fault_values_everywhere for ei in eval_inputs], axis=1
        )
        all_fault_ref = np.concatenate([ei.fault_internal.fault_values_ref for ei in eval_inputs], axis=0)
        all_fault_rest = np.concatenate([ei.fault_internal.fault_values_rest for ei in eval_inputs], axis=0)
        all_fault_on_sp = np.concatenate([ei.fault_internal.fault_values_on_sp for ei in eval_inputs], axis=0)
        
        stacked_faults = FaultsData(
            fault_values_everywhere=all_fault_everywhere,
            fault_values_on_sp=all_fault_on_sp,
            fault_values_ref=all_fault_ref,
            fault_values_rest=all_fault_rest
        )
    else:
        stacked_faults = FaultsData(
            fault_values_everywhere=np.zeros((0, 0), dtype=all_xyz.dtype),
            fault_values_on_sp=np.zeros((0, 0), dtype=all_xyz.dtype)
        )
    
    # Build stacked SolverInput_v2
    stacked_solver = SolverInput_v2(
        sp_internal=stacked_sp,
        ori_internal=stacked_ori,
        fault_internal=stacked_faults
    )
    stacked_solver.weights_x0 = None
    
    # Build stacked EvaluatorInput (bypass __init__ since we have pre-built arrays)
    stacked_eval = EvaluatorInput.__new__(EvaluatorInput)
    stacked_eval.solver_input = stacked_solver
    stacked_eval.xyz_to_interpolate = all_xyz
    stacked_eval._n_points_per_surface = None
    stacked_eval._slice_feature = slice(None, None)
    stacked_eval._grid_size = None
    
    return stacked_eval
