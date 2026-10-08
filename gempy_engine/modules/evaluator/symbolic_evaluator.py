from typing import Optional

import numpy as np

import gempy_engine
from ..kernel_constructor._kernels_assembler import create_scalar_kernel, create_grad_kernel
from ..kernel_constructor._structs import KernelInput
from ..kernel_constructor._vectors_preparation import evaluation_vectors_preparations
from ...core.backend_tensor import BackendTensor
from ...core.data import InterpolationOptions
from ...core.data.exported_fields import ExportedFields
from ...core.data.internal_structs import SolverInput, EvaluatorInput
from ..kernel_constructor.kernel_constructor_interface import yield_evaluation_grad_kernel, yield_evaluation_kernel
from ..kernel_constructor.execution_mode import KernelExecutionMode
from ..kernel_constructor._internalDistancesMatrices import DistancesBuffer


def symbolic_evaluator(solver_input: SolverInput, weights: np.ndarray, options: InterpolationOptions):
    if BackendTensor.engine_backend == gempy_engine.config.AvailableBackends.numpy and solver_input.xyz_to_interpolate.flags['C_CONTIGUOUS'] is False:  # ! This is not working with TF yet
        print("xyz is not C_CONTIGUOUS")
    # ! Seems not to make any difference but we need this if we want to change the backend
    # ! We need to benchmark GPU vs CPU with more input
    backend_string = "GPU" if BackendTensor.use_gpu else "CPU"

    distance_buffer = DistancesBuffer() if options.compute_scalar_gradient else None
    eval_kernel = yield_evaluation_kernel(
        solver_input,
        options.kernel_options,
        pykeops=True,
        distance_buffer=distance_buffer,
    )
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
        eval_gx_kernel = yield_evaluation_grad_kernel(
            solver_input,
            options.kernel_options,
            axis=0,
            pykeops=True,
            distance_buffer=distance_buffer,
        )
        eval_gy_kernel = yield_evaluation_grad_kernel(
            solver_input,
            options.kernel_options,
            axis=1,
            pykeops=True,
            distance_buffer=distance_buffer,
        )

        gx_field = (eval_gx_kernel * lazy_weights).sum(axis=0, backend=backend_string).reshape(-1)
        gy_field = (eval_gy_kernel * lazy_weights).sum(axis=0, backend=backend_string).reshape(-1)

        if options.number_dimensions == 3:
            eval_gz_kernel = yield_evaluation_grad_kernel(
                solver_input,
                options.kernel_options,
                axis=2,
                pykeops=True,
                distance_buffer=distance_buffer,
            )
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

    if BackendTensor.engine_backend == gempy_engine.config.AvailableBackends.PYTORCH:
        device = "cuda" if BackendTensor.use_gpu else "cpu"
        return tuple(BackendTensor.t.array(range_).to(device) for range_ in numpy_ranges)

    return numpy_ranges


def _validate_stacked_dimensions(eval_kernel, weights, M_sizes: list[int], N_sizes: list[int]) -> None:
    """Fail before PyKeOps when block ranges do not describe the lazy kernel."""
    expected_i = sum(N_sizes)
    expected_j = sum(M_sizes)
    kernel_shape = tuple(eval_kernel.shape)
    weights_size = weights.shape[0]

    if kernel_shape[0] != expected_i or kernel_shape[1] != expected_j or weights_size != expected_i:
        raise ValueError(
            "Inconsistent stacked PyKeOps dimensions: "
            f"kernel={kernel_shape}, weights={weights_size}, "
            f"range dimensions=({expected_i}, {expected_j})."
        )


def symbolic_evaluator_optimized_stacked(
        eval_inputs: list[EvaluatorInput],
        weights_list: list[np.ndarray],
        options_list: list[InterpolationOptions],
) -> list[ExportedFields]:
    """Evaluate a FLAT chunk, attaching failure-only tensor and execution metadata."""
    diagnostics = {
        'stage': 'collect_sizes',
        'backend_at_entry': (BackendTensor.engine_backend, BackendTensor.use_gpu, BackendTensor.dtype,
                             BackendTensor.use_pykeops, BackendTensor.pykeops_enabled, BackendTensor.COMPUTE_GRADS),
    }
    try:
        return _symbolic_evaluator_optimized_stacked(eval_inputs, weights_list, options_list, diagnostics)
    except Exception as exc:
        try:
            details = _stacked_failure_details(eval_inputs, weights_list, options_list, diagnostics)
        except Exception as diagnostic_exc:
            details = f"Diagnostic collection failed: {type(diagnostic_exc).__name__}: {diagnostic_exc}"
        raise RuntimeError(
            f"FLAT PyKeOps evaluation failed at stage={diagnostics['stage']}: "
            f"{type(exc).__name__}: {exc}\n{details}"
        ) from exc


def _stacked_failure_details(eval_inputs, weights_list, options_list, diagnostics) -> str:
    """Describe metadata only, without copying tensors or synchronizing CUDA."""
    import os
    import platform
    import threading
    from importlib.metadata import version

    def describe(value):
        if value is None or isinstance(value, (int, float)):
            return repr(value)
        metadata = {'type': type(value).__name__, 'id': hex(id(value))}
        try:
            for attr in ('shape', 'ni', 'nj', 'dtype', 'device', 'requires_grad'):
                if hasattr(value, attr):
                    metadata[attr] = getattr(value, attr)
            if hasattr(value, 'stride'):
                metadata['stride'] = value.stride()
            elif hasattr(value, 'strides'):
                metadata['strides_bytes'] = value.strides
            if hasattr(value, 'variables'):
                metadata['variables'] = [describe(var) for var in value.variables]
        except Exception as exc:
            metadata['metadata_error'] = f"{type(exc).__name__}: {exc}"
        return str(metadata)

    lines = [
        f"evaluation_id={hex(id(diagnostics))}, pid={os.getpid()}, thread={threading.current_thread().name}, "
        f"python={platform.python_version()}",
        f"backend_at_entry (backend, gpu, dtype, use_pykeops, pykeops_enabled, grads)="
        f"{diagnostics['backend_at_entry']}",
        f"backend={BackendTensor.engine_backend}, use_gpu={BackendTensor.use_gpu}, "
        f"dtype={BackendTensor.dtype}, use_pykeops={BackendTensor.use_pykeops}, "
        f"pykeops_enabled={BackendTensor.pykeops_enabled}, grads={BackendTensor.COMPUTE_GRADS}",
        f"numpy={np.__version__}",
        f"list lengths: inputs={len(eval_inputs)}, weights={len(weights_list)}, options={len(options_list)}; "
        f"GEMPY_MAX_CHUNK_SIZE={os.getenv('GEMPY_MAX_CHUNK_SIZE', '32_000_000_000')}",
    ]
    try:
        lines.append(f"pykeops={version('pykeops')}")
    except Exception as exc:
        lines.append(f"PyKeOps version unavailable: {exc}")
    if BackendTensor.engine_backend == gempy_engine.config.AvailableBackends.PYTORCH:
        import torch
        lines.append(f"torch={torch.__version__}, cuda={torch.version.cuda}")
        if BackendTensor.use_gpu:
            try:
                device = torch.cuda.current_device()
                lines.append(
                    f"CUDA device={device}, allocated_bytes={torch.cuda.memory_allocated(device)}, "
                    f"reserved_bytes={torch.cuda.memory_reserved(device)}"
                )
            except Exception as exc:
                lines.append(f"CUDA memory metadata unavailable: {exc}")
    for group, ei in enumerate(eval_inputs):
        lines.append(f"group {group}: input_id={hex(id(ei))}, grid={describe(ei.xyz_to_interpolate)}")
        if group < len(weights_list):
            lines.append(f"group {group}: weights={describe(weights_list[group])}")
        if group < len(options_list):
            opt = options_list[group]
            kernel = opt.kernel_options
            settings = {key: getattr(kernel, key) for key in (
                'kernel_function', 'range', 'c_o', 'uni_degree', 'number_dimensions', 'i_res', 'gi_res',
            )}
            settings = {key: describe(value) if hasattr(value, 'shape') else value
                        for key, value in settings.items()}
            lines.append(
                f"group {group}: options_id={hex(id(opt))}, kernel={settings}, "
                f"compute_scalar={opt.evaluation_options.compute_scalar}, "
                f"compute_scalar_gradient={opt.evaluation_options.compute_scalar_gradient}, "
                f"deduplicate_corners={opt.evaluation_options.deduplicate_octree_corners}"
            )
    for name in ('M_sizes', 'N_sizes', 'tile_factor', 'prep_layout'):
        if name in diagnostics:
            lines.append(f"{name}={diagnostics[name]}")
    for label, kernels in (
        ('prepared', diagnostics.get('kernel_data_list', [])),
        ('stacked', [diagnostics['concat_kernel_data']] if 'concat_kernel_data' in diagnostics else []),
    ):
        for group, kernel in enumerate(kernels):
            for struct_name, struct in vars(kernel).items():
                if struct is None:
                    lines.append(f"{label}[{group}].{struct_name}=None")
                elif hasattr(struct, '__dict__'):
                    for field, value in vars(struct).items():
                        lines.append(f"{label}[{group}].{struct_name}.{field}: {describe(value)}")
    for name in ('eval_kernel_scalar', 'eval_kernel_grad', 'eval_kernel', 'lazy_weights', 'all_weights'):
        if name in diagnostics:
            lines.append(f"{name}: {describe(diagnostics[name])}")
    if 'ranges' in diagnostics:
        lines.append(f"ranges: {[describe(value) for value in diagnostics['ranges']]}")
    return '\n'.join(lines)


def _symbolic_evaluator_optimized_stacked(eval_inputs, weights_list, options_list, diagnostics):
    """Evaluate multiple fields in a single PyKeOps call using block-sparse ranges.
    
    Concatenates all evaluation inputs across stacks, builds block-diagonal ranges,
    performs a single PyKeOps reduction, then splits results back per field.
    """

    n_fields = len(eval_inputs)

    # Collect sizes: M = grid/eval points (j-dim), N = weights/cov_size (i-dim)
    M_sizes = [ei.xyz_to_interpolate.shape[0] for ei in eval_inputs]
    N_sizes = [w.shape[0] for w in weights_list]
    diagnostics.update(M_sizes=M_sizes, N_sizes=N_sizes)

    kernel_data_list = []

    # 2. Define a small wrapper function for the executor to map over
    def _run_prep(args):
        ei, axis, opt, group = args
        try:
            # noinspection PyTypeChecker
            return evaluation_vectors_preparations(
                ei,
                opt.kernel_options,
                axis=axis,
                slice_array=None
            )
        except Exception as exc:
            raise RuntimeError(f"Input preparation failed for group={group}, gradient_axis={axis}: {exc}") from exc

    # We assume all options have the same compute_scalar and compute_scalar_gradient
    # For now, we take from the first one
    base_options = options_list[0]

    if base_options.compute_scalar is True:
        diagnostics['stage'] = 'prepare_scalar_inputs'
        prep_tasks = [(ei, None, options_list[idx], idx) for idx, ei in enumerate(eval_inputs)]
        diagnostics['prep_layout'] = [(idx, None) for idx in range(n_fields)]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            kernel_data_list = list(executor.map(_run_prep, prep_tasks))

        diagnostics.update(stage='stack_scalar_inputs', kernel_data_list=kernel_data_list)
        concat_kernel_data: KernelInput = _build_stacked_kernel_data(kernel_data_list)

        diagnostics.update(stage='create_scalar_kernel', concat_kernel_data=concat_kernel_data)
        eval_kernel_scalar = create_scalar_kernel(
            concat_kernel_data,
            base_options.kernel_options,
            execution_mode=KernelExecutionMode.SYMBOLIC,
        )
        diagnostics['eval_kernel_scalar'] = eval_kernel_scalar

    if base_options.compute_scalar_gradient is True:
        diagnostics['stage'] = 'prepare_gradient_inputs'
        diagnostics.pop('kernel_data_list', None)
        diagnostics.pop('concat_kernel_data', None)
        prep_tasks = []
        for idx, ei in enumerate(eval_inputs):
            prep_tasks.append((ei, 0, options_list[idx], idx))  # X gradient
        for idx, ei in enumerate(eval_inputs):
            prep_tasks.append((ei, 1, options_list[idx], idx))  # Y gradient
        for idx, ei in enumerate(eval_inputs):
            prep_tasks.append((ei, 2, options_list[idx], idx))  # Z gradient

        diagnostics['prep_layout'] = [(task[3], task[1]) for task in prep_tasks]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            kernel_data_list = list(executor.map(_run_prep, prep_tasks))

        diagnostics.update(stage='stack_gradient_inputs', kernel_data_list=kernel_data_list)
        concat_kernel_data: KernelInput = _build_stacked_kernel_data(kernel_data_list)

        diagnostics.update(stage='create_gradient_kernel', concat_kernel_data=concat_kernel_data)
        eval_kernel_grad = create_grad_kernel(
            concat_kernel_data,
            base_options.kernel_options,
            execution_mode=KernelExecutionMode.SYMBOLIC,
        )
        diagnostics['eval_kernel_grad'] = eval_kernel_grad

    # region kernels
    diagnostics['stage'] = 'combine_kernels'
    match (base_options.compute_scalar, base_options.compute_scalar_gradient):
        case (True, True):
            # Concatenate eval kernel
            eval_kernel = BackendTensor.t.concatenate([eval_kernel_scalar, eval_kernel_grad], axis=1)
            tile_factor = 4
        case (True, False):
            eval_kernel = eval_kernel_scalar
            tile_factor = 1
        case (False, True):
            eval_kernel = eval_kernel_grad
            tile_factor = 3
        case (False, False):
            raise ValueError("Cannot compute scalar and scalar gradient simultaneously")
    # endregion

    M_sizes = M_sizes * tile_factor
    N_sizes = N_sizes * tile_factor
    diagnostics.update(stage='build_ranges_and_weights', eval_kernel=eval_kernel,
                       M_sizes=M_sizes, N_sizes=N_sizes, tile_factor=tile_factor)
    # Build block-sparse ranges
    ranges = _build_block_sparse_ranges(M_sizes, N_sizes)
    diagnostics['ranges'] = ranges
    # Concatenate weights
    all_weights = BackendTensor.t.concatenate(weights_list, axis=0)
    all_weights = BackendTensor.t.tile(all_weights, tile_factor)
    diagnostics.update(stage='validate_stacked_dimensions', all_weights=all_weights)
    _validate_stacked_dimensions(eval_kernel, all_weights, M_sizes, N_sizes)

    diagnostics['stage'] = 'prepare_lazy_weights'
    if BackendTensor.engine_backend == gempy_engine.config.AvailableBackends.numpy:
        from pykeops.numpy import LazyTensor
        all_weights_np = BackendTensor.t.to_numpy(all_weights)
        lazy_weights = LazyTensor(np.asfortranarray(all_weights_np.reshape(-1, 1)), axis=0)
        
        # Ensure eval_kernel is also a LazyTensor
        import pykeops.numpy
        if not isinstance(eval_kernel, pykeops.numpy.LazyTensor):
             print(f"DEBUG: eval_kernel type: {type(eval_kernel)}")
             if isinstance(eval_kernel, np.ndarray):
                 eval_kernel = LazyTensor(eval_kernel)

        backend_string = "GPU" if BackendTensor.use_gpu else "CPU"
        diagnostics.update(stage='weighted_reduction', lazy_weights=lazy_weights)
        all_results_concat: np.ndarray = (eval_kernel * lazy_weights).sum(
            axis=0,
            backend=backend_string,
            ranges=ranges
        ).reshape(-1)
    else:
        from pykeops.torch import LazyTensor
        try:
            if BackendTensor.use_gpu:
                all_weights = all_weights.to("cuda", non_blocking=True)
                diagnostics['all_weights'] = all_weights
            lazy_weights = LazyTensor(all_weights.view((-1, 1)), axis=0)

            backend_string = "GPU" if BackendTensor.use_gpu else "CPU"
            diagnostics.update(stage='weighted_reduction', lazy_weights=lazy_weights)
            all_results_concat = (eval_kernel * lazy_weights).sum(
                axis=0,
                backend=backend_string,
                ranges=ranges
            ).reshape(-1)

            # 2. Add explicit synchronization for eGPU stability
            if BackendTensor.use_gpu:
                diagnostics['stage'] = 'cuda_synchronize'
                import torch
                torch.cuda.synchronize()
        except TypeError as exc:
            raise ValueError("Failed to compute symbolic evaluation with PyKeOps. Ensure that all_weights and eval_kernel are compatible for lazy tensor operations.") from exc

    diagnostics['stage'] = 'split_results'
    if BackendTensor.engine_backend == gempy_engine.config.AvailableBackends.numpy:
        split_indices = np.cumsum(M_sizes)[:-1]
        all_results_split = np.split(all_results_concat, split_indices)
    else:
        all_results_split = BackendTensor.t.split(all_results_concat, M_sizes)

    original_n_fields = len(eval_inputs)
    scalar_fields = all_results_split[:original_n_fields]
    gx_fields = [None] * original_n_fields
    gy_fields = [None] * original_n_fields
    gz_fields = [None] * original_n_fields

    match (base_options.compute_scalar, base_options.compute_scalar_gradient):
        case (True, True):
            # Concatenate eval kernel
            gx_fields = all_results_split[original_n_fields: 2 * original_n_fields]
            gy_fields = all_results_split[2 * original_n_fields: 3 * original_n_fields]
            gz_fields = all_results_split[3 * original_n_fields: 4 * original_n_fields]
        case (True, False):
            pass
        case (False, True):
            gx_fields = all_results_split[0 * original_n_fields: 1 * original_n_fields]
            gy_fields = all_results_split[1 * original_n_fields: 2 * original_n_fields]
            gz_fields = all_results_split[2 * original_n_fields: 3 * original_n_fields]
        case (False, False):
            raise ValueError("Cannot compute scalar and scalar gradient simultaneously")

    # Build ExportedFields per stack
    diagnostics['stage'] = 'build_exported_fields'
    results = []
    for idx in range(n_fields):
        # We need to make sure the results are numpy arrays if using numpy backend
        s_field = scalar_fields[idx]
        gx_field = gx_fields[idx]
        gy_field = gy_fields[idx]
        gz_field = gz_fields[idx]

        if BackendTensor.engine_backend == gempy_engine.config.AvailableBackends.numpy:
            s_field = BackendTensor.t.to_numpy(s_field)
            if gx_field is not None: gx_field = BackendTensor.t.to_numpy(gx_field)
            if gy_field is not None: gy_field = BackendTensor.t.to_numpy(gy_field)
            if gz_field is not None: gz_field = BackendTensor.t.to_numpy(gz_field)

        fields = ExportedFields(s_field, gx_field, gy_field, gz_field)
        results.append(fields)

    return results


import concurrent.futures


def _build_stacked_kernel_data(kernel_data_list: list[KernelInput]) -> KernelInput:
    """Build a stacked KernelInput by concatenating the internal arrays sequentially."""
    from ..kernel_constructor._structs import (
        OrientationSurfacePointsCoords, CartesianSelector, OrientationsDrift,
        PointsDrift, FaultDrift, DriftMatrixSelector
    )
    j_fields = {
        OrientationSurfacePointsCoords: {'dip_ref_j', 'diprest_j'},
        CartesianSelector: {'hu_sel_j', 'hv_sel_j', 'h_sel_ref_j', 'h_sel_rest_j'},
        OrientationsDrift: {'dips_ug_aj', 'dips_ug_bj', 'dips_ug_cj', 'selector_cj'},
        PointsDrift: {'dipsPoints_ui_aj', 'dipsPoints_ui_bj1', 'dipsPoints_ui_bj2'},
        FaultDrift: {'faults_j'},
        DriftMatrixSelector: {'sel_vj'},
    }
    sizes = [(kd.ori_sp_matrices.dip_ref_i.shape[0], kd.ori_sp_matrices.dip_ref_j.shape[1])
             for kd in kernel_data_list]

    def _stack_sub_struct_split(items, cls, struct_name):
        """Concatenate fields with explicitly split steps for easier profiling."""
        if any((item is None) != (items[0] is None) for item in items):
            raise ValueError(f"Inconsistent stacked PyKeOps field {struct_name}: present in only some groups")
        if items[0] is None:
            return None

        result = cls.__new__(cls)
        first_item = items[0]

        # --- Helper functions for profiling ---
        def _extract_values(item_list, field):
            res = []
            for item in item_list:
                val = getattr(item, field)
                # If we encounter a LazyTensor, we try to get the original tensor
                try:
                    # noinspection PyUnresolvedReferences
                    from pykeops.torch import LazyTensor as LazyTensorTorch
                    # noinspection PyUnresolvedReferences
                    from pykeops.numpy import LazyTensor as LazyTensorNumpy
                    if isinstance(val, (LazyTensorTorch, LazyTensorNumpy)):
                        val = val.variables[0]  # This is a bit hacky, depends on PyKeOps version
                except (ImportError, AttributeError):
                    pass
                res.append(val)
            return res


        def _concatenate_tensors(tensor_list, concat_axis):
            # Time spent here is strictly the memory allocation and copying
            return BackendTensor.t.concatenate(tensor_list, axis=concat_axis)

        # --------------------------------------

        for field_name in first_item.__dict__:
            # 1. Extraction
            vals = _extract_values(items, field_name)
            first_val = vals[0]

            if first_val is None:
                setattr(result, field_name, None)
                continue

            if isinstance(first_val, int):
                if any(val != first_val for val in vals):
                    raise ValueError(
                        f"Inconsistent stacked PyKeOps field {struct_name}.{field_name}: values={vals}"
                    )
                setattr(result, field_name, first_val)
                continue

            # Check if it's already a LazyTensor (it should not be before upgrade_tensors, but safety first)
            try:
                # noinspection PyUnresolvedReferences
                from pykeops.torch import LazyTensor as LazyTensorTorch
                # noinspection PyUnresolvedReferences
                from pykeops.numpy import LazyTensor as LazyTensorNumpy
                if isinstance(first_val, (LazyTensorTorch, LazyTensorNumpy)):
                    # This path is tricky, usually we stack raw tensors
                    setattr(result, field_name, first_val)
                    continue
            except ImportError:
                pass

            # Singleton dimensions cannot identify whether a field is i- or j-indexed.
            axis = 1 if field_name in j_fields[cls] else 0
            shapes = [tuple(val.shape) for val in vals]
            for group, (shape, (n_i, n_j)) in enumerate(zip(shapes, sizes)):
                expected_size = n_j if axis == 1 else n_i
                # This covariance selector is unused in evaluation and retains the constraint count.
                if cls is OrientationsDrift and field_name == 'selector_cj':
                    expected_size = n_i
                if (len(shape) != 3 or shape[1 - axis] != 1
                        or shape[axis] != expected_size or shape[2] != shapes[0][2]):
                    raise ValueError(
                        f"Inconsistent stacked PyKeOps field {struct_name}.{field_name} in group {group}: "
                        f"shapes={shapes}, axis={axis}, expected size={expected_size}, "
                        f"constraint/grid sizes={sizes}"
                    )

            # 2 & 3. Transfer and Concatenate
            # noinspection PyUnresolvedReferences
            import gempy_engine.config
            import torch
            if BackendTensor.engine_backend == gempy_engine.config.AvailableBackends.numpy:
                tensor_vals = []
                for v in vals:
                    if not isinstance(v, np.ndarray):
                        v = np.array(v)
                    tensor_vals.append(v.astype(BackendTensor.dtype_obj))
                concat_val = _concatenate_tensors(tensor_vals, axis).copy()
            else:
                # noinspection PyUnresolvedReferences
                tensor_vals = []
                for v in vals:
                    if not isinstance(v, torch.Tensor):
                        v = BackendTensor.t.array(v)
                    if BackendTensor.use_gpu:
                        v = v.to("cuda", non_blocking=True)
                    tensor_vals.append(v.type(BackendTensor.dtype_obj))
                concat_val = _concatenate_tensors(tensor_vals, axis).contiguous()

            setattr(result, field_name, concat_val)

        return result.upgrade_tensors()
        # return result


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
    if any(getattr(kd, 'ref_fault', None) is not None for kd in kernel_data_list):
        tasks.extend([
                ("ref_fault", FaultDrift),
                ("rest_fault", FaultDrift)
        ])
    else:
        # Explicitly set them to None on the stacked object to match your original structure
        stacked.ref_fault = None
        stacked.rest_fault = None

    # Execute concatenation sequentially
    for field_name, cls in tasks:
        try:
            items_to_stack = [getattr(kd, field_name) for kd in kernel_data_list]
            stacked_result = _stack_sub_struct_split(items_to_stack, cls, field_name)
            setattr(stacked, field_name, stacked_result)
        except Exception as exc:
            raise RuntimeError(f"Stacking field {field_name} generated an exception: {exc}") from exc

    # Handle scalars
    stacked.nugget_scalar = kernel_data_list[0].nugget_scalar
    stacked.nugget_grad = kernel_data_list[0].nugget_grad

    return stacked
