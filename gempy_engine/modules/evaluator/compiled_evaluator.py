"""Evaluation kernels built once per stack and compute, called with each new grid.

The KeOps formula of a stack's evaluation (scalar field and gradient) is the same at every octree
level and in every extraction query: only the evaluation points, the fault values on them and the
weights change. ``compiled_evaluation`` builds the reduction once with those three as symbolic
KeOps variables, the grid-independent data side as concrete tensors, and keeps it in the stack
cache; later evaluations just call it. Same formula and same inputs as building it per call, so the
results are the same.
"""
import numpy as np

from ._keops_reduction import pack_constants
from ..kernel_constructor._internalDistancesMatrices import DistancesBuffer
from ..kernel_constructor._kernels_assembler import create_scalar_kernel, create_grad_kernel
from ..kernel_constructor._structs import (KernelInput, OrientationSurfacePointsCoords, OrientationsDrift, PointsDrift,
                                           FaultDrift, CartesianSelector, DriftMatrixSelector)
from ..kernel_constructor._vectors_preparation import _evaluation_data_side
from ..kernel_constructor.execution_mode import KernelExecutionMode
from ...core.backend_tensor import BackendTensor
from ...core.data.exported_fields import ExportedFields
from ...core.data.matrices_sizes import MatricesSizes
from ...core.host_arrays import upload


def compiled_evaluation(solver_input, weights, options, cache) -> ExportedFields:
    faults = solver_input.fault_internal
    n_faults = faults.n_faults
    gradient = options.compute_scalar_gradient is True
    if gradient and options.number_dimensions not in (2, 3):
        raise ValueError("Number of dimensions have to be 2 or 3")
    backend = "GPU" if BackendTensor.use_gpu else "CPU"
    key = ("evaluation", gradient, options.number_dimensions, n_faults, backend)
    reduction = cache.compiled.get(key)
    if reduction is None:
        reduction = cache.compiled[key] = _compile(solver_input, options, n_faults, gradient, backend)

    args = [solver_input.xyz_to_interpolate.contiguous()]
    if n_faults:
        args.append(BackendTensor.t.ascontiguousarray(faults.fault_values_everywhere.T))
    args.append(weights.reshape(-1, 1))
    n_outputs = 1 + (options.number_dimensions if gradient else 0)
    fields = reduction(*args).reshape(-1, n_outputs)
    columns = [fields[:, k].contiguous() for k in range(n_outputs)]
    return ExportedFields(columns[0], *(columns[1:] + [None] * (4 - n_outputs)))


def _compile(solver_input, options, n_faults, gradient, backend):
    from pykeops.torch import LazyTensor
    grid = LazyTensor((0, options.number_dimensions, 1))
    faults = LazyTensor((1, n_faults, 1)) if n_faults else None
    weights = LazyTensor((2 if n_faults else 1, 1, 0))
    distance_buffer, distance_key = (DistancesBuffer(), object()) if gradient else (None, None)

    kernels = [create_scalar_kernel(symbolic_kernel_input(solver_input, options.kernel_options, None, grid, faults),
                                    options.kernel_options, execution_mode=KernelExecutionMode.SYMBOLIC,
                                    distance_buffer=distance_buffer, distance_cache_key=distance_key)]
    if gradient:
        kernels += [create_grad_kernel(symbolic_kernel_input(solver_input, options.kernel_options, axis, grid, faults),
                                       options.kernel_options, execution_mode=KernelExecutionMode.SYMBOLIC,
                                       distance_buffer=distance_buffer, distance_cache_key=distance_key)
                    for axis in range(options.number_dimensions)]
    stacked = kernels[0]
    for kernel in kernels[1:]:
        stacked = stacked.concat(kernel)
    packed, _ = pack_constants(stacked * weights)  # one upload for the formula's float constants
    return packed.sum(axis=0, backend=backend, call=False)


# region Stacked (fused) evaluation: several stacks in one block-sparse reduction

# Grid-side terms that are the same in every row of a block; per block they may differ (gradient axis).
_ROW_FIELDS = ("hu_sel_j", "hv_sel_j", "h_sel_ref_j", "h_sel_rest_j", "grid_1", "sel_vj", "fault_sel_vj")


def compiled_stacked_evaluation(eval_inputs, weights_list, options_list, caches, shared):
    """Output of the stacked block-sparse reduction (blocks: stacks, per gradient axis) and its block sizes.

    The same formula and layout as building the stacked kernel per level
    (``_built_stacked_evaluation``), built once per chunk and compute: the concatenated data side is
    concrete; the concatenated grids, fault values, each block's constant rows (repeated by its grid
    size) and the weights are symbolic, and the block-sparse ranges are passed per call.
    """
    import torch
    base = options_list[0]
    axes = [None] if base.compute_scalar else [0, 1, 2]
    blocks = [(idx, axis) for axis in axes for idx in range(len(eval_inputs))]
    n_faults = eval_inputs[0].fault_internal.n_faults
    backend = "GPU" if BackendTensor.use_gpu else "CPU"
    key = ("stacked", tuple(id(cache) for cache in caches), tuple(axes), n_faults, backend)
    entry = shared.get(key)
    if entry is None:
        entry = shared[key] = _compile_stacked(eval_inputs, options_list, blocks, n_faults, backend)
    reduction, order, rows = entry

    block_sizes = [eval_inputs[idx].xyz_to_interpolate.shape[0] for idx, _ in blocks]
    weight_sizes = [weights_list[idx].shape[0] for idx, _ in blocks]
    counts = upload(np.asarray(block_sizes, dtype=np.int64))
    values = {}
    for name in order:
        if name == "grid":
            values[name] = torch.cat([eval_inputs[idx].xyz_to_interpolate for idx, _ in blocks])
        elif name == "faults":
            values[name] = torch.cat([BackendTensor.t.ascontiguousarray(eval_inputs[idx].fault_internal.fault_values_everywhere.T)
                                      for idx, _ in blocks])
        elif name == "weights":
            values[name] = torch.cat([weights_list[idx] for idx, _ in blocks]).reshape(-1, 1)
        else:
            values[name] = torch.repeat_interleave(rows[name], counts, dim=0, output_size=sum(block_sizes))
    from .symbolic_evaluator import _build_block_sparse_ranges
    ranges = _build_block_sparse_ranges(block_sizes, weight_sizes)
    return reduction(*[values[name] for name in order], ranges=ranges).reshape(-1), block_sizes


def _compile_stacked(eval_inputs, options_list, blocks, n_faults, backend):
    import torch
    from pykeops.torch import LazyTensor
    dtype = BackendTensor.dtype_obj
    datas = [_data_side(eval_inputs[idx], options_list[idx].kernel_options, axis) for idx, axis in blocks]
    fault_selectors = [DriftMatrixSelector(x_size=data.fault_vector_ref.shape[0], y_size=1, n_drift_eq=n_faults,
                                           drift_start_post_x=data.fault_vector_ref.shape[0] - n_faults,
                                           drift_start_post_y=1) for data in datas] if n_faults else None
    row_of = {"hu_sel_j": lambda d, k: d.cartesian_selector.hu_sel_j, "hv_sel_j": lambda d, k: d.cartesian_selector.hv_sel_j,
              "h_sel_ref_j": lambda d, k: d.cartesian_selector.h_sel_ref_j,
              "h_sel_rest_j": lambda d, k: d.cartesian_selector.h_sel_rest_j,
              "grid_1": lambda d, k: d.grid_1, "sel_vj": lambda d, k: d.drift_selection.sel_vj,
              "fault_sel_vj": lambda d, k: fault_selectors[k].sel_vj}
    names = ["grid"] + [name for name in _ROW_FIELDS if n_faults or name != "fault_sel_vj"]
    names += (["faults"] if n_faults else []) + ["weights"]
    rows = {name: torch.cat([row_of[name](data, k).reshape(1, -1).type(dtype) for k, data in enumerate(datas)])
            for name in names if name in row_of}

    def cat_i(get, shaped=False):  # concatenated data side, (sum n, 1, d)
        values = torch.cat([get(data, k).type(dtype) for k, data in enumerate(datas)])
        return LazyTensor(values if shaped else values[:, None, :])

    i_side = {
        "dips_ref_coord": cat_i(lambda d, k: d.dips_ref_coord), "dips_rest_coord": cat_i(lambda d, k: d.dips_rest_coord),
        "dips_ug_d1": cat_i(lambda d, k: d.dips_ug_d1), "dips_ug_d2a": cat_i(lambda d, k: d.dips_ug_d2a),
        "dips_ug_d2b_sel": cat_i(lambda d, k: d.dips_ug_d2b_sel),
        "second_degree_selector": cat_i(lambda d, k: d.second_degree_selector),
        "dips_ref_d1": cat_i(lambda d, k: d.dips_ref_d1), "dips_ref_d2a": cat_i(lambda d, k: d.dips_ref_d2a),
        "dips_ref_d2b": cat_i(lambda d, k: d.dips_ref_d2b), "dips_rest_d1": cat_i(lambda d, k: d.dips_rest_d1),
        "dips_rest_d2a": cat_i(lambda d, k: d.dips_rest_d2a), "dips_rest_d2b": cat_i(lambda d, k: d.dips_rest_d2b),
        "hu_sel_i": cat_i(lambda d, k: d.cartesian_selector.hu_sel_i, True),
        "hv_sel_i": cat_i(lambda d, k: d.cartesian_selector.hv_sel_i, True),
        "h_sel_ref_i": cat_i(lambda d, k: d.cartesian_selector.h_sel_ref_i, True),
        "h_sel_rest_i": cat_i(lambda d, k: d.cartesian_selector.h_sel_rest_i, True),
        "sel_ui": cat_i(lambda d, k: d.drift_selection.sel_ui, True),
    }
    if n_faults:
        i_side["fault_vector_ref"] = cat_i(lambda d, k: d.fault_vector_ref)
        i_side["fault_sel_ui"] = cat_i(lambda d, k: fault_selectors[k].sel_ui, True)

    def build(symbolic_names):
        symbols = {name: LazyTensor((index, (1 if name == "weights" else rows[name].shape[1] if name in rows
                                             else (n_faults if name == "faults" else 3)),
                                     0 if name == "weights" else 1))
                   for index, name in enumerate(symbolic_names)}
        for name in names:  # unused placeholders never reach the formula
            symbols.setdefault(name, LazyTensor(rows[name][:1][None] if name in rows else
                                                torch.zeros((1, 1, n_faults if name == "faults" else 3), dtype=dtype)))
        ki = _stacked_kernel_input(i_side, symbols, n_faults)
        options = options_list[0]
        if options.compute_scalar:
            kernel = create_scalar_kernel(ki, options.kernel_options, execution_mode=KernelExecutionMode.SYMBOLIC)
        else:
            kernel = create_grad_kernel(ki, options.kernel_options, execution_mode=KernelExecutionMode.SYMBOLIC)
        return kernel * symbols["weights"]

    expression = build(names)
    used = {variable[0] for variable in expression.symbolic_variables}
    order = [name for index, name in enumerate(names) if index in used]
    if len(order) != len(names):
        expression = build(order)
    packed, _ = pack_constants(expression)
    return packed.sum(axis=0, backend=backend, call=False), order, rows


def _stacked_kernel_input(i_side, symbols, n_faults) -> KernelInput:
    grid = symbols["grid"]
    ki = KernelInput(
        ori_sp_matrices=_struct(OrientationSurfacePointsCoords, dip_ref_i=i_side["dips_ref_coord"], dip_ref_j=grid,
                                diprest_i=i_side["dips_rest_coord"], diprest_j=grid),
        cartesian_selector=_struct(CartesianSelector,
                                   hu_sel_i=i_side["hu_sel_i"], hu_sel_j=symbols["hu_sel_j"],
                                   hv_sel_i=i_side["hv_sel_i"], hv_sel_j=symbols["hv_sel_j"],
                                   h_sel_ref_i=i_side["h_sel_ref_i"], h_sel_ref_j=symbols["h_sel_ref_j"],
                                   h_sel_rest_i=i_side["h_sel_rest_i"], h_sel_rest_j=symbols["h_sel_rest_j"]),
        nugget_scalar=None,
        nugget_grad=None,
        ori_drift=_struct(OrientationsDrift,
                          dips_ug_ai=i_side["dips_ug_d1"], dips_ug_aj=symbols["grid_1"],
                          dips_ug_bi=i_side["dips_ug_d2a"], dips_ug_bj=grid * symbols["grid_1"],
                          dips_ug_ci=i_side["dips_ug_d2b_sel"], dips_ug_cj=grid,
                          selector_ci=i_side["second_degree_selector"], selector_cj=None),
        ref_drift=_struct(PointsDrift, dipsPoints_ui_ai=i_side["dips_ref_d1"], dipsPoints_ui_aj=grid,
                          dipsPoints_ui_bi1=i_side["dips_ref_d2a"], dipsPoints_ui_bj1=grid,
                          dipsPoints_ui_bi2=i_side["dips_ref_d2b"], dipsPoints_ui_bj2=grid),
        rest_drift=_struct(PointsDrift, dipsPoints_ui_ai=i_side["dips_rest_d1"], dipsPoints_ui_aj=grid,
                           dipsPoints_ui_bi1=i_side["dips_rest_d2a"], dipsPoints_ui_bj1=grid,
                           dipsPoints_ui_bi2=i_side["dips_rest_d2b"], dipsPoints_ui_bj2=grid),
        drift_matrix_selector=_struct(DriftMatrixSelector, sel_ui=i_side["sel_ui"], sel_vj=symbols["sel_vj"]),
        ref_fault=(_struct(FaultDrift, faults_i=i_side["fault_vector_ref"], faults_j=symbols["faults"],
                           n_faults_i=n_faults) if n_faults else None),
        rest_fault=None,
    )
    if n_faults:
        # Every block's own trailing fault rows (see _stacked_fault_drift_selector).
        ki.fault_drift_selector = _struct(DriftMatrixSelector, sel_ui=i_side["fault_sel_ui"], sel_vj=symbols["fault_sel_vj"])
    return ki

# endregion


def _data_side(interp_input, kernel_options, axis):
    sp_, ori_ = interp_input.sp_internal, interp_input.ori_internal
    sizes = MatricesSizes(ori_size=ori_.n_orientations_tiled, sp_size=sp_.n_points,
                          uni_drift_size=kernel_options.n_uni_eq, faults_size=interp_input.fault_internal.n_faults,
                          dim=kernel_options.number_dimensions, n_dips=ori_.n_orientations, grid_size=1)
    return _evaluation_data_side(interp_input, kernel_options, sizes, axis)


def symbolic_kernel_input(interp_input, kernel_options, axis, grid, faults) -> KernelInput:
    """Evaluation KernelInput of LazyTensors: the data side concrete (cached), the grid side symbolic.

    Mirrors ``evaluation_vectors_preparations`` + ``upgrade_tensors``; ``grid * grid_1`` is formed inside
    the formula (grid_1 is a one-row parameter).
    """
    from pykeops.torch import LazyTensor
    data = _data_side(interp_input, kernel_options, axis)
    dtype = BackendTensor.dtype_obj

    def i_side(x):
        return LazyTensor(x.type(dtype)[:, None, :])

    def upgraded(struct):  # cached selector structs, already shaped (i: (n, 1, d), j: (1, 1, d))
        return _struct(type(struct), **{name: LazyTensor(value.type(dtype)) for name, value in vars(struct).items()})

    grid_1 = LazyTensor(data.grid_1.type(dtype)[None, :, :])
    return KernelInput(
        ori_sp_matrices=_struct(OrientationSurfacePointsCoords, dip_ref_i=i_side(data.dips_ref_coord), dip_ref_j=grid,
                                diprest_i=i_side(data.dips_rest_coord), diprest_j=grid),
        cartesian_selector=upgraded(data.cartesian_selector),
        nugget_scalar=None,
        nugget_grad=None,
        ori_drift=_struct(OrientationsDrift,
                          dips_ug_ai=i_side(data.dips_ug_d1), dips_ug_aj=grid_1,
                          dips_ug_bi=i_side(data.dips_ug_d2a), dips_ug_bj=grid * grid_1,
                          dips_ug_ci=i_side(data.dips_ug_d2b_sel), dips_ug_cj=grid,
                          selector_ci=i_side(data.second_degree_selector), selector_cj=None),
        ref_drift=_points_drift(i_side, grid, data.dips_ref_d1, data.dips_ref_d2a, data.dips_ref_d2b),
        rest_drift=_points_drift(i_side, grid, data.dips_rest_d1, data.dips_rest_d2a, data.dips_rest_d2b),
        drift_matrix_selector=upgraded(data.drift_selection),
        ref_fault=(_struct(FaultDrift, faults_i=i_side(data.fault_vector_ref), faults_j=faults,
                           n_faults_i=data.fault_vector_ref.shape[1]) if faults is not None else None),
        rest_fault=None,
    )


def _points_drift(i_side, grid, d1, d2a, d2b):
    return _struct(PointsDrift, dipsPoints_ui_ai=i_side(d1), dipsPoints_ui_aj=grid,
                   dipsPoints_ui_bi1=i_side(d2a), dipsPoints_ui_bj1=grid,
                   dipsPoints_ui_bi2=i_side(d2b), dipsPoints_ui_bj2=grid)


def _struct(cls, **fields):
    struct = cls.__new__(cls)
    struct.__dict__.update(fields)
    return struct
