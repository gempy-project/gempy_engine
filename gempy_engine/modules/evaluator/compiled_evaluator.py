"""Evaluation kernels built once per stack and compute, called with each new grid.

The KeOps formula of a stack's evaluation (scalar field and gradient) is the same at every octree
level and in every extraction query: only the evaluation points, the fault values on them and the
weights change. ``compiled_evaluation`` builds the reduction once with those three as symbolic
KeOps variables and keeps it in the stack cache; later evaluations just call it.

The grid-independent data side (surface points, orientations, drift and selector vectors) is passed
as call arguments too: every data tensor is a symbolic variable of the category and dimension the
tensor itself would have had (one per tensor, as before). The traced formula then depends only on the
model's structure (kernel options, kernel kind, number of faults, the data's layout), and
``_formula`` keeps it for the process: a later compute of the same structure, e.g. after moving or
adding points, skips the trace. A formula that would still hold a data tensor is never kept beyond
its compute. ``GEMPY_CHECK_COMPUTE_CACHE`` traces again and compares; ``GEMPY_FORMULA_CACHE=False``
traces in every compute.
"""
import copy
import os
import re
import threading
from collections import OrderedDict
from dataclasses import asdict

import numpy as np

from ._keops_reduction import pack_constants
from ..kernel_constructor._internalDistancesMatrices import DistancesBuffer
from ..kernel_constructor._kernels_assembler import create_scalar_kernel, create_grad_kernel
from ..kernel_constructor._structs import (KernelInput, OrientationSurfacePointsCoords, OrientationsDrift, PointsDrift,
                                           FaultDrift, CartesianSelector, DriftMatrixSelector)
from ..kernel_constructor._vectors_preparation import _evaluation_data_side
from ..kernel_constructor.execution_mode import KernelExecutionMode
from ...core.backend_tensor import BackendTensor
from ...core.compute_scope import checking
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
    entry = cache.compiled.get(key)
    if entry is None:
        entry = cache.compiled[key] = _compile(solver_input, options, n_faults, gradient, backend)
    reduction, call_kept, data_arguments = entry

    args = [solver_input.xyz_to_interpolate.contiguous()]
    if n_faults:
        args.append(BackendTensor.t.ascontiguousarray(faults.fault_values_everywhere.T))
    args.append(weights.reshape(-1, 1))
    n_outputs = 1 + (options.number_dimensions if gradient else 0)
    fields = reduction(*[args[index] for index in call_kept], *data_arguments).reshape(-1, n_outputs)
    columns = [fields[:, k].contiguous() for k in range(n_outputs)]
    return ExportedFields(columns[0], *(columns[1:] + [None] * (4 - n_outputs)))


def _compile(solver_input, options, n_faults, gradient, backend):
    """(reduction, call variables kept, data arguments): the reduction takes the kept ones of (grid,
    faults, weights), then the data arguments."""
    from pykeops.torch import LazyTensor
    n_call = 3 if n_faults else 2
    axes = [None] + (list(range(options.number_dimensions)) if gradient else [])

    def kernel_inputs(lazy):
        if lazy:
            grid, weights = LazyTensor((0, options.number_dimensions, 1)), LazyTensor((n_call - 1, 1, 0))
            faults = LazyTensor((1, n_faults, 1)) if n_faults else None
        else:
            grid, weights, faults = _STAND_IN, _STAND_IN, _STAND_IN if n_faults else None
        symbols = _DataSymbols(n_call, lazy)
        inputs = [symbolic_kernel_input(solver_input, options.kernel_options, axis, grid, faults, symbols)
                  for axis in axes]
        return symbols, inputs, weights

    symbols, _, _ = kernel_inputs(lazy=False)  # the arguments and the structure, without KeOps objects

    def trace():
        traced, inputs, weights = kernel_inputs(lazy=True)
        if traced.signature != symbols.signature:
            raise AssertionError("compiled evaluation: the traced data variables differ from the collected ones")
        distance_buffer, distance_key = (DistancesBuffer(), object()) if gradient else (None, None)
        kernels = [create_scalar_kernel(inputs[0], options.kernel_options, execution_mode=KernelExecutionMode.SYMBOLIC,
                                        distance_buffer=distance_buffer, distance_cache_key=distance_key)]
        kernels += [create_grad_kernel(ki, options.kernel_options, execution_mode=KernelExecutionMode.SYMBOLIC,
                                       distance_buffer=distance_buffer, distance_cache_key=distance_key)
                    for ki in inputs[1:]]
        stacked = kernels[0]
        for kernel in kernels[1:]:
            stacked = stacked.concat(kernel)
        return stacked * weights

    structure = ("evaluation", gradient, options.number_dimensions, n_faults,
                 _settings(options.kernel_options), tuple(symbols.signature))
    reduction, kept = _formula(structure, trace, backend)
    return (reduction, [index for index in kept if index < n_call],
            [symbols.argument(index - n_call) for index in kept if index >= n_call])


class _DataSymbols:
    """Symbolic KeOps variables standing for data tensors, numbered from ``first``: one new variable per
    call, as one ``LazyTensor(tensor)`` per call would be, of the category LazyTensor infers from the
    shape ((M,1,D) i, (1,N,D) j, (1,1,D) parameter). ``argument(position)`` is a variable's tensor shaped
    as KeOps takes it (made only for the variables a formula keeps); ``signature`` the dimensions and
    categories. With ``lazy=False`` a call only records the tensor and returns a stand-in: enough to
    find the formula and its arguments."""

    def __init__(self, first, lazy=True):
        self.first, self.lazy, self.tensors, self.signature = first, lazy, [], []

    def argument(self, position):
        tensor, category = self.tensors[position]
        return (tensor[:, 0] if category == 0 else tensor[0] if category == 1 else tensor[0, 0]).contiguous()

    def variable(self, position):
        from pykeops.torch import LazyTensor
        dim, category = self.signature[position]
        return LazyTensor((self.first + position, dim, category))

    def __call__(self, tensor):
        shape = tensor.shape
        if len(shape) != 3 or (shape[0] != 1 and shape[1] != 1):
            raise ValueError("compiled evaluation: data variables are (M,1,D), (1,N,D) or (1,1,D) tensors")
        category = (2 if shape[1] == 1 else 1) if shape[0] == 1 else 0
        self.tensors.append((tensor, category))
        self.signature.append((shape[-1], category))
        return self.variable(len(self.tensors) - 1) if self.lazy else _STAND_IN


class _StandIn:
    """Handed out instead of KeOps variables while only collecting arguments: absorbs the arithmetic
    ``symbolic_kernel_input`` does on them (``grid * grid_1``)."""

    def __mul__(self, other):
        return self

    __rmul__ = __mul__


_STAND_IN = _StandIn()


# Traced reductions by structure, for the process (least recently used dropped first).
FORMULA_CACHE_SIZE = 32
_FORMULAS: OrderedDict = OrderedDict()
_FORMULAS_LOCK = threading.Lock()


def clear_formula_cache():
    with _FORMULAS_LOCK:
        _FORMULAS.clear()


def _formula(structure, trace, backend):
    """(reduction, kept symbol indices) of the formula ``trace()`` builds: compacted (``_compact_symbols``),
    its float constants packed into one parameter, as a callable sum reduction over i.

    Kept for the process when the formula holds no data (only symbolic variables and constants), so later
    computes of the same ``structure`` skip the trace."""
    key = (structure, backend, BackendTensor.dtype, str(BackendTensor.device))
    reuse = os.getenv("GEMPY_FORMULA_CACHE", "True").lower() in ("true", "1", "t", "y", "yes")
    with _FORMULAS_LOCK:
        cached = _FORMULAS.get(key) if reuse else None
        if cached is not None:
            _FORMULAS.move_to_end(key)
    if cached is not None and not checking():
        return cached[0], cached[1]
    expression, kept = _compact_symbols(trace())
    holds_data = any(type(variable) is not list for variable in getattr(expression, "variables", ()))
    packed, _ = pack_constants(expression, dtype=BackendTensor.dtype, device=BackendTensor.device)
    entry = (packed.sum(axis=0, backend=backend, call=False), kept, _formula_text(packed))
    if cached is not None:  # checking(): the kept formula must be the one this structure traces to
        if cached[1:] != entry[1:]:
            raise AssertionError("formula cache: the cached formula differs from a new trace of the same structure")
        return cached[0], cached[1]
    if reuse and not holds_data:
        with _FORMULAS_LOCK:
            _FORMULAS[key] = entry
            while len(_FORMULAS) > FORMULA_CACHE_SIZE:
                _FORMULAS.popitem(last=False)
    return entry[0], entry[1]


def _formula_text(lazy):
    """The formula with the packed constants' tensor id masked (it differs from trace to trace)."""
    return re.sub(r"Var\(\d+,", "Var(constants,", lazy.formula + "|" + (lazy.formula2 or ""))


def _settings(kernel_options):
    """Every kernel setting (not the condition numbers the solver stores), as a hashable key."""
    settings = asdict(kernel_options)
    for runtime in ("condition_number", "condition_number_before", "condition_number_after"):
        settings.pop(runtime, None)
    return tuple(sorted((name, value if isinstance(value, (int, float, str, bool, type(None))) else repr(value))
                        for name, value in settings.items()))


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
    reduction, order, rows, data_arguments = entry

    block_sizes = [eval_inputs[idx].xyz_to_interpolate.shape[0] for idx, _ in blocks]
    weight_sizes = [weights_list[idx].shape[0] for idx, _ in blocks]
    counts, ranges = _block_indices(block_sizes, weight_sizes)
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
    return reduction(*[values[name] for name in order], *data_arguments, ranges=ranges).reshape(-1), block_sizes


def _block_indices(block_sizes: list[int], weight_sizes: list[int]):
    """The blocks' repeat counts and PyKeOps block-sparse ranges from one host copy.

    The same values as ``_build_block_sparse_ranges`` (int32): ranges ``(i, slices, j, j, slices, i)``
    with the i side over the weights and the j side over the grids; the counts are the grid sizes.
    """
    n = len(block_sizes)
    keep_i = np.cumsum([0] + weight_sizes)
    keep_j = np.cumsum([0] + block_sizes)
    host = np.concatenate([block_sizes, np.stack([keep_i[:-1], keep_i[1:]], axis=1).reshape(-1),
                           np.arange(1, n + 1), np.stack([keep_j[:-1], keep_j[1:]], axis=1).reshape(-1)])
    counts, ranges_i, slices, ranges_j = upload(host.astype(np.int32)).split([n, 2 * n, n, 2 * n])
    ranges_i, ranges_j = ranges_i.view(n, 2), ranges_j.view(n, 2)
    return counts, (ranges_i, slices, ranges_j, ranges_j, slices, ranges_i)


def _compile_stacked(eval_inputs, options_list, blocks, n_faults, backend):
    """(reduction, call variables kept, their rows, data arguments): the reduction takes the kept call
    variables (``order``), then the data arguments."""
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
    symbols = _DataSymbols(len(names), lazy=False)

    def cat_i(get, shaped=False):  # concatenated data side, (sum n, 1, d): its data variable's position
        values = torch.cat([get(data, k).type(dtype) for k, data in enumerate(datas)])
        symbols(values if shaped else values[:, None, :])
        return len(symbols.tensors) - 1

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

    options = options_list[0]

    def build():
        call = {name: LazyTensor((index, (1 if name == "weights" else rows[name].shape[1] if name in rows
                                          else (n_faults if name == "faults" else 3)),
                                  0 if name == "weights" else 1))
                for index, name in enumerate(names)}
        ki = _stacked_kernel_input({name: symbols.variable(position) for name, position in i_side.items()}, call,
                                   n_faults)
        if options.compute_scalar:
            kernel = create_scalar_kernel(ki, options.kernel_options, execution_mode=KernelExecutionMode.SYMBOLIC)
        else:
            kernel = create_grad_kernel(ki, options.kernel_options, execution_mode=KernelExecutionMode.SYMBOLIC)
        return kernel * call["weights"]

    structure = ("stacked", options.compute_scalar, n_faults, _settings(options.kernel_options), tuple(names),
                 tuple((name, rows[name].shape[1]) for name in rows), tuple(symbols.signature))
    reduction, kept = _formula(structure, build, backend)
    order = [names[index] for index in kept if index < len(names)]
    return reduction, order, rows, [symbols.argument(index - len(names)) for index in kept if index >= len(names)]


def _compact_symbols(expression):
    """``expression`` with its symbolic variables renumbered 0, 1, ... in index order, and the old indices kept.

    KeOps needs every declared variable index to appear in the formula, and some placeholders (row fields
    of other kernel kinds) do not. Renumbering gives the formula a second trace with only the used
    placeholders would give, without tracing again: the unused ones never entered the formula."""
    formulas = expression.formula + (expression.formula2 or "")
    kept = sorted({int(index) for index in re.findall(r"VarSymb\((\d+),", formulas)})
    if kept != sorted({variable[0] for variable in expression.symbolic_variables}):
        raise ValueError("compiled evaluation: symbolic variables and formula disagree")
    if kept == list(range(len(kept))):
        return expression, kept
    new = {old: position for position, old in enumerate(kept)}

    def renumber(formula):
        return formula and re.sub(r"VarSymb\((\d+),", lambda match: f"VarSymb({new[int(match.group(1))]},", formula)

    compact = copy.copy(expression)
    compact.formula, compact.formula2 = renumber(expression.formula), renumber(expression.formula2)
    compact.symbolic_variables = tuple((new[index], dim, cat) for index, dim, cat in expression.symbolic_variables)
    return compact, kept


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


def symbolic_kernel_input(interp_input, kernel_options, axis, grid, faults, symbols) -> KernelInput:
    """Evaluation KernelInput of LazyTensors: the data side (cached) as ``symbols`` data variables, the grid
    side symbolic.

    Mirrors ``evaluation_vectors_preparations`` + ``upgrade_tensors``; ``grid * grid_1`` is formed inside
    the formula (grid_1 is a one-row parameter). The scalar kernel's fault drift selector, which
    ``create_scalar_kernel`` would build from the fault vectors' size, is data as well.
    """
    data = _data_side(interp_input, kernel_options, axis)
    dtype = BackendTensor.dtype_obj

    def typed(x):
        return x if x.dtype == dtype else x.type(dtype)

    def i_side(x):
        return symbols(typed(x)[:, None, :])

    def upgraded(struct):  # cached selector structs, already shaped (i: (n, 1, d), j: (1, 1, d))
        return _struct(type(struct), **{name: symbols(typed(value)) for name, value in vars(struct).items()})

    grid_1 = symbols(typed(data.grid_1)[None, :, :])
    ki = KernelInput(
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
    if faults is not None and axis is None:
        # As create_scalar_kernel builds it: x the fault vectors' rows, y the one row of the symbolic faults.
        rows, n_faults = data.fault_vector_ref.shape
        ki.fault_drift_selector = upgraded(DriftMatrixSelector(x_size=rows, y_size=1, n_drift_eq=n_faults,
                                                               drift_start_post_x=rows - n_faults, drift_start_post_y=1))
    return ki


def _points_drift(i_side, grid, d1, d2a, d2b):
    return _struct(PointsDrift, dipsPoints_ui_ai=i_side(d1), dipsPoints_ui_aj=grid,
                   dipsPoints_ui_bi1=i_side(d2a), dipsPoints_ui_bj1=grid,
                   dipsPoints_ui_bi2=i_side(d2b), dipsPoints_ui_bj2=grid)


def _struct(cls, **fields):
    struct = cls.__new__(cls)
    struct.__dict__.update(fields)
    return struct
