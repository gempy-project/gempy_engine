"""Evaluation kernels built once per stack and compute, called with each new grid.

The KeOps formula of a stack's evaluation (scalar field and gradient) is the same at every octree
level and in every extraction query: only the evaluation points, the fault values on them and the
weights change. ``compiled_evaluation`` builds the reduction once with those three as symbolic
KeOps variables, the grid-independent data side as concrete tensors, and keeps it in the stack
cache; later evaluations just call it. Same formula and same inputs as building it per call, so the
results are the same.
"""
from ._keops_reduction import pack_constants
from ..kernel_constructor._internalDistancesMatrices import DistancesBuffer
from ..kernel_constructor._kernels_assembler import create_scalar_kernel, create_grad_kernel
from ..kernel_constructor._structs import (KernelInput, OrientationSurfacePointsCoords, OrientationsDrift, PointsDrift,
                                           FaultDrift)
from ..kernel_constructor._vectors_preparation import _evaluation_data_side
from ..kernel_constructor.execution_mode import KernelExecutionMode
from ...core.backend_tensor import BackendTensor
from ...core.data.exported_fields import ExportedFields
from ...core.data.matrices_sizes import MatricesSizes


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


def symbolic_kernel_input(interp_input, kernel_options, axis, grid, faults) -> KernelInput:
    """Evaluation KernelInput of LazyTensors: the data side concrete (cached), the grid side symbolic.

    Mirrors ``evaluation_vectors_preparations`` + ``upgrade_tensors``; ``grid * grid_1`` is formed inside
    the formula (grid_1 is a one-row parameter).
    """
    from pykeops.torch import LazyTensor
    sp_, ori_ = interp_input.sp_internal, interp_input.ori_internal
    sizes = MatricesSizes(ori_size=ori_.n_orientations_tiled, sp_size=sp_.n_points,
                          uni_drift_size=kernel_options.n_uni_eq, faults_size=interp_input.fault_internal.n_faults,
                          dim=kernel_options.number_dimensions, n_dips=ori_.n_orientations, grid_size=1)
    data = _evaluation_data_side(interp_input, kernel_options, sizes, axis)
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
