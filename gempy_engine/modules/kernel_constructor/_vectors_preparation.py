from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from ...core.backend_tensor import BackendTensor
from ...core.compute_scope import checking, assert_same, stack_cache_of
from ._kernel_constructors import assembly_dips_points_tensor, assembly_dips_ug_coords, assembly_dips_points_coords
from ._kernel_selectors import dips_sp_cartesian_selector, grid_cartesian_selector
from ._structs import OrientationSurfacePointsCoords, FaultDrift, PointsDrift, DriftMatrixSelector, KernelInput, CartesianSelector, OrientationsDrift, SurfacePointNuggets
from ...core.data.kernel_classes.faults import FaultsData
from ...core.data.internal_structs import SolverInput
from ...core.data.kernel_classes.orientations import OrientationsInternals
from ...core.data.kernel_classes.surface_points import SurfacePointsInternals
from ...core.data.options import KernelOptions
from ...core.data.matrices_sizes import MatricesSizes


def cov_vectors_preparation(interp_input: SolverInput, kernel_options: KernelOptions) -> KernelInput:
    sp_: SurfacePointsInternals = interp_input.sp_internal
    ori_: OrientationsInternals = interp_input.ori_internal
    faults_val: FaultsData = interp_input.fault_internal
    options: KernelOptions = kernel_options

    matrices_sizes = MatricesSizes(
        ori_size=ori_.n_orientations_tiled,
        sp_size=sp_.n_points,
        uni_drift_size=options.n_uni_eq,
        faults_size=faults_val.n_faults,
        dim=options.number_dimensions,
        n_dips=ori_.n_orientations
    )
    orientations_sp_matrices: OrientationSurfacePointsCoords = _assembly_dips_points_tensors(matrices_sizes, ori_, sp_)
    dips_ref_ui, dips_rest_ui, dips_ug = _assembly_drift_tensors(options, ori_, sp_, matrices_sizes)

    # Selectors :
    cartesian_selector = _assembly_cartesian_selector_tensors(matrices_sizes)
    drift_selection = DriftMatrixSelector(
        x_size=matrices_sizes.cov_size,
        y_size=matrices_sizes.cov_size,
        drift_start_post_x=matrices_sizes.ori_size + matrices_sizes.sp_size,
        drift_start_post_y=matrices_sizes.ori_size + matrices_sizes.sp_size,
        n_drift_eq=matrices_sizes.uni_drift_size)

    if matrices_sizes.faults_size > 0:
        fault_vector_ref, fault_vector_rest = _assembly_fault_tensors(options, faults_val, matrices_sizes.ori_size)
    else:
        fault_vector_ref, fault_vector_rest = None, None

    return KernelInput(
        ori_sp_matrices=orientations_sp_matrices,
        cartesian_selector=cartesian_selector,
        nugget_scalar=SurfacePointNuggets(
            rest=interp_input.sp_internal.nugget_effect_rest,
            reference=interp_input.sp_internal.nugget_effect_ref,
            surface_ids=interp_input.sp_internal.surface_ids,
        ),
        nugget_grad=interp_input.ori_internal.nugget_effect_grad,
        # Drift
        ori_drift=dips_ug,
        ref_drift=dips_ref_ui,
        rest_drift=dips_rest_ui,
        drift_matrix_selector=drift_selection,
        # Faults
        ref_fault=fault_vector_ref,
        rest_fault=fault_vector_rest
    )


@dataclass(frozen=True)
class EvaluationDataSide:
    """Grid-independent (x) side of an evaluation ``KernelInput`` for one gradient axis."""
    dips_ref_coord: object
    dips_rest_coord: object
    dips_ug_d1: object
    dips_ug_d2a: object
    dips_ug_d2b_sel: object
    second_degree_selector: object
    dips_ref_d1: object
    dips_ref_d2a: object
    dips_ref_d2b: object
    dips_rest_d1: object
    dips_rest_d2a: object
    dips_rest_d2b: object
    sel_hu_input: object
    sel_hv_input: object
    sel_hu_points_input: object
    fault_vector_ref: object


def evaluation_vectors_preparations(interp_input: SolverInput, kernel_options: KernelOptions,
                                    axis: Optional[int] = None, slice_array=None) -> KernelInput:
    sp_: SurfacePointsInternals = interp_input.sp_internal
    ori_: OrientationsInternals = interp_input.ori_internal

    # if is none just get the whole array
    if slice_array is not None:
        grid: np.ndarray = interp_input.xyz_to_interpolate[slice_array]
    else:
        grid: np.ndarray = interp_input.xyz_to_interpolate

    faults_vals: FaultsData = interp_input.fault_internal
    options: KernelOptions = kernel_options

    matrices_sizes = MatricesSizes(
        ori_size=ori_.n_orientations_tiled,
        sp_size=sp_.n_points,
        uni_drift_size=options.n_uni_eq,
        faults_size=faults_vals.n_faults,
        dim=options.number_dimensions,
        n_dips=ori_.n_orientations,
        grid_size=grid.shape[0]
    )
    data = _evaluation_data_side(interp_input, options, matrices_sizes, axis)

    orientations_sp_matrices = OrientationSurfacePointsCoords(data.dips_ref_coord, grid, data.dips_rest_coord, grid)  # When we create que core covariance this are the repeated since the distance are with themselves

    # Universal
    grid_1 = BackendTensor.t.zeros_like(grid)
    grid_1[:, axis] = 1
    dips_ug = OrientationsDrift(
        x_degree_1=data.dips_ug_d1, y_degree_1=grid_1,
        x_degree_2=data.dips_ug_d2a, y_degree_2=grid * grid_1,
        x_degree_2b=data.dips_ug_d2b_sel, y_degree_2b=grid,
        selector_degree_2=data.second_degree_selector)
    dips_ref_ui = PointsDrift(data.dips_ref_d1, grid, data.dips_ref_d2a, grid, data.dips_ref_d2b, grid)
    dips_rest_ui = PointsDrift(data.dips_rest_d1, grid, data.dips_rest_d2a, grid, data.dips_rest_d2b, grid)

    # Faults
    fault_drift: Optional[FaultDrift]
    if matrices_sizes.faults_size > 0:
        if slice_array is not None:
            faults_val_on_grid: np.ndarray = faults_vals.fault_values_everywhere[:, slice_array]
        else:
            faults_val_on_grid = faults_vals.fault_values_everywhere
        fault_drift = FaultDrift(
            x_degree_1=data.fault_vector_ref,
            y_degree_1=BackendTensor.t.ascontiguousarray(faults_val_on_grid.T)
        )
    else:
        fault_drift = None

    # Selectors :
    sel_hu_grid, sel_hv_grid, sel_hu_points_grid = grid_cartesian_selector(matrices_sizes, axis)
    cartesian_selector = CartesianSelector(
        x_sel_hu=data.sel_hu_input, y_sel_hu=sel_hu_grid,
        x_sel_hv=data.sel_hv_input, y_sel_hv=sel_hv_grid,
        x_sel_h_ref=data.sel_hu_points_input, y_sel_h_ref=sel_hu_points_grid,
        x_sel_h_rest=data.sel_hu_points_input, y_sel_h_rest=sel_hu_points_grid)
    drift_selection = DriftMatrixSelector(
        x_size=matrices_sizes.cov_size,
        y_size=matrices_sizes.grid_size,
        drift_start_post_x=matrices_sizes.ori_size + matrices_sizes.sp_size,
        drift_start_post_y=matrices_sizes.grid_size,
        n_drift_eq=matrices_sizes.uni_drift_size)

    return KernelInput(
        ori_sp_matrices=orientations_sp_matrices,
        cartesian_selector=cartesian_selector,
        nugget_scalar=None,
        nugget_grad=None,
        # drift
        ori_drift=dips_ug,
        ref_drift=dips_ref_ui,
        rest_drift=dips_rest_ui,
        drift_matrix_selector=drift_selection,
        # faults
        ref_fault=fault_drift,
        rest_fault=None
    )


def _evaluation_data_side(interp_input, options: KernelOptions, matrices_sizes: MatricesSizes, axis) -> EvaluationDataSide:
    """Built once per stack and axis within a compute (see ``compute_scope``); the grid never enters it."""
    cache = stack_cache_of(interp_input)
    key = (axis, options.uni_degree, options.number_dimensions, options.n_uni_eq)
    if cache is not None and not checking() and key in cache.data_side:
        return cache.data_side[key]

    sp_: SurfacePointsInternals = interp_input.sp_internal
    ori_: OrientationsInternals = interp_input.ori_internal
    dips_ref_coord = assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.ref_surface_points, matrices_sizes)
    dips_rest_coord = assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.rest_surface_points, matrices_sizes)

    dips_ug_d1, dips_ug_d2a, dips_ug_d2b, second_degree_selector = assembly_dips_ug_coords(ori_, options, matrices_sizes)
    sel = BackendTensor.t.ones(options.number_dimensions)
    sel[axis] = 0
    dips_ref_d1, dips_ref_d2a, dips_ref_d2b = assembly_dips_points_coords(sp_.ref_surface_points, matrices_sizes, options)
    dips_rest_d1, dips_rest_d2a, dips_rest_d2b = assembly_dips_points_coords(sp_.rest_surface_points, matrices_sizes, options)

    if matrices_sizes.faults_size > 0:
        fault_vector_ref, _ = _assembly_fault_internals(interp_input.fault_internal, options, matrices_sizes.ori_size)
    else:
        fault_vector_ref = None

    sel_hu_input, sel_hv_input, sel_hu_points_input = dips_sp_cartesian_selector(matrices_sizes, axis)
    data = EvaluationDataSide(
        dips_ref_coord=dips_ref_coord, dips_rest_coord=dips_rest_coord,
        dips_ug_d1=dips_ug_d1, dips_ug_d2a=dips_ug_d2a, dips_ug_d2b_sel=dips_ug_d2b * sel,
        second_degree_selector=second_degree_selector,
        dips_ref_d1=dips_ref_d1, dips_ref_d2a=dips_ref_d2a, dips_ref_d2b=dips_ref_d2b,
        dips_rest_d1=dips_rest_d1, dips_rest_d2a=dips_rest_d2a, dips_rest_d2b=dips_rest_d2b,
        sel_hu_input=sel_hu_input, sel_hv_input=sel_hv_input, sel_hu_points_input=sel_hu_points_input,
        fault_vector_ref=fault_vector_ref,
    )
    if cache is None:
        return data
    if key in cache.data_side:  # checking()
        cached = cache.data_side[key]
        for name in EvaluationDataSide.__dataclass_fields__:
            assert_same(getattr(cached, name), getattr(data, name), f"evaluation data side {name}")
        return cached
    cache.data_side[key] = data
    return data


def _assembly_dips_points_tensors(matrices_size: MatricesSizes, ori_, sp_) -> OrientationSurfacePointsCoords:
    dips_ref_coord = assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.ref_surface_points, matrices_size)
    dips_rest_coord = assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.rest_surface_points, matrices_size)

    orientations_sp_matrices = OrientationSurfacePointsCoords(dips_ref_coord, dips_ref_coord, dips_rest_coord,
                                                              dips_rest_coord)  # When we create que core covariance these are the repeated since the distance are with themselves

    return orientations_sp_matrices


def _assembly_cartesian_selector_tensors(matrices_sizes: MatricesSizes):
    """Used for kernel construction"""
    sel_hu_input, sel_hv_input, sel_hu_points_input = dips_sp_cartesian_selector(matrices_sizes)
    cartesian_selector = CartesianSelector(
        x_sel_hu=sel_hu_input, y_sel_hu=sel_hv_input,
        x_sel_hv=sel_hv_input, y_sel_hv=sel_hu_input,
        x_sel_h_ref=sel_hu_points_input, y_sel_h_ref=sel_hu_points_input,
        x_sel_h_rest=sel_hu_points_input, y_sel_h_rest=sel_hu_points_input
    )
    return cartesian_selector


def _assembly_drift_tensors(options: KernelOptions, ori_: OrientationsInternals, sp_: SurfacePointsInternals,
                            matrices_sizes: MatricesSizes):
    dips_ug_d1, dips_ug_d2a, dips_ug_d2b, second_degree_selector = assembly_dips_ug_coords(ori_, options, matrices_sizes)

    dips_ug = OrientationsDrift(
        dips_ug_d1, dips_ug_d1,
        dips_ug_d2a, dips_ug_d2a,
        dips_ug_d2b, dips_ug_d2b,
        second_degree_selector
    )

    dips_ref_d1, dips_ref_d2a, dips_ref_d2b = assembly_dips_points_coords(sp_.ref_surface_points, matrices_sizes, options)
    dips_rest_d1, dips_rest_d2a, dips_rest_d2b = assembly_dips_points_coords(sp_.rest_surface_points, matrices_sizes, options)

    dips_ref_ui = PointsDrift(dips_ref_d1, dips_ref_d1, dips_ref_d2a, dips_ref_d2a, dips_ref_d2b, dips_ref_d2b)
    dips_rest_ui = PointsDrift(dips_rest_d1, dips_rest_d1, dips_rest_d2a, dips_rest_d2a, dips_rest_d2b, dips_rest_d2b)

    return dips_ref_ui, dips_rest_ui, dips_ug


def _assembly_fault_tensors(options, faults_val: FaultsData, ori_size: int) -> Tuple[FaultDrift, FaultDrift]:
    fault_vector_ref, fault_vector_rest = _assembly_fault_internals(faults_val, options, ori_size)
    return FaultDrift(fault_vector_ref, fault_vector_ref), FaultDrift(fault_vector_rest, fault_vector_rest)


def _assembly_fault_internals(faults_val, options, ori_size):
    def _assembler(matrix_val, ori_size_: int, uni_drift_size: int):  # TODO: This function (probably)needs to be extracted to _kernel_constructors
        n_uni_eq = uni_drift_size  # * Number of equations. This should be how many faults are active
        n_faults = matrix_val.shape[1]  # TODO [ ]: We are going to have to tweak this for multiple faults
        z = BackendTensor.t.zeros((ori_size_, n_faults), dtype=BackendTensor.dtype_obj)
        z2 = BackendTensor.t.zeros((n_uni_eq, n_faults), dtype=BackendTensor.dtype_obj)
        z3 = BackendTensor.t.eye(n_faults, dtype=BackendTensor.dtype_obj)
        # Degree 1
        return BackendTensor.t.vstack((z, matrix_val, z2, z3))

    ref_matrix_val = faults_val.fault_values_ref
    rest_matrix_val = faults_val.fault_values_rest
    ref_matrix_contig = BackendTensor.t.ascontiguousarray(ref_matrix_val.T)
    fault_vector_ref = _assembler(ref_matrix_contig, ori_size, options.n_uni_eq)
    fault_vector_rest = _assembler(rest_matrix_val.T, ori_size, options.n_uni_eq)
    return fault_vector_ref, fault_vector_rest
