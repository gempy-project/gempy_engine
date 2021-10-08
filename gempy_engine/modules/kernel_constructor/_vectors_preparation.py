from ...core.backend_tensor import BackendTensor, AvailableBackends
from ...core.data.internal_structs import SolverInput
from ...core.data.options import InterpolationOptions
from ...core.data.kernel_classes.surface_points import SurfacePointsInternals
from ...core.data.kernel_classes.orientations import OrientationsInternals

from . import _kernel_constructors
from ._kernel_selectors import dips_sp_cartesian_selector, grid_cartesian_selector
from . import _structs


import numpy as np


def cov_vectors_preparation(interp_input: SolverInput) -> _structs.KernelInput:
    sp_: SurfacePointsInternals = interp_input.sp_internal
    ori_: OrientationsInternals = interp_input.ori_internal
    options: InterpolationOptions = interp_input.options

    cov_size = ori_.n_orientations_tiled + sp_.n_points + options.n_uni_eq

    orientations_sp_matrices = _assembly_dips_points_tensors(options, ori_, sp_)
    dips_ref_ui, dips_rest_ui, dips_ug = _assembly_drift_tensors(options, ori_, sp_)

    # Selectors :
    cartesian_selector = _assembly_cartesian_selector_tensors(cov_size, options, ori_, sp_)
    drift_selection = _structs.DriftMatrixSelector(cov_size, cov_size, options.n_uni_eq)

    kernel_input_args = [orientations_sp_matrices, cartesian_selector, dips_ug, dips_ref_ui, dips_rest_ui,
                         drift_selection]

    return _structs.KernelInput(*kernel_input_args)


def evaluation_vectors_preparations(grid: np.array, interp_input: SolverInput, axis=None) -> _structs.KernelInput:

    sp_: SurfacePointsInternals = interp_input.sp_internal
    ori_: OrientationsInternals = interp_input.ori_internal
    options: InterpolationOptions = interp_input.options

    cov_size = ori_.n_orientations_tiled + sp_.n_points + options.n_uni_eq

    orientations_sp_matrices = _assembly_dips_points_grid_tensors(grid, options, ori_, sp_)

    # Universal
    dips_ref_ui, dips_rest_ui, dips_ug = _assembly_drift_grid_tensors(grid, options, ori_, sp_)

    # Selectors :
    cartesian_selector = _assembly_cartesian_selector_grid(cov_size, grid, options, ori_, sp_, axis)
    drift_selection = _structs.DriftMatrixSelector(cov_size, grid.shape[0], options.n_uni_eq)

    kernel_input_args = [orientations_sp_matrices, cartesian_selector, dips_ug,
                         dips_ref_ui, dips_rest_ui, drift_selection]

    return _structs.KernelInput(*kernel_input_args)


def _assembly_dips_points_tensors(options, ori_, sp_):
    dips_ref_coord = _kernel_constructors.assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.ref_surface_points, options)
    dips_rest_coord = _kernel_constructors.assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.rest_surface_points, options)
    # When we create que core covariance this are the repeated since the distance are with themselves
    orientations_sp_matrices = _structs.OrientationSurfacePointsCoords(dips_ref_coord, dips_ref_coord, dips_rest_coord,
                                                                       dips_rest_coord)

    return orientations_sp_matrices


def _assembly_dips_points_grid_tensors(grid, options, ori_, sp_):
    dips_ref_coord = _kernel_constructors.assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.ref_surface_points, options)
    dips_rest_coord = _kernel_constructors.assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.rest_surface_points, options)
    # When we create que core covariance this are the repeated since the distance are with themselves
    orientations_sp_matrices = _structs.OrientationSurfacePointsCoords(dips_ref_coord, grid, dips_rest_coord, grid)

    return orientations_sp_matrices


def _assembly_cartesian_selector_tensors(cov_size, options, ori_, sp_):
    sel_hu_input, sel_hv_input, sel_hu_points_input = dips_sp_cartesian_selector(cov_size, options.number_dimensions,
                                                                                 ori_.n_orientations, sp_.n_points)
    cartesian_selector = _structs.CartesianSelector(
        x_sel_hu=sel_hu_input, y_sel_hu=sel_hv_input, x_sel_hv=sel_hv_input,
        y_sel_hv=sel_hu_input, x_sel_h_ref=sel_hu_points_input,
        y_sel_h_ref=sel_hu_points_input, x_sel_h_rest=sel_hu_points_input,
        y_sel_h_rest=sel_hu_points_input)
    return cartesian_selector


def _assembly_cartesian_selector_grid(cov_size, grid, options, ori_, sp_, axis=None):
    sel_hu_input, sel_hv_input, sel_hu_points_input = dips_sp_cartesian_selector(cov_size, options.number_dimensions,
                                                                                 ori_.n_orientations, sp_.n_points,
                                                                                 axis)
    sel_hu_grid, sel_hv_grid, sel_hu_points_grid = grid_cartesian_selector(grid.shape[0], options.number_dimensions,
                                                                           axis)

    # TODO: Check I need this
    a = 1 if axis is None else -1

    cartesian_selector = _structs.CartesianSelector(
        x_sel_hu=sel_hu_input, y_sel_hu=sel_hu_grid,
        x_sel_hv=sel_hv_input, y_sel_hv=sel_hv_grid,
        x_sel_h_ref=a * sel_hu_points_input, y_sel_h_ref=sel_hu_points_grid,
        x_sel_h_rest=sel_hu_points_input, y_sel_h_rest=sel_hu_points_grid)
    return cartesian_selector


def _assembly_drift_tensors(options, ori_, sp_):
    dips_ug_d1, dips_ug_d2a, dips_ug_d2b, second_degree_selector  = _kernel_constructors.assembly_dips_ug_coords(ori_, sp_.n_points, options)
    dips_ug = _structs.OrientationsDrift(dips_ug_d1, dips_ug_d1,
                                         dips_ug_d2a, dips_ug_d2a,
                                         dips_ug_d2b, dips_ug_d2b,
                                         second_degree_selector
                                         )

    dips_ref_d1, dips_ref_d2a, dips_ref_d2b = _kernel_constructors.assembly_dips_points_coords(
        sp_.ref_surface_points, ori_.n_orientations_tiled, options)
    dips_rest_d1, dips_rest_d2a, dips_rest_d2b = _kernel_constructors.assembly_dips_points_coords(
        sp_.rest_surface_points, ori_.n_orientations_tiled, options)

    dips_ref_ui = _structs.PointsDrift(dips_ref_d1, dips_ref_d1, dips_ref_d2a, dips_ref_d2a, dips_ref_d2b, dips_ref_d2b)
    dips_rest_ui = _structs.PointsDrift(dips_rest_d1, dips_rest_d1, dips_rest_d2a, dips_rest_d2a, dips_rest_d2b,
                                       dips_rest_d2b)

    return dips_ref_ui, dips_rest_ui, dips_ug


def _assembly_drift_grid_tensors(grid, options, ori_, sp_):

    # UG
    dips_ug_d1, dips_ug_d2a, dips_ug_d2b, second_degree_selector = _kernel_constructors.assembly_dips_ug_coords(ori_, sp_.n_points, options)
    dips_ug = _structs.OrientationsDrift(dips_ug_d1, grid,
                                         dips_ug_d2a, grid,
                                         dips_ug_d2b, grid,
                                         second_degree_selector)
    # UI
    dips_ref_d1, dips_ref_d2a, dips_ref_d2b = _kernel_constructors.assembly_dips_points_coords(
        sp_.ref_surface_points, ori_.n_orientations_tiled, options)
    dips_rest_d1, dips_rest_d2a, dips_rest_d2b = _kernel_constructors.assembly_dips_points_coords(
        sp_.rest_surface_points, ori_.n_orientations_tiled, options)
    dips_ref_ui = _structs.PointsDrift(dips_ref_d1, grid, dips_ref_d2a, grid, dips_ref_d2b, grid)
    dips_rest_ui = _structs.PointsDrift(dips_rest_d1, grid, dips_rest_d2a, grid, dips_rest_d2b, grid)

    return dips_ref_ui, dips_rest_ui, dips_ug


def _new_kernel_input(kernel_input_args) -> _structs.KernelInput:

    ki = _structs.KernelInput(*kernel_input_args)
    return ki
