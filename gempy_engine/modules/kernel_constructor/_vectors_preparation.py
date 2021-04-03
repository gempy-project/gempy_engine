from gempy_engine.config import BackendTensor, AvailableBackends
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePointsInternals
from gempy_engine.core.data.kernel_classes.orientations import OrientationsInternals

from ._kernel_constructors import assembly_dips_points_tensor, \
    assembly_dips_ug_coords, assembly_dips_points_coords
from ._kernel_selectors import dips_sp_cartesian_selector, grid_cartesian_selector
from ._structs import KernelInput, OrientationSurfacePointsCoords, CartesianSelector, DriftMatrixSelector, \
    OrientationsDrift, PointsDrift

import numpy as np

def cov_vectors_preparation(sp_: SurfacePointsInternals, ori_: OrientationsInternals,
                            options: InterpolationOptions) -> KernelInput:
    cov_size = ori_.n_orientations_tiled + sp_.n_points + options.n_uni_eq

    orientations_sp_matrices = _assembly_dips_points_tensors(options, ori_, sp_)
    dips_ref_ui, dips_rest_ui, dips_ug = _assembly_drift_tensors(options, ori_, sp_)

    # Selectors :
    cartesian_selector = _assembly_cartesian_selector_tensors(cov_size, options, ori_, sp_)
    drift_selection = DriftMatrixSelector(cov_size, cov_size, options.n_uni_eq)

    kernel_input_args = [orientations_sp_matrices, cartesian_selector, dips_ug, dips_ref_ui, dips_rest_ui,
                         drift_selection]

    return KernelInput(*kernel_input_args)


def evaluation_vectors_preparations(grid: np.array, sp_: SurfacePointsInternals, ori_: OrientationsInternals,
                                    options: InterpolationOptions, axis=None) -> KernelInput:
    cov_size = ori_.n_orientations_tiled + sp_.n_points + options.n_uni_eq

    orientations_sp_matrices = _assembly_dips_points_grid_tensors(grid, options, ori_, sp_)

    # Universal
    dips_ref_ui, dips_rest_ui, dips_ug = _assembly_drift_grid_tensors(grid, options, ori_, sp_)

    # Selectors :
    # Cartesian selector
    cartesian_selector = _assembly_cartesian_selector_grid(cov_size, grid, options, ori_, sp_, axis)

    # Drift selector
    drift_selection = DriftMatrixSelector(cov_size, grid.shape[0], options.n_uni_eq)

    kernel_input_args = [orientations_sp_matrices, cartesian_selector, dips_ug, dips_ref_ui, dips_rest_ui,
                         drift_selection]

    return KernelInput(*kernel_input_args)


def _assembly_dips_points_tensors(options, ori_, sp_):
    dips_ref_coord = assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.ref_surface_points, options)
    dips_rest_coord = assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.rest_surface_points, options)
    # When we create que core covariance this are the repeated since the distance are with themselves
    orientations_sp_matrices = OrientationSurfacePointsCoords(dips_ref_coord, dips_ref_coord, dips_rest_coord,
                                                              dips_rest_coord)

    return orientations_sp_matrices


def _assembly_dips_points_grid_tensors(grid, options, ori_, sp_):
    dips_ref_coord = assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.ref_surface_points, options)
    dips_rest_coord = assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.rest_surface_points, options)
    # When we create que core covariance this are the repeated since the distance are with themselves
    orientations_sp_matrices = OrientationSurfacePointsCoords(dips_ref_coord, grid, dips_rest_coord, grid)

    return orientations_sp_matrices


def _assembly_cartesian_selector_tensors(cov_size, options, ori_, sp_):
    sel_hu_input, sel_hv_input, sel_hu_points_input = dips_sp_cartesian_selector(cov_size, options.number_dimensions,
                                                                                 ori_.n_orientations, sp_.n_points)
    cartesian_selector = CartesianSelector(
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

    cartesian_selector = CartesianSelector(
        x_sel_hu=sel_hu_input, y_sel_hu=sel_hu_grid,
        x_sel_hv=sel_hv_input, y_sel_hv=sel_hv_grid,
        x_sel_h_ref=a * sel_hu_points_input, y_sel_h_ref=sel_hu_points_grid,
        x_sel_h_rest=sel_hu_points_input, y_sel_h_rest=sel_hu_points_grid)
    return cartesian_selector


def _assembly_drift_tensors(options, ori_, sp_):
    dips_ug_d1, dips_ug_d2 = assembly_dips_ug_coords(ori_, sp_.n_points, options)
    dips_ug = OrientationsDrift(dips_ug_d1, dips_ug_d1, dips_ug_d2, dips_ug_d1)

    dips_ref_d1, dips_ref_d2a, dips_ref_d2b = assembly_dips_points_coords(
        sp_.ref_surface_points, ori_.n_orientations_tiled, options)
    dips_rest_d1, dips_rest_d2a, dips_rest_d2b = assembly_dips_points_coords(
        sp_.rest_surface_points, ori_.n_orientations_tiled, options)

    dips_ref_ui = PointsDrift(dips_ref_d1, dips_ref_d1, dips_ref_d2a, dips_ref_d2a, dips_ref_d2b, dips_ref_d2b)
    dips_rest_ui = PointsDrift(dips_rest_d1, dips_rest_d1, dips_rest_d2a, dips_rest_d2a, dips_rest_d2b,
                               dips_rest_d2b)
    return dips_ref_ui, dips_rest_ui, dips_ug


def _assembly_drift_grid_tensors(grid, options, ori_, sp_):
    dips_ug_d1, dips_ug_d2 = assembly_dips_ug_coords(ori_, sp_.n_points, options)
    dips_ug = OrientationsDrift(dips_ug_d1, grid, dips_ug_d2, grid)
    dips_ref_d1, dips_ref_d2a, dips_ref_d2b = assembly_dips_points_coords(sp_.ref_surface_points,
                                                                          ori_.n_orientations_tiled, options)
    dips_rest_d1, dips_rest_d2a, dips_rest_d2b = assembly_dips_points_coords(sp_.rest_surface_points,
                                                                             ori_.n_orientations_tiled, options)
    dips_ref_ui = PointsDrift(dips_ref_d1, grid, dips_ref_d2a, dips_ref_d2a, dips_ref_d2b, dips_ref_d2b)
    dips_rest_ui = PointsDrift(dips_rest_d1, grid, dips_rest_d2a, grid, dips_rest_d2b, grid)
    return dips_ref_ui, dips_rest_ui, dips_ug


def _new_kernel_input(kernel_input_args) -> KernelInput:

    ki = KernelInput(*kernel_input_args)
    return ki
