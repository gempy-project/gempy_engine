from typing import Tuple

from ._structs import OrientationSurfacePointsCoords, FaultDrift
from ...core.backend_tensor import BackendTensor, AvailableBackends
from ...core.data import FaultsInternals
from ...core.data.internal_structs import SolverInput
from ...core.data.options import InterpolationOptions, KernelOptions
from ...core.data.kernel_classes.surface_points import SurfacePointsInternals
from ...core.data.kernel_classes.orientations import OrientationsInternals

from . import _kernel_constructors
from ._kernel_selectors import dips_sp_cartesian_selector, grid_cartesian_selector
from . import _structs

import numpy as np


def cov_vectors_preparation(interp_input: SolverInput) -> _structs.KernelInput:
    sp_: SurfacePointsInternals = interp_input.sp_internal
    ori_: OrientationsInternals = interp_input.ori_internal
    faults_val: FaultsInternals = interp_input.fault_internal
    options: KernelOptions = interp_input.options

    ori_size = ori_.n_orientations_tiled
    sp_size = sp_.n_points
    drift_size = options.n_uni_eq
    faults_size = faults_val.n_faults

    cov_size = ori_size + sp_size + drift_size + faults_size

    orientations_sp_matrices: OrientationSurfacePointsCoords = _assembly_dips_points_tensors(options, ori_, sp_)
    dips_ref_ui, dips_rest_ui, dips_ug = _assembly_drift_tensors(options, ori_, sp_)

    # Selectors :
    cartesian_selector = _assembly_cartesian_selector_tensors(cov_size, options, ori_, sp_)

    drift_start_position = ori_size + sp_size
    drift_selection = _structs.DriftMatrixSelector(cov_size, cov_size, drift_start_position, drift_size)

    if faults_size > 0:
        fault_vector_ref, fault_vector_rest = _assembly_fault_tensors(options, faults_val, ori_size)
    else:
        fault_vector_ref, fault_vector_rest = None, None
    
    return _structs.KernelInput(
        ori_sp_matrices=orientations_sp_matrices,
        cartesian_selector=cartesian_selector,
        ori_drift=dips_ug,
        ref_drift=dips_ref_ui,
        rest_drift=dips_rest_ui,
        drift_matrix_selector=drift_selection,
        ref_fault=fault_vector_ref,
        rest_fault=fault_vector_rest
    )


def evaluation_vectors_preparations(grid: np.array, interp_input: SolverInput, axis=None) -> _structs.KernelInput:
    sp_: SurfacePointsInternals = interp_input.sp_internal
    ori_: OrientationsInternals = interp_input.ori_internal
    faults_val: FaultsInternals = interp_input.fault_internal
    options: KernelOptions = interp_input.options

    ori_size = ori_.n_orientations_tiled
    sp_size = sp_.n_points
    drift_size = options.n_uni_eq
    faults_size = faults_val.n_faults

    cov_size = ori_size + sp_size + drift_size + faults_size

    orientations_sp_matrices = _assembly_dips_points_grid_tensors(grid, options, ori_, sp_)

    # Universal
    dips_ref_ui, dips_rest_ui, dips_ug = _assembly_drift_grid_tensors(grid, options, ori_, sp_, axis)

    # Faults
    if faults_size > 0:
        fault_vector_ref, fault_vector_rest = _assembly_fault_grid_tensors(grid, options, faults_val, ori_size)
    else:
        fault_vector_ref, fault_vector_rest = None, None

    # Selectors :
    cartesian_selector = _assembly_cartesian_selector_grid(cov_size, grid, options, ori_, sp_, axis)
    drift_start_position = ori_size + sp_size

    drift_selection = _structs.DriftMatrixSelector(cov_size, grid.shape[0], drift_start_position, options.n_uni_eq)

    return _structs.KernelInput(
        ori_sp_matrices=orientations_sp_matrices,
        cartesian_selector=cartesian_selector,
        ori_drift=dips_ug,
        ref_drift=dips_ref_ui,
        rest_drift=dips_rest_ui,
        drift_matrix_selector=drift_selection,
        ref_fault=fault_vector_ref,
        rest_fault=fault_vector_rest
    )


def _assembly_dips_points_tensors(options, ori_, sp_) -> _structs.OrientationSurfacePointsCoords:
    dips_ref_coord = _kernel_constructors.assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.ref_surface_points, options)
    dips_rest_coord = _kernel_constructors.assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.rest_surface_points, options)
    # When we create que core covariance this are the repeated since the distance are with themselves
    orientations_sp_matrices = _structs.OrientationSurfacePointsCoords(dips_ref_coord, dips_ref_coord, dips_rest_coord,
                                                                       dips_rest_coord)

    return orientations_sp_matrices


def _assembly_dips_points_grid_tensors(grid, options, ori_, sp_):
    """Used for kernel construction"""
    dips_ref_coord = _kernel_constructors.assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.ref_surface_points, options)
    dips_rest_coord = _kernel_constructors.assembly_dips_points_tensor(ori_.dip_positions_tiled, sp_.rest_surface_points, options)
    # When we create que core covariance this are the repeated since the distance are with themselves
    orientations_sp_matrices = _structs.OrientationSurfacePointsCoords(dips_ref_coord, grid, dips_rest_coord, grid)

    return orientations_sp_matrices


def _assembly_cartesian_selector_tensors(cov_size, options, ori_, sp_):
    """Used for kernel construction"""
    sel_hu_input, sel_hv_input, sel_hu_points_input = dips_sp_cartesian_selector(cov_size, options.number_dimensions,
                                                                                 ori_.n_orientations, sp_.n_points)
    cartesian_selector = _structs.CartesianSelector(
        x_sel_hu=sel_hu_input, y_sel_hu=sel_hv_input,
        x_sel_hv=sel_hv_input, y_sel_hv=sel_hu_input,
        x_sel_h_ref=sel_hu_points_input, y_sel_h_ref=sel_hu_points_input,
        x_sel_h_rest=sel_hu_points_input, y_sel_h_rest=sel_hu_points_input)
    return cartesian_selector


def _assembly_cartesian_selector_grid(cov_size, grid, options, ori_, sp_, axis=None):
    """Use for evaluation"""
    sel_hu_input, sel_hv_input, sel_hu_points_input = dips_sp_cartesian_selector(cov_size, options.number_dimensions,
                                                                                 ori_.n_orientations, sp_.n_points,
                                                                                 axis)
    sel_hu_grid, sel_hv_grid, sel_hu_points_grid = grid_cartesian_selector(grid.shape[0], options.number_dimensions,
                                                                           axis)

    cartesian_selector = _structs.CartesianSelector(
        x_sel_hu=sel_hu_input, y_sel_hu=sel_hu_grid,
        x_sel_hv=sel_hv_input, y_sel_hv=sel_hv_grid,
        x_sel_h_ref=sel_hu_points_input, y_sel_h_ref=sel_hu_points_grid,
        x_sel_h_rest=sel_hu_points_input, y_sel_h_rest=sel_hu_points_grid)
    return cartesian_selector


def _assembly_drift_tensors(options, ori_, sp_):
    dips_ug_d1, dips_ug_d2a, dips_ug_d2b, second_degree_selector = _kernel_constructors.assembly_dips_ug_coords(
        ori_, sp_.n_points, options)
    dips_ug = _structs.OrientationsDrift(
        dips_ug_d1, dips_ug_d1,
        dips_ug_d2a, dips_ug_d2a,
        dips_ug_d2b, dips_ug_d2b,
        second_degree_selector
    )

    dips_ref_d1, dips_ref_d2a, dips_ref_d2b = _kernel_constructors.assembly_dips_points_coords(
        sp_.ref_surface_points, ori_.n_orientations_tiled, options)
    dips_rest_d1, dips_rest_d2a, dips_rest_d2b = _kernel_constructors.assembly_dips_points_coords(
        sp_.rest_surface_points, ori_.n_orientations_tiled, options)

    dips_ref_ui = _structs.PointsDrift(dips_ref_d1, dips_ref_d1, dips_ref_d2a, dips_ref_d2a, dips_ref_d2b, dips_ref_d2b)
    dips_rest_ui = _structs.PointsDrift(dips_rest_d1, dips_rest_d1, dips_rest_d2a, dips_rest_d2a, dips_rest_d2b, dips_rest_d2b)

    return dips_ref_ui, dips_rest_ui, dips_ug


def _assembly_drift_grid_tensors(grid, options, ori_, sp_, axis):
    # region UG
    dips_ug_d1, dips_ug_d2a, dips_ug_d2b, second_degree_selector = _kernel_constructors.assembly_dips_ug_coords(
        ori_, sp_.n_points, options)

    grid_1 = np.zeros_like(grid)
    grid_1[:, axis] = 1

    sel = np.ones(options.number_dimensions)
    sel[axis] = 0

    dips_ug = _structs.OrientationsDrift(
        dips_ug_d1, grid_1,
        dips_ug_d2a, grid * grid_1,
        dips_ug_d2b * sel, grid,
        second_degree_selector)
    # endregion

    # region UI
    dips_ref_d1, dips_ref_d2a, dips_ref_d2b = _kernel_constructors.assembly_dips_points_coords(
        sp_.ref_surface_points, ori_.n_orientations_tiled, options)

    dips_rest_d1, dips_rest_d2a, dips_rest_d2b = _kernel_constructors.assembly_dips_points_coords(
        sp_.rest_surface_points, ori_.n_orientations_tiled, options)

    dips_ref_ui = _structs.PointsDrift(dips_ref_d1, grid, dips_ref_d2a, grid, dips_ref_d2b, grid)
    dips_rest_ui = _structs.PointsDrift(dips_rest_d1, grid, dips_rest_d2a, grid, dips_rest_d2b, grid)
    # endregion

    return dips_ref_ui, dips_rest_ui, dips_ug


def _assembly_fault_grid_tensors(grid, options, faults_val, ori_size):
    fault_vector_ref, fault_vector_rest = _assembly_fault_internals(faults_val, options, ori_size)
    return FaultDrift(fault_vector_ref, grid), FaultDrift(fault_vector_rest, grid)


def _assembly_fault_tensors(options, faults_val: FaultsInternals, ori_size: int) -> Tuple[FaultDrift, FaultDrift]:
    fault_vector_ref, fault_vector_rest = _assembly_fault_internals(faults_val, options, ori_size)
    return FaultDrift(fault_vector_ref, fault_vector_ref), FaultDrift(fault_vector_rest, fault_vector_rest)


def _assembly_fault_internals(faults_val, options, ori_size):
    def _assembler(matrix_val, ori_size: int, uni_drift_size: int):  # TODO: This function (probably)needs to be extracted to _kernel_constructors 
        n_dim = 1
        n_uni_eq = uni_drift_size  # * Number of equations. This should be how many faults are active
        n_faults = 1  # TODO: We are going to have to tweak this for multiple faults
        z = np.zeros((ori_size, n_dim))
        z2 = np.zeros((n_uni_eq, n_dim))
        z3 = np.ones((n_faults, n_dim))
        # Degree 1
        return np.vstack((z, matrix_val, z2, z3))

    ref_matrix_val = faults_val.fault_values_ref
    rest_matrix_val = faults_val.fault_values_rest
    fault_vector_ref = _assembler(ref_matrix_val.reshape(-1, 1), ori_size, options.n_uni_eq)
    fault_vector_rest = _assembler(rest_matrix_val.reshape(-1, 1), ori_size, options.n_uni_eq)
    return fault_vector_ref, fault_vector_rest
