from typing import Tuple

import numpy as np
from pykeops.numpy import LazyTensor

from ...core import data
from ...core.data import FaultsData, Orientations, SurfacePoints, SurfacePointsInternals, OrientationsInternals
from ...core.data.exported_fields import ExportedFields
from ...core.data.input_data_descriptor import TensorsStructure
from ...core.data.internal_structs import SolverInput
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.options import KernelOptions

from ...modules.data_preprocess import data_preprocess_interface
from ...modules.kernel_constructor import kernel_constructor_interface as kernel_constructor
from ...modules.solver import solver_interface


class Buffer:
    weights = None

    @classmethod
    def clean(cls):
        cls.weights = None


def interpolate_scalar_field(interpolation_input: InterpolationInput, options: KernelOptions,
                             data_shape: data.TensorsStructure) -> Tuple[np.ndarray, ExportedFields]:
    # Within series
    xyz_lvl0, ori_internal, sp_internal, fault_internal = _input_preprocess(data_shape, interpolation_input)
    solver_input = SolverInput(
        sp_internal=sp_internal,
        ori_internal=ori_internal,
        fault_internal=fault_internal,
        options=options)

    # region Solver
    if Buffer.weights is None:
        weights = _solve_interpolation(solver_input)
        Buffer.weights = weights
    else:
        weights = Buffer.weights

    # endregion
    
    
    exported_fields = _evaluate_sys_eq(xyz_lvl0, solver_input, weights)

    # TODO: This should be in the TensorsStructure
    exported_fields.n_points_per_surface = data_shape.reference_sp_position
    exported_fields.slice_feature = interpolation_input.slice_feature
    exported_fields.grid_size = interpolation_input.grid.len_all_grids

    exported_fields.debug = solver_input.debug

    Buffer.clean()
    return weights, exported_fields


def _solve_interpolation(interp_input: SolverInput):
    A_matrix = kernel_constructor.yield_covariance(interp_input)
    b_vector = kernel_constructor.yield_b_vector(interp_input.ori_internal, A_matrix.shape[0])
    # TODO: Smooth should be taken from options
    weights = solver_interface.kernel_reduction(A_matrix, b_vector, smooth=0.01)
    return weights


def _input_preprocess(data_shape: TensorsStructure, interpolation_input: InterpolationInput) -> \
        Tuple[np.ndarray, data.OrientationsInternals, data.SurfacePointsInternals, data.FaultsData]:
    grid = interpolation_input.grid
    surface_points: SurfacePoints = interpolation_input.surface_points
    orientations: Orientations = interpolation_input.orientations

    sp_internal: SurfacePointsInternals = data_preprocess_interface.prepare_surface_points(surface_points, data_shape)
    ori_internal: OrientationsInternals = data_preprocess_interface.prepare_orientations(orientations)

    # * We need to interpolate in ALL the surface points not only the surface points of the stack
    grid_internal = data_preprocess_interface.prepare_grid(grid.values, interpolation_input.all_surface_points)

    fault_values: FaultsData = interpolation_input.fault_values
    faults_on_sp: np.ndarray = fault_values.fault_values_on_sp
    fault_ref, fault_rest = data_preprocess_interface.prepare_faults(faults_on_sp, data_shape)
    fault_values.fault_values_ref, fault_values.fault_values_rest = fault_ref, fault_rest

    return grid_internal, ori_internal, sp_internal, fault_values


def _evaluate_sys_eq(xyz: np.ndarray, interp_input: SolverInput, weights: np.ndarray) -> ExportedFields:
    options = interp_input.options
    
    if xyz.flags['C_CONTIGUOUS'] is False:
        print("xyz is not C_CONTIGUOUS")
        
    eval_kernel = kernel_constructor.yield_evaluation_kernel(xyz, interp_input)
    eval_gx_kernel = kernel_constructor.yield_evaluation_grad_kernel(xyz, interp_input, axis=0)
    eval_gy_kernel = kernel_constructor.yield_evaluation_grad_kernel(xyz, interp_input, axis=1)

    if True:
        # ! Seems not to make any difference but we need this if we want to change the backend
        # ! We need to benchmark GPU vs CPU with more input
        scalar_field = (eval_kernel.T * LazyTensor(np.asfortranarray(weights), axis=1)).sum(axis=1, backend="GPU").reshape(-1)
        gx_field = (eval_gx_kernel.T * LazyTensor(weights, axis=1)).sum(axis=1, backend="GPU").reshape(-1)
        gy_field = (eval_gy_kernel.T * LazyTensor(weights, axis=1)).sum(axis=1, backend="GPU").reshape(-1)

        if options.number_dimensions == 3:
            eval_gz_kernel = kernel_constructor.yield_evaluation_grad_kernel(xyz, interp_input, axis=2)
            gz_field = (eval_gz_kernel.T * LazyTensor(weights, axis=1)).sum(axis=1, backend="GPU").reshape(-1)
        elif options.number_dimensions == 2:
            gz_field = None
        else:
            raise ValueError("Number of dimensions have to be 2 or 3")
    else:
        scalar_field = (eval_kernel.T @ weights).reshape(-1)
        gx_field = (eval_gx_kernel.T @ weights).reshape(-1)
        gy_field = (eval_gy_kernel.T @ weights).reshape(-1)
    
        if options.number_dimensions == 3:
            eval_gz_kernel = kernel_constructor.yield_evaluation_grad_kernel(xyz, interp_input, axis=2)
            gz_field = (eval_gz_kernel.T @ weights).reshape(-1)
        elif options.number_dimensions == 2:
            gz_field = None
        else:
            raise ValueError("Number of dimensions have to be 2 or 3")

    return ExportedFields(scalar_field, gx_field, gy_field, gz_field)
