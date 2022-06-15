from typing import Tuple

import numpy as np

from ...core import data
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

    if Buffer.weights is None:
        weights = _solve_interpolation(solver_input)
        Buffer.weights = weights
    else:
        weights = Buffer.weights

    # Within octree level
    # +++++++++++++++++++
    exported_fields = _evaluate_sys_eq(xyz_lvl0, solver_input, weights)
    exported_fields.n_points_per_surface = data_shape.reference_sp_position
    exported_fields.n_surface_points = interpolation_input.surface_points.n_points

    Buffer.clean()
    return weights, exported_fields


def _solve_interpolation(interp_input: SolverInput):
    A_matrix = kernel_constructor.yield_covariance(interp_input)
    b_vector = kernel_constructor.yield_b_vector(interp_input.ori_internal, A_matrix.shape[0])
    # TODO: Smooth should be taken from options
    weights = solver_interface.kernel_reduction(A_matrix, b_vector, smooth=0.01)
    return weights


def _input_preprocess(data_shape: TensorsStructure, interpolation_input: InterpolationInput) -> \
        Tuple[np.ndarray, data.OrientationsInternals, data.SurfacePointsInternals, data.FaultsInternals]:
    grid = interpolation_input.grid
    surface_points = interpolation_input.surface_points
    orientations = interpolation_input.orientations
    faults = interpolation_input.fault_values
    
    sp_internal = data_preprocess_interface.prepare_surface_points(surface_points, data_shape)
    ori_internal = data_preprocess_interface.prepare_orientations(orientations)
    grid_internal = data_preprocess_interface.prepare_grid(grid.values, surface_points)
    faults_internal = data_preprocess_interface.prepare_faults(faults, data_shape)
    return grid_internal, ori_internal, sp_internal, faults_internal


def _evaluate_sys_eq(xyz: np.ndarray, interp_input: SolverInput, weights: np.ndarray) -> ExportedFields:
    options = interp_input.options

    eval_kernel = kernel_constructor.yield_evaluation_kernel(xyz, interp_input)
    eval_gx_kernel = kernel_constructor.yield_evaluation_grad_kernel(xyz, interp_input, axis=0)
    eval_gy_kernel = kernel_constructor.yield_evaluation_grad_kernel(xyz, interp_input, axis=1)

    scalar_field = weights @ eval_kernel
    gx_field = weights @ eval_gx_kernel
    gy_field = weights @ eval_gy_kernel

    if options.number_dimensions == 3:
        eval_gz_kernel = kernel_constructor.yield_evaluation_grad_kernel(xyz, interp_input, axis=2)
        gz_field = weights @ eval_gz_kernel
    elif options.number_dimensions == 2:
        gz_field = None
    else:
        raise ValueError("Number of dimensions have to be 2 or 3")

    return ExportedFields(scalar_field, gx_field, gy_field, gz_field)


def _get_scalar_field_at_surface_points(Z_x: np.ndarray, number_of_points_per_surface: np.ndarray, n_surface_points: int):
    npf = number_of_points_per_surface
    scalar_field_at_surface_points_values = Z_x[-n_surface_points:][npf]

    return scalar_field_at_surface_points_values


def _set_scalar_field_at_surface_points(exported_fields: ExportedFields, number_of_points_per_surface: np.ndarray,
                                        n_surface_points: int):
    exported_fields.n_points_per_surface = number_of_points_per_surface
    exported_fields.n_surface_points = n_surface_points
