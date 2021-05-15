from numpy import ndarray

from ...core import data
from ...core.data import exported_structs, SurfacePointsInternals
from ...modules.activator import activator_interface
from ...modules.data_preprocess import data_preprocess_interface
from ...modules.kernel_constructor import kernel_constructor_interface
from ...modules.solver import solver_interface
from ...modules.octrees import octrees_interface

import numpy as np



def interpolate_single_scalar(surface_points: data.SurfacePoints,
                              orientations: data.Orientations,
                              grid: np.ndarray,
                              unit_values: np.ndarray,
                              options: data.InterpolationOptions,
                              data_shape: data.TensorsStructure):
    # Within series
    sp_internal: SurfacePointsInternals = data_preprocess_interface.prepare_surface_points(surface_points,
                                                                   data_shape.number_of_points_per_surface)
    ori_internal = data_preprocess_interface.prepare_orientations(orientations)
    grid_internal = data_preprocess_interface.prepare_grid(grid, surface_points)

    A_matrix = kernel_constructor_interface.yield_covariance(sp_internal, ori_internal, options)
    b_vector = kernel_constructor_interface.yield_b_vector(ori_internal, A_matrix.shape[0])

    weights = solver_interface.kernel_reduction(A_matrix, b_vector, smooth=0.01)  # TODO: Smooth should be taken from options

    # Within octree level
    # +++++++++++++++++++
    exported_fields = _evaluate_sys_eq(grid_internal, options, ori_internal, sp_internal, weights)

    scalar_at_surface_points = _get_scalar_field_at_surface_points(
        exported_fields.scalar_field, data_shape.nspv, surface_points.n_points)

    # -----------------
    # Export and Masking operations can happen even in parallel
    # TODO: [~X] Export block
    values_block: ndarray = activator_interface.activate_formation_block(exported_fields.scalar_field, scalar_at_surface_points,
                                                                         unit_values, sigmoid_slope=50000)
    #-----------------
    # TODO: [ ] Topology

    new_grid = octrees_interface.create_oct_level_dense(values_block, grid)

    # TODO: [ ] Octree: Define new grid

    # ------------------

    # TODO: [ ] Masking OPs. This is for series, i.e. which voxels are active. During development until we
    # TODO: multiple series we can assume all true so final_block = values_block
    # mask_matrix = mask_matrix(exported_fields.scalar_field, scalar_at_surface_points, some_sort_of_array_with_erode_onlap)
    final_block = values_block.copy() # TODO (dev hack May 2021): this should be values_block * mask_matrix

    output = exported_structs.Output(exported_fields, scalar_at_surface_points, values_block, final_block)
    return output


def _evaluate_sys_eq(grid, options, ori_internal, sp_internal, weights) -> exported_structs.ExportedFields:
    eval_kernel = kernel_constructor_interface.yield_evaluation_kernel(grid, sp_internal, ori_internal, options)
    eval_gx_kernel = kernel_constructor_interface.yield_evaluation_grad_kernel(grid, sp_internal, ori_internal, options,
                                                                               axis=0)
    eval_gy_kernel = kernel_constructor_interface.yield_evaluation_grad_kernel(grid, sp_internal, ori_internal, options,
                                                                               axis=1)

    scalar_field = weights @ eval_kernel
    gx_field = weights @ eval_gx_kernel
    gy_field = weights @ eval_gy_kernel
    if options.number_dimensions == 3:
        eval_gz_kernel = kernel_constructor_interface.yield_evaluation_grad_kernel(grid, sp_internal, ori_internal,
                                                                                   options, axis=2)
        gz_field = weights @ eval_gz_kernel
    elif options.number_dimensions == 2:
        gz_field = None
    else:
        raise ValueError("Number of dimensions have to be 2 or 3")

    return exported_structs.ExportedFields(scalar_field, gx_field, gy_field, gz_field)


def _get_scalar_field_at_surface_points(Z_x: np.ndarray, number_of_points_per_surface: np.ndarray,
                                        n_surface_points: int):
    npf = number_of_points_per_surface
    scalar_field_at_surface_points_values = Z_x[-n_surface_points:][npf]

    return scalar_field_at_surface_points_values
