from typing import Tuple

from numpy import ndarray

from ...core import data
from ...core.data import exported_structs, SurfacePointsInternals
from ...core.data.exported_structs import InterpOutput, OctreeLevel
from ...core.data.grid import Grid
from ...core.data.internal_structs import SolverInput
from ...core.data.interpolation_input import InterpolationInput
from ...modules.activator import activator_interface
from ...modules.data_preprocess import data_preprocess_interface
from ...modules.kernel_constructor import kernel_constructor_interface as kernel_constructor
from ...modules.solver import solver_interface
from ...modules.octrees_topology import octrees_topology_interface as octrees

import numpy as np


class Buffer:
    weights = None

    @classmethod
    def clean(cls):
        cls.weights = None


def interpolate_single_scalar(interpolation_input: InterpolationInput,
                              options: data.InterpolationOptions,
                              data_shape: data.TensorsStructure,
                              clean_buffer = True
                              ):
    grid = interpolation_input.grid
    unit_values = interpolation_input.unit_values

    surface_points = interpolation_input.surface_points
    orientations = interpolation_input.orientations

        # Within series
    xyz_lvl0, ori_internal, sp_internal = input_preprocess(data_shape, grid, orientations, surface_points)

    interp_input = SolverInput(sp_internal, ori_internal, options)

    if Buffer.weights is None:
        weights = solve_interpolation(interp_input)
        Buffer.weights = weights
    else:
        weights = Buffer.weights

    # Within octree level
    # +++++++++++++++++++
    exported_fields = _evaluate_sys_eq(xyz_lvl0, interp_input, weights)

    scalar_field_at_sp = _get_scalar_field_at_surface_points(
        exported_fields.scalar_field, data_shape.nspv, surface_points.n_points)

    # -----------------
    # Export and Masking operations can happen even in parallel
    # TODO: [~X] Export block
    values_block = activator_interface.activate_formation_block(
        exported_fields.scalar_field, scalar_field_at_sp, unit_values, sigmoid_slope=50000)

    # Init InterpOutput Class empty
    output = InterpOutput()
    output.grid = grid
    output.weights = weights
    output.exported_fields = exported_fields
    output.scalar_field_at_sp = scalar_field_at_sp
    output.values_block = values_block

    # -----------------
    # TODO: [ ] Octree - Topology

    # octree_lvl0 = OctreeLevel(grid.values, output.ids_block_regular_grid, output.exported_fields_regular_grid,
    #                           is_root=True)
    #
    # octree_lvl1 = octrees.compute_octree_root(octree_lvl0, grid.regular_grid_values, grid.dxdydz, compute_topology=True, )
    #
    # n_levels = 3 # TODO: Move to options
    # octree_list = [octree_lvl0, octree_lvl1]
    # for i in range(2, n_levels):
    #     next_octree = compute_octree_level_n(octree_list[-1], interp_input, output, unit_values, grid.dxdydz, i)
    #     octree_list.append(next_octree)
    #
    # output.octrees = octree_list
    # ------------------

    # TODO: [ ] Masking OPs. This is for series, i.e. which voxels are active. During development until we
    # TODO: multiple series we can assume all true so final_block = values_block
    # mask_matrix = mask_matrix(exported_fields.scalar_field, scalar_at_surface_points, some_sort_of_array_with_erode_onlap)
    output.final_block = output.values_block.copy()  # TODO (dev hack May 2021): this should be values_block * mask_matrix

    if clean_buffer: Buffer.clean()

    return output

def interpolate_on_octree(octree:OctreeLevel, interpolation_input: InterpolationInput,
                          options: data.InterpolationOptions, data_shape: data.TensorsStructure)-> OctreeLevel:
    def _generate_faces(xyz_coord, dxdydz, level=1):
        x_coord, y_coord, z_coord = xyz_coord[:, 0], xyz_coord[:, 1], xyz_coord[:, 2]
        dx, dy, dz = dxdydz

        x = np.array([[x_coord - dx / 2, x_coord + dx / 2],
                      [x_coord, x_coord],
                      [x_coord, x_coord]]).ravel()
        y = np.array([[y_coord, y_coord],
                      [y_coord - dy / 2, y_coord + dy / 2],
                      [y_coord, y_coord]]).ravel()
        z = np.array([[z_coord, z_coord],
                      [z_coord, z_coord],
                      [z_coord - dz / 2, z_coord + dz / 2]]).ravel()

        new_xyz = np.stack((x, y, z)).T
        return new_xyz

    def _generate_corners(xyz_coord, dxdydz, level=1):
        x_coord, y_coord, z_coord = xyz_coord[:, 0], xyz_coord[:, 1], xyz_coord[:, 2]
        dx, dy, dz = dxdydz

        def stack_left_right(a_edg, d_a):
            return np.stack((a_edg - d_a / level / 2, a_edg + d_a / level / 2), axis=1)

        x_ = np.repeat(stack_left_right(x_coord, dx), 4, axis=1)
        x = x_.ravel()
        y_ = np.tile(np.repeat(stack_left_right(y_coord, dy), 2, axis=1), (1, 2))
        y = y_.ravel()
        z_ = np.tile(stack_left_right(z_coord, dz), (1, 4))
        z = z_.ravel()

        new_xyz = np.stack((x, y, z)).T
        return new_xyz

    def _generate_corners_branch(xyz_coord, dxdydz, level=1):
        x_coord, y_coord, z_coord = xyz_coord[:, 0], xyz_coord[:, 1], xyz_coord[:, 2]
        dx, dy, dz = dxdydz

        def stack_left_right(a_edg, d_a):
            return np.stack((a_edg - d_a / level / 2, a_edg + d_a / level / 2), axis=1)

        def stack_left_right_i(a_edg, d_a): # TODO: it does not seem to do anything
            return np.stack((a_edg + d_a / level / 2, a_edg - d_a / level / 2), axis=1)

        x_ = np.repeat(stack_left_right(x_coord, dx), 4, axis=1)
        x = x_.ravel()
        y_ = np.tile(np.repeat(stack_left_right(y_coord, dy), 2, axis=1), (1, 2))
        y = y_.ravel()
        z_ = np.tile(stack_left_right(z_coord, dz), (1, 4))
        z = z_.ravel()

        new_xyz = np.stack((x, y, z)).T
        return new_xyz

    grid_0_centers = interpolation_input.grid

    # interpolate level 0 - center
    output_0_centers = interpolate_single_scalar(interpolation_input, options, data_shape, clean_buffer=False)
    # Interpolate level 0 - faces
    if octree.is_root:
        grid_0_corners = Grid(_generate_corners(grid_0_centers.values, grid_0_centers.dxdydz))
    else:
        grid_0_corners = Grid(_generate_corners_branch(grid_0_centers.values, grid_0_centers.dxdydz))
    interpolation_input.grid = grid_0_corners
    output_0_corners = interpolate_single_scalar(interpolation_input, options, data_shape, clean_buffer=False)
    # Create octree level 0

    octree.set_interpolation(grid_0_centers, grid_0_corners, output_0_centers, output_0_corners)
    return octree


def interpolate_on_octree_faces(octree:OctreeLevel, interpolation_input: InterpolationInput,
                          options: data.InterpolationOptions, data_shape: data.TensorsStructure)-> OctreeLevel:
    def _generate_faces(xyz_coord, dxdydz, level=1):
        x_coord, y_coord, z_coord = xyz_coord[:, 0], xyz_coord[:, 1], xyz_coord[:, 2]
        dx, dy, dz = dxdydz

        x = np.array([[x_coord - dx / 2, x_coord + dx / 2],
                      [x_coord, x_coord],
                      [x_coord, x_coord]]).ravel()
        y = np.array([[y_coord, y_coord],
                      [y_coord - dy / 2, y_coord + dy / 2],
                      [y_coord, y_coord]]).ravel()
        z = np.array([[z_coord, z_coord],
                      [z_coord, z_coord],
                      [z_coord - dz / 2, z_coord + dz / 2]]).ravel()

        new_xyz = np.stack((x, y, z)).T
        return new_xyz


    grid_0_centers = interpolation_input.grid

    # interpolate level 0 - center
    output_0_centers = interpolate_single_scalar(interpolation_input, options, data_shape, clean_buffer=False)
    # Interpolate level 0 - faces
    grid_0_faces = Grid(_generate_faces(grid_0_centers.values, grid_0_centers.dxdydz))
    interpolation_input.grid = grid_0_faces
    output_0_faces = interpolate_single_scalar(interpolation_input, options, data_shape, clean_buffer=False)
    # Create octree level 0

    octree.set_interpolation(grid_0_centers, grid_0_faces, output_0_centers, output_0_faces)
    return octree


def compute_n_octree_levels(n_levels, interpolation_input, options, data_shape, on_faces = False):
    octree_list = []
    next_octree = OctreeLevel()
    next_octree.is_root = True

    for i in range(0, n_levels):

        if on_faces: # TODO: Probably this is useless
            next_octree = interpolate_on_octree_faces(next_octree, interpolation_input, options, data_shape)
            from gempy_engine.modules.octrees_topology._octree_root import compute_octree_root_on_faces
            grid_1_centers = compute_octree_root_on_faces(next_octree, compute_topology=False, debug=False)
        else:
            next_octree = interpolate_on_octree(next_octree, interpolation_input, options, data_shape)

            grid_1_centers  = octrees.get_next_octree_grid(next_octree, compute_topology=False, debug=False)

        interpolation_input.grid = grid_1_centers
        octree_list.append(next_octree)

        next_octree = OctreeLevel()
    Buffer.clean()
    return octree_list



# def compute_octree_level_n(prev_octree: OctreeLevel, interp_input: SolverInput, output: InterpOutput,
#                            unit_values, i):
#     # Old octree
#     prev_octree.exported_fields = _evaluate_sys_eq(prev_octree.xyz_coords, interp_input, output.weights)
#
#     values_block: ndarray = activator_interface.activate_formation_block(prev_octree.exported_fields.scalar_field,
#                                                                          output.scalar_field_at_sp, unit_values,
#                                                                          sigmoid_slope=50000)
#     prev_octree.id_block = np.rint(values_block[0])
#
#     # TODO: Probably we want to store either both or centers
#     exported_fields_ = _evaluate_sys_eq(prev_octree.grid.custom_grid["centers"], interp_input, output.weights)
#     values_block: ndarray = activator_interface.activate_formation_block(exported_fields_.scalar_field,
#                                                                          output.scalar_field_at_sp, unit_values,
#                                                                          sigmoid_slope=50000)
#     prev_octree.id_block_centers = np.rint(values_block[0])
#
#     # New octree
#     new_octree_level = octrees.compute_octree_branch(prev_octree, level=i)
#     return new_octree_level
#
#
# def compute_octree_last_level(prev_octree: OctreeLevel, interp_input: SolverInput, output: InterpOutput,
#                               unit_values):
#     # Old octree
#     prev_octree.exported_fields = _evaluate_sys_eq(prev_octree.xyz_coords, interp_input, output.weights)
#
#     values_block: ndarray = activator_interface.activate_formation_block(
#         prev_octree.exported_fields.scalar_field, output.scalar_field_at_sp, unit_values, sigmoid_slope=50000)
#     prev_octree.id_block = np.rint(values_block[0])
#
#     # New-Last octree
#     new_xyz = octrees.compute_octree_leaf(prev_octree)
#     new_octree_level = OctreeLevel(new_xyz)
#
#     new_octree_level.exported_fields = _evaluate_sys_eq(new_octree_level.xyz_coords, interp_input, output.weights)
#     return new_octree_level


def solve_interpolation(interp_input: SolverInput):
    A_matrix = kernel_constructor.yield_covariance(interp_input)
    b_vector = kernel_constructor.yield_b_vector(interp_input.ori_internal, A_matrix.shape[0])
    # TODO: Smooth should be taken from options
    weights = solver_interface.kernel_reduction(A_matrix, b_vector, smooth=0.01)
    return weights


def input_preprocess(data_shape, grid, orientations, surface_points) -> \
        Tuple[np.ndarray, data.OrientationsInternals, data.SurfacePointsInternals]:
    sp_internal = data_preprocess_interface.prepare_surface_points(surface_points,
                                                                   data_shape.number_of_points_per_surface)
    ori_internal = data_preprocess_interface.prepare_orientations(orientations)
    grid_internal = data_preprocess_interface.prepare_grid(grid.values, surface_points)
    return grid_internal, ori_internal, sp_internal


def _evaluate_sys_eq(xyz: np.ndarray, interp_input: SolverInput,
                     weights: np.ndarray) -> exported_structs.ExportedFields:
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

    return exported_structs.ExportedFields(scalar_field, gx_field, gy_field, gz_field)


def _get_scalar_field_at_surface_points(Z_x: np.ndarray, number_of_points_per_surface: np.ndarray,
                                        n_surface_points: int):
    npf = number_of_points_per_surface
    scalar_field_at_surface_points_values = Z_x[-n_surface_points:][npf]

    return scalar_field_at_surface_points_values
