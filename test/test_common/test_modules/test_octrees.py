import numpy as np
import pytest

from gempy_engine.core.data.exported_structs import OctreeLevel, InterpOutput
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.integrations.interp_single.interp_single_interface import input_preprocess, solve_interpolation, \
    _evaluate_sys_eq, _get_scalar_field_at_surface_points, compute_octree_level_n
from gempy_engine.modules.activator.activator_interface import activate_formation_block
from gempy_engine.modules.octrees_topology.octrees_topology_interface import _create_oct_level_dense, \
    calculate_topology, compute_octree_root
import matplotlib.pyplot as plt



def test_regular_grid_preparation(simple_grid_3d_more_points_grid):
    engine_grid = simple_grid_3d_more_points_grid
    print(engine_grid.regular_grid[45, 2, 4, 2])
    np.testing.assert_almost_equal(engine_grid.regular_grid[45, 2, 4, 2], .295)


def test_octree_and_topo_root(simple_model_values_block_output, simple_grid_3d_more_points_grid):

    grid = simple_grid_3d_more_points_grid
    output = simple_model_values_block_output

    octree_lvl0 = OctreeLevel(grid.values, output.ids_block_regular_grid,
                              output.exported_fields_regular_grid,
                              is_root=True)

    octree_lvl1 = compute_octree_root(octree_lvl0, grid.regular_grid, grid.dxdydz, compute_topology=True)

    np.testing.assert_array_almost_equal(octree_lvl0.count_edges, np.array([413, 32], dtype=int))

    print(f"Edges id: {octree_lvl0.edges_id}")
    print(f"Count edges: {octree_lvl0.count_edges}")


    if True:
        slice = 2
        plt.contourf(
            octree_lvl0.id_block[:, slice, :].T, N=40, cmap="viridis",
            extent=(0.25, .75, 0.25, .75)
        )

        new_xyz = octree_lvl1.xyz_coords
        plt.scatter(new_xyz[:, 0], new_xyz[:, 2], c="w", s= .5)

        plt.colorbar()
        plt.show()


def test_octree_multiple_levels(simple_model, simple_grid_3d_more_points_grid):
    surface_points = simple_model[0]
    orientations = simple_model[1]
    options = simple_model[2]
    data_shape = simple_model[3]
    grid = simple_grid_3d_more_points_grid

    ids = np.array([1, 2])
    unit_values = ids

    grid_internal, ori_internal, sp_internal = input_preprocess(data_shape, grid, orientations,
                                                                surface_points)
    interp_input = SolverInput(sp_internal, ori_internal, options)

    weights = solve_interpolation(interp_input)

    exported_fields = _evaluate_sys_eq(grid_internal, interp_input, weights)

    scalar_at_surface_points = _get_scalar_field_at_surface_points(
        exported_fields.scalar_field, data_shape.nspv, surface_points.n_points)

    # -----------------
    # Export and Masking operations can happen even in parallel
    # TODO: [~X] Export block
    values_block: np.ndarray = activate_formation_block(exported_fields.scalar_field, scalar_at_surface_points,
                                                         ids, sigmoid_slope=50000)

    output = InterpOutput()
    output.grid = grid
    output.exported_fields = exported_fields
    output.weights = weights
    output.scalar_field_at_sp = scalar_at_surface_points
    output.values_block = values_block

    grid = simple_grid_3d_more_points_grid

    octree_lvl0 = OctreeLevel(grid.values, output.ids_block_regular_grid,
                              output.exported_fields_regular_grid,
                              is_root=True)

    octree_lvl1 = compute_octree_root(octree_lvl0, grid.regular_grid, grid.dxdydz, compute_topology=True)

    n_levels = 3 # TODO: Move to options
    octree_list = [octree_lvl0, octree_lvl1]
    for i in range(2, n_levels):
        next_octree = compute_octree_level_n(octree_list[-1], interp_input, output, unit_values, grid.dxdydz, i)
        octree_list.append(next_octree)

    print(octree_list)


    if True:
        slice = 2
        plt.contourf(
            octree_lvl0.id_block[:, slice, :].T, N=40, cmap="viridis",
            extent=(0.25, .75, 0.25, .75)
        )

        new_xyz = octree_lvl1.xyz_coords
        plt.scatter(new_xyz[:, 0], new_xyz[:, 2], c="w", s= .5)

        plt.scatter(octree_list[-1].xyz_coords[:, 0], octree_list[-1].xyz_coords[:, 2], c="r", s=.3)

        plt.colorbar()
        plt.show()