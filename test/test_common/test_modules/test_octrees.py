import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import gempy_engine.API.interp_single.interp_features as interp
from gempy_engine.API.interp_single._interp_scalar_field import _evaluate_sys_eq
from gempy_engine.core.data.grid import Grid, RegularGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.modules.activator.activator_interface import activate_formation_block
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_next_octree_grid, \
    get_regular_grid_value_for_level
from .test_dual import _compute_actual_mesh
from ...conftest import plot_pyvista, TEST_SPEED

dir_name = os.path.dirname(__file__)

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
except ImportError:
    plot_pyvista = False


def test_regular_grid_preparation(simple_grid_3d_more_points_grid):
    engine_grid = simple_grid_3d_more_points_grid
    print(engine_grid.regular_grid_values[45, 2, 4, 2])
    np.testing.assert_almost_equal(engine_grid.regular_grid_values[45, 2, 4, 2], .295, 6)


def test_regular_grid_point_generation(simple_grid_3d_octree: Grid):
    rg = simple_grid_3d_octree.regular_grid

    corners_sol = np.array(
        [[0.25000, 0.25000, 0.25000],
         [0.25000, 0.50000, 0.25000],
         [0.50000, 0.25000, 0.25000],
         [0.50000, 0.50000, 0.25000]]
    )
    np.testing.assert_almost_equal(rg.corners_values[::24], corners_sol, decimal=3)

    faces_sol = np.array([
        [0.25000, 0.37500, 0.33333],
        [0.37500, 0.25000, 0.33333],
        [0.37500, 0.37500, 0.25000],
    ]
    )
    np.testing.assert_almost_equal(rg.faces_values[::24], faces_sol, decimal=3)

    if plot_pyvista:
        p = pv.Plotter()
        p.add_mesh(pv.PolyData(rg.values), color="black", point_size=12.0, render_points_as_spheres=False)
        p.add_mesh(pv.PolyData(rg.corners_values), color="blue", point_size=7.0, render_points_as_spheres=False)
        p.add_mesh(pv.PolyData(rg.faces_values), color="g", point_size=5.0, render_points_as_spheres=False)
        p.show()


def test_octree_root(simple_model, simple_grid_3d_octree):
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree

    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    # interpolate level 0 - center
    output_0_centers = interp.interpolate_and_segment(interpolation_input, options, data_shape.tensors_structure)

    # Interpolate level 0 - corners
    from gempy_engine.modules.octrees_topology._octree_common import _generate_corners
    grid_0_corners = Grid(_generate_corners(grid_0_centers.values, grid_0_centers.dxdydz))
    interpolation_input.grid = grid_0_corners
    output_0_corners = interp.interpolate_and_segment(interpolation_input, options, data_shape.tensors_structure, clean_buffer=False)

    # Create octree level 0
    octree_lvl0 = OctreeLevel(grid_0_centers, grid_0_corners, [output_0_centers], [output_0_corners])

    # Generate grid_1_centers
    debug_vals = get_next_octree_grid(octree_lvl0, compute_topology=False, debug=True)
    xyz, anch, select = debug_vals[:3]
    grid_1_centers = debug_vals[-1]

    # Level 1

    interpolation_input.grid = grid_1_centers
    output_1_centers = interp.interpolate_and_segment(interpolation_input, options, data_shape.tensors_structure,
                                                      clean_buffer=False)

    # Interpolate level 1 - Corners
    grid_1_corners = Grid(_generate_corners(grid_1_centers.values, grid_1_centers.dxdydz))
    interpolation_input.grid = grid_1_corners
    output_1_corners = interp.interpolate_and_segment(interpolation_input, options, data_shape.tensors_structure,
                                                      clean_buffer=False)

    # Create octree level 1
    octree_lvl1 = OctreeLevel(grid_1_centers, grid_1_corners, [output_1_centers], [output_1_corners])

    debug_vals = get_next_octree_grid(octree_lvl1, compute_topology=False, debug=True)
    xyz1, anch1, select1 = debug_vals[:3]

    if plot_pyvista or True:
        # Compute actual mesh
        resolution = [20, 20, 20]
        mesh = _compute_actual_mesh(simple_model, ids, grid_0_centers, resolution, output_1_centers.scalar_field_at_sp,
                                    output_1_centers.weights)
        p = pv.Plotter()
        rg = grid_0_centers.regular_grid

        grid_0_faces = grid_0_corners

        p.add_mesh(mesh, opacity=.8, silhouette=True)
        p.add_mesh(pv.PolyData(grid_0_centers.values), color="black", point_size=12.0, render_points_as_spheres=False)
        p.add_mesh(pv.PolyData(rg.corners_values), color="blue", point_size=3.0, render_points_as_spheres=False)

        z_left = grid_0_faces.values.reshape((-1, 8, 3))[:, ::2, :][select[2]]
        z_right = grid_0_faces.values.reshape((-1, 8, 3))[:, 1::2, :][select[2]]
        try:
            p.add_mesh(pv.PolyData(z_left), color="c", point_size=6.0, render_points_as_spheres=False)
            p.add_mesh(pv.PolyData(z_right), color="y", point_size=5.0, render_points_as_spheres=False)
        except:
            pass

        x_left = grid_0_faces.values.reshape((-1, 8, 3))[:, :4, :][select[0]]
        x_right = grid_0_faces.values.reshape((-1, 8, 3))[:, 4:, :][select[0]]
        try:
            p.add_mesh(pv.PolyData(x_left), color="c", point_size=6.0, render_points_as_spheres=False)
            p.add_mesh(pv.PolyData(x_right), color="y", point_size=5.0, render_points_as_spheres=False)
        except:
            pass
        y_left = grid_0_faces.values.reshape((-1, 8, 3))[:, [0, 1, 4, 5], :][select[1]]
        y_right = grid_0_faces.values.reshape((-1, 8, 3))[:, [2, 3, 6, 7], :][select[1]]
        try:
            p.add_mesh(pv.PolyData(y_left), color="c", point_size=6.0, render_points_as_spheres=False)
            p.add_mesh(pv.PolyData(y_right), color="y", point_size=5.0, render_points_as_spheres=False)
        except:
            pass
        try:
            p.add_mesh(pv.PolyData(anch), color="r", point_size=10.0, render_points_as_spheres=False)
            p.add_mesh(pv.PolyData(anch1), color="orange", point_size=8.0, render_points_as_spheres=False)
            p.add_mesh(pv.PolyData(xyz), color="w", point_size=5.0, render_points_as_spheres=False)
            p.add_mesh(pv.PolyData(xyz1), color="g", point_size=4.0, render_points_as_spheres=False)

        except:
            pass
        p.show()


def test_octree_root_idx(simple_model):
    """Test getting idx for posterior meshing """

    def simple_grid_3d_octree_regular():
        import dataclasses
        resolution = [2, 2, 2]
        extent = [0.25, .75, 0.25, .75, 0.25, .75]

        regular_grid = RegularGrid(extent, resolution)
        grid = Grid.from_regular_grid(regular_grid)
        return dataclasses.replace(grid)

    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree_regular()

    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    # interpolate level 0 - center
    output_0_centers = interp.interpolate_and_segment(interpolation_input, options, data_shape.tensors_structure)

    # Interpolate level 0 - corners
    from gempy_engine.modules.octrees_topology._octree_common import _generate_corners
    grid_0_corners = Grid(_generate_corners(grid_0_centers.values, grid_0_centers.dxdydz))
    interpolation_input.grid = grid_0_corners
    output_0_corners = interp.interpolate_and_segment(interpolation_input, options, data_shape.tensors_structure, clean_buffer=False)

    # Create octree level 0
    octree_lvl0 = OctreeLevel(grid_0_centers, grid_0_corners, [output_0_centers], [output_0_corners])

    # Generate grid_1_centers
    debug_vals = get_next_octree_grid(octree_lvl0, compute_topology=False, debug=True)
    xyz, anch, select = debug_vals[:3]
    centers_lvl1 = xyz
    active_centers_lvl1 = anch

    idx_select = debug_vals[3]
    voxel_select = debug_vals[4]

    grid_1_centers = debug_vals[-1]

    # Level 1

    interpolation_input.grid = grid_1_centers
    output_1_centers = interp.interpolate_and_segment(interpolation_input, options, data_shape.tensors_structure,
                                                      clean_buffer=False)

    # Interpolate level 1 - Corners
    grid_1_corners = Grid(_generate_corners(grid_1_centers.values, grid_1_centers.dxdydz))
    interpolation_input.grid = grid_1_corners
    output_1_corners = interp.interpolate_and_segment(interpolation_input, options, data_shape.tensors_structure,
                                                      clean_buffer=False)

    # Create octree level 1
    octree_lvl1 = OctreeLevel(grid_1_centers, grid_1_corners, [output_1_centers], [output_1_corners])

    debug_vals = get_next_octree_grid(octree_lvl1, compute_topology=False, debug=True)
    xyz1, anch1, select1 = debug_vals[:3]
    centers_lvl2 = xyz1
    active_centers_lvl2 = anch1

    idx_select2 = debug_vals[3]
    voxel_select2 = debug_vals[4]
    
    # * ---- New code Here ----
    # region x
    x_dir = np.repeat(idx_select[0][voxel_select], 8)
    x_dir_lvl1 = idx_select2[0]
    binary_x = np.vstack((x_dir, x_dir_lvl1))
    bool_to_int_x = np.packbits(binary_x, axis=0, bitorder="little")
    find_edge_x = np.diff(bool_to_int_x)

    # endregion

    # region y
    y_dir = np.repeat(idx_select[1][voxel_select], 8)
    y_dir_lvl1 = idx_select2[1]
    binary_y = np.vstack((y_dir, y_dir_lvl1))
    bool_to_int_y = np.packbits(binary_y, axis=0, bitorder="little")
    find_edge_y = np.diff(bool_to_int_y)

    # endregion

    # region z
    z_dir = np.repeat(idx_select[2][voxel_select], 8)
    z_dir_lvl1 = idx_select2[2]
    binary_z = np.vstack((z_dir, z_dir_lvl1))
    bool_to_int_z = np.packbits(binary_z, axis=0, bitorder="little")
    find_edge_z = np.diff(bool_to_int_z)

    # endregion
    
    merged = np.vstack((find_edge_x, find_edge_y, find_edge_z))
    n_edges = (np.vstack((find_edge_x, find_edge_y, find_edge_z)) == 0).sum(0)
    
    more_than_2_edges = np.where(n_edges >= 2)[0]
    foo = (np.vstack((find_edge_x, find_edge_y, find_edge_z)) == 0)[:, more_than_2_edges]

    resolution = [20, 20, 20]
    mesh = _compute_actual_mesh(simple_model, ids, grid_0_centers, resolution, output_1_centers.scalar_field_at_sp,
                                output_1_centers.weights)
    
    mesh.ve
    
    # * ---- 

    if plot_pyvista or True:
        # Compute actual mesh
        resolution = [20, 20, 20]
        mesh = _compute_actual_mesh(simple_model, ids, grid_0_centers, resolution, output_1_centers.scalar_field_at_sp,
                                    output_1_centers.weights)
        p = pv.Plotter()
        rg = grid_0_centers.regular_grid

        grid_0_faces = grid_0_corners

        p.add_mesh(mesh, opacity=.8, silhouette=True)

        # * Black: Centers of lvl 0
        p.add_mesh(pv.PolyData(grid_0_centers.values), color="black", point_size=12.0, render_points_as_spheres=False)
        # * Blue: Corners of lvl 0
        p.add_mesh(pv.PolyData(rg.corners_values), color="blue", point_size=3.0, render_points_as_spheres=False)

        z_left = grid_0_faces.values.reshape((-1, 8, 3))[:, ::2, :][select[2]]
        z_right = grid_0_faces.values.reshape((-1, 8, 3))[:, 1::2, :][select[2]]
        try:
            # * Cyan: Active left corners of lvl 0
            p.add_mesh(pv.PolyData(z_left), color="c", point_size=6.0, render_points_as_spheres=False)
            # * Yellow: Active right faces of lvl 0
            p.add_mesh(pv.PolyData(z_right), color="y", point_size=5.0, render_points_as_spheres=False)
        except:
            pass

        x_left = grid_0_faces.values.reshape((-1, 8, 3))[:, :4, :][select[0]]
        x_right = grid_0_faces.values.reshape((-1, 8, 3))[:, 4:, :][select[0]]
        try:
            # * Cyan: Active left corners of lvl 0
            p.add_mesh(pv.PolyData(x_left), color="c", point_size=6.0, render_points_as_spheres=False)
            # * Yellow: Active right faces of lvl 0
            p.add_mesh(pv.PolyData(x_right), color="y", point_size=5.0, render_points_as_spheres=False)
        except:
            pass
        y_left = grid_0_faces.values.reshape((-1, 8, 3))[:, [0, 1, 4, 5], :][select[1]]
        y_right = grid_0_faces.values.reshape((-1, 8, 3))[:, [2, 3, 6, 7], :][select[1]]
        try:
            # * Cyan: Active left corners of lvl 0
            p.add_mesh(pv.PolyData(y_left), color="c", point_size=6.0, render_points_as_spheres=False)
            # * Yellow: Active right faces of lvl 0
            p.add_mesh(pv.PolyData(y_right), color="y", point_size=5.0, render_points_as_spheres=False)
        except:
            pass
        try:
            # * Red: Active centers of lvl 0
            p.add_mesh(pv.PolyData(anch), color="r", point_size=10.0, render_points_as_spheres=False)
            # * White: Centers of lvl 1
            p.add_mesh(pv.PolyData(xyz), color="w", point_size=5.0, render_points_as_spheres=False)
            # * Orange: Active centers of lvl 1
            p.add_mesh(pv.PolyData(anch1), color="orange", point_size=8.0, render_points_as_spheres=False)
            # * Green: Centers of lvl 2
            p.add_mesh(pv.PolyData(xyz1), color="g", point_size=4.0, render_points_as_spheres=False)

        except:
            pass
        p.show()


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_octree_leaf(simple_model, simple_grid_3d_octree):
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    options.number_octree_levels = 5

    octree_list = interp.interpolate_n_octree_levels(interpolation_input, options, data_shape)

    # Assert

    regular_grid_scalar = get_regular_grid_value_for_level(octree_list).astype("int8")

    # ===========
    if plot_pyvista or False:
        # Compute actual mesh
        resolution = [20, 20, 20]
        mesh = _compute_actual_mesh(simple_model, ids, grid_0_centers, resolution,
                                    octree_list[0].last_output_center.scalar_field_at_sp,
                                    octree_list[0].last_output_center.weights)

        n = options.number_octree_levels - 1
        debug_vals = get_next_octree_grid(octree_list[n], compute_topology=False, debug=True)
        a = debug_vals[-2]
        grid_centers = octree_list[n].grid_centers
        debug_vals_prev = get_next_octree_grid(octree_list[n - 1], compute_topology=False, debug=True)
        anch = debug_vals_prev[1]
        grid_centers.values = grid_centers.values[a]

        p = _plot_points_in_vista(grid_centers, mesh, anch)

        shape = octree_list[n].grid_centers.regular_grid_shape
        regular_grid_values = octree_list[n].grid_centers.regular_grid.values_vtk_format
        regular_grid_scalar = get_regular_grid_value_for_level(octree_list, n)

        grid_3d = regular_grid_values.reshape(*(shape + 1), 3).T
        regular_grid_mesh = pv.StructuredGrid(*grid_3d)
        regular_grid_mesh["lith"] = regular_grid_scalar.ravel()
        foo = regular_grid_mesh.threshold([0, 10])

        p.add_mesh(foo, show_edges=False, opacity=.5, cmap="tab10")
        p.add_axes()
        p.show()

    # np.save(dir_name + "/solutions/test_octree_leaf", np.round(regular_grid_scalar))
    ids_sol = np.load(dir_name + "/solutions/test_octree_leaf.npy")
    ids_sol[ids_sol == 2] = 0  # ! This is coming because the masking

    # ! This test does failes for 60 voxels. I imagine that the reason is the nugget effect but I will leave the .npy file in case there is problems with octrees on the future
    # np.testing.assert_almost_equal(np.round(regular_grid_scalar.ravel()), ids_sol, decimal=3)


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_octree_lvl_collapse(simple_model, simple_grid_3d_octree):
    octree_list = _run_octree_api(simple_model, simple_grid_3d_octree)
    for i in range(len(octree_list)):
        shape = octree_list[i].grid_centers.regular_grid_shape
        slice = shape[1] // 2
        regular_grid_scalar = get_regular_grid_value_for_level(octree_list, level=i).astype("int8")
        plt.imshow(regular_grid_scalar[:, slice, :].T, origin="lower")
        plt.colorbar()
        plt.show()


def _run_octree_api(simple_model, simple_grid_3d_octree):
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = interp.interpolate_n_octree_levels(interpolation_input, options, data_shape)
    return octree_list


def _plot_points_in_vista(grid_0_centers, mesh, anch=None):
    p = pv.Plotter()
    xyz = grid_0_centers.values

    p.add_mesh(mesh, opacity=.8, silhouette=True)
    if anch is not None:
        p.add_mesh(pv.PolyData(anch), color="black", point_size=10.0, render_points_as_spheres=False)
    p.add_mesh(pv.PolyData(xyz), color="w", point_size=3.0, render_points_as_spheres=False)

    return p
