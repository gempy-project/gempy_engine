import numpy as np
import pytest

from gempy_engine.core.data.exported_structs import OctreeLevel, InterpOutput
from gempy_engine.core.data.grid import Grid, RegularGrid
from gempy_engine.core.data.internal_structs import SolverInput
import gempy_engine.integrations.interp_single.interp_single_interface as interp
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.modules.activator.activator_interface import activate_formation_block
import matplotlib.pyplot as plt

from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_next_octree_grid, \
    get_regular_grid_for_level


def test_regular_grid_preparation(simple_grid_3d_more_points_grid):
    engine_grid = simple_grid_3d_more_points_grid
    print(engine_grid.regular_grid_values[45, 2, 4, 2])
    np.testing.assert_almost_equal(engine_grid.regular_grid_values[45, 2, 4, 2], .295)


def test_regular_grid_point_generation(simple_grid_3d_octree: Grid):
    rg = simple_grid_3d_octree.regular_grid

    import pyvista as pv
    p = pv.Plotter()
    if False:  # This only works for pseudo 2d
        dims = np.asarray((rg.regular_grid_shape)) + 1
        grid = pv.ExplicitStructuredGrid(dims, rg.corners_values)
        grid.compute_connectivity()
        p.add_mesh(grid)
    p.add_mesh(pv.PolyData(rg.values), color="black", point_size=12.0, render_points_as_spheres=False)
    p.add_mesh(pv.PolyData(rg.corners_values), color="blue", point_size=7.0, render_points_as_spheres=False)
    p.add_mesh(pv.PolyData(rg.faces_values), color="g", point_size=5.0, render_points_as_spheres=False)
    p.show()


def test_octree_and_topo_root(simple_model, simple_grid_3d_octree, pyvista_plot=True):
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree

    interpolation_input = InterpolationInput(spi,ori_i, grid_0_centers, ids)

    # interpolate level 0 - center
    output_0_centers = interp.interpolate_single_scalar(interpolation_input, options, data_shape)

    # Interpolate level 0 - faces
    from gempy_engine.modules.octrees_topology._octree_common import _generate_corners
    grid_0_corners = Grid(_generate_corners(grid_0_centers.values, grid_0_centers.dxdydz))
    interpolation_input.grid = grid_0_corners
    output_0_corners = interp.interpolate_single_scalar(interpolation_input, options, data_shape, clean_buffer=False)

    # Create octree level 0
    octree_lvl0 = OctreeLevel()

    octree_lvl0 = octree_lvl0.set_interpolation(grid_0_centers, grid_0_corners,
                                                output_0_centers, output_0_corners)

    # Generate grid_1_centers
    debug_vals = get_next_octree_grid(octree_lvl0, compute_topology=False, debug=True)
    xyz, anch, select = debug_vals[:3]

    grid_1_centers = debug_vals[-1]

    # Level 2
    octree_lvl1 = OctreeLevel()
    interpolation_input.grid = grid_1_centers

    output_1_centers = interp.interpolate_single_scalar(interpolation_input, options, data_shape, clean_buffer=False)
    # Interpolate level 0 - faces
    grid_1_corners = Grid(_generate_corners(grid_1_centers.values, grid_1_centers.dxdydz))
    interpolation_input.grid = grid_1_corners
    output_1_corners = interp.interpolate_single_scalar(interpolation_input, options, data_shape, clean_buffer=False)
    # Create octree level 0

    octree_lvl1.set_interpolation(grid_1_centers, grid_1_corners, output_1_centers, output_1_corners)

    debug_vals = get_next_octree_grid(octree_lvl1, compute_topology=False, debug=True)
    xyz1, anch1, select1 = debug_vals[:3]

    # Compute actual mesh
    resolution = [20, 20, 20]
    mesh = compute_actual_mesh(simple_model, ids, grid_0_centers, resolution,
                               output_1_centers.scalar_field_at_sp, output_1_centers.weights)

    if pyvista_plot:
        import pyvista as pv
        p = pv.Plotter()
        rg = grid_0_centers.regular_grid

        grid_0_faces = grid_0_corners

        if False:  # This only works for pseudo 2d
            dims = np.asarray((rg.regular_grid_shape)) + 1
            grid = pv.ExplicitStructuredGrid(dims, rg.corners_values)
            grid.compute_connectivity()
            p.add_mesh(grid, opacity=.5, color=True)

        p.add_mesh(mesh, opacity=.8, silhouette=True)
        p.add_mesh(pv.PolyData(grid_0_centers.values), color="black", point_size=12.0, render_points_as_spheres=False)
        p.add_mesh(pv.PolyData(rg.corners_values), color="blue", point_size=3.0, render_points_as_spheres=False)
#        p.add_mesh(pv.PolyData(rg.faces_values), color="g", point_size=8.0, render_points_as_spheres=False)

        z_left =grid_0_faces.values.reshape((-1,8,3))[:, ::2, :][select[2]]
        z_right = grid_0_faces.values.reshape((-1,8,3))[:, 1::2, :][select[2]]
        try:
            p.add_mesh(pv.PolyData(z_left), color="c", point_size=6.0, render_points_as_spheres=False)
            p.add_mesh(pv.PolyData(z_right), color="y", point_size=5.0, render_points_as_spheres=False)
        except:
            pass

        x_left = grid_0_faces.values.reshape((-1,8,3))[:,:4, :][select[0]]
        x_right = grid_0_faces.values.reshape((-1,8,3))[:, 4:, :][select[0]]
        try:
            p.add_mesh(pv.PolyData(x_left), color="c", point_size=6.0, render_points_as_spheres=False)
            p.add_mesh(pv.PolyData(x_right), color="y", point_size=5.0, render_points_as_spheres=False)
        except:
            pass
        y_left = grid_0_faces.values.reshape((-1,8,3))[:, [0, 1, 4, 5], :][select[1]]
        y_right = grid_0_faces.values.reshape((-1,8,3))[:, [2, 3, 6, 7], :][select[1]]
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


def test_octree_leaf_on_faces(simple_model, simple_grid_3d_octree, pyvista_plot=True):
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = interp.compute_n_octree_levels(5, interpolation_input, options, data_shape, on_faces=True)
    # Compute actual mesh
    resolution = [20, 20, 20]
    mesh = compute_actual_mesh(simple_model, ids, grid_0_centers, resolution,
                               octree_list[0].output_centers.scalar_field_at_sp,
                               octree_list[0].output_centers.weights)

    grid_centers = octree_list[-1].grid_centers
    grid_faces = octree_list[-1].grid_faces

    if pyvista_plot:
        plot_points_in_vista(grid_centers, grid_faces, mesh)

def test_octree_leaf(simple_model, simple_grid_3d_octree, pyvista_plot=True):
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = interp.compute_n_octree_levels(6, interpolation_input, options, data_shape, on_faces=False)
    # Compute actual mesh
    resolution = [20, 20, 20]
    mesh = compute_actual_mesh(simple_model, ids, grid_0_centers, resolution,
                               octree_list[0].output_centers.scalar_field_at_sp,
                               octree_list[0].output_centers.weights)



    debug_vals = get_next_octree_grid(octree_list[-1], compute_topology=False, debug=True)
    a = debug_vals[-2]

    grid_centers = octree_list[-1].grid_centers
    grid_centers.values = grid_centers.values[a]

    grid_faces = octree_list[-1].grid_faces

    if pyvista_plot:
        plot_points_in_vista(grid_centers, grid_faces, mesh)


def plot_points_in_vista(grid_0_centers, grid_0_faces, mesh):
    import pyvista as pv
    p = pv.Plotter()
    rg = grid_0_centers.regular_grid
    xyz = grid_0_centers.values

    if False:  # This only works for pseudo 2d
        dims = np.asarray((rg.regular_grid_shape)) + 1
        grid = pv.ExplicitStructuredGrid(dims, rg.corners_values)
        grid.compute_connectivity()
        p.add_mesh(grid, opacity=.5, color=True)
    p.add_mesh(mesh, opacity=.8, silhouette=True)

    if False:
        p.add_mesh(pv.PolyData(grid_0_centers.values), color="black", point_size=12.0, render_points_as_spheres=False)
    #p.add_mesh(pv.PolyData(rg.corners_values), color="blue", point_size=10.0, render_points_as_spheres=False)
#    p.add_mesh(pv.PolyData(rg.faces_values), color="g", point_size=8.0, render_points_as_spheres=False)

    if False:
        z_left = grid_0_faces.values  # .reshape((6, -1, 3))[4][select[2]]
        z_right = grid_0_faces.values  # .reshape((6, -1, 3))[5][select[2]]
        try:
            p.add_mesh(pv.PolyData(z_left), color="c", point_size=6.0, render_points_as_spheres=False)
            p.add_mesh(pv.PolyData(z_right), color="y", point_size=5.0, render_points_as_spheres=False)
        except:
            pass
        x_left = grid_0_faces.values  # .reshape((6, -1, 3))[0][select[0]]
        x_right = grid_0_faces.values  # .reshape((6, -1, 3))[1][select[0]]
        try:
            p.add_mesh(pv.PolyData(x_left), color="c", point_size=6.0, render_points_as_spheres=False)
            p.add_mesh(pv.PolyData(x_right), color="y", point_size=5.0, render_points_as_spheres=False)
        except:
            pass
        y_left = grid_0_faces.values#.reshape((6, -1, 3))[2][select[1]]
        y_right = grid_0_faces.values#.reshape((6, -1, 3))[3][select[1]]
        try:
            p.add_mesh(pv.PolyData(y_left), color="c", point_size=6.0, render_points_as_spheres=False)
            p.add_mesh(pv.PolyData(y_right), color="y", point_size=5.0, render_points_as_spheres=False)
        except:
            pass
    try:
      #  p.add_mesh(pv.PolyData(anch), color="r", point_size=4.0, render_points_as_spheres=False)
        p.add_mesh(pv.PolyData(xyz), color="w", point_size=3.0, render_points_as_spheres=False)
    except:
        pass
    p.show()


def test_octree_api(simple_model, simple_grid_3d_octree):
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = interp.compute_n_octree_levels(7, interpolation_input, options, data_shape)
    return octree_list


def test_octree_2d_plot(simple_model, simple_grid_3d_octree):
    octree_list = test_octree_api(simple_model, simple_grid_3d_octree)
    slice = 0
    # Lith - Only Level 0
    lvl0 = octree_list[0].output_centers
    plt.imshow(lvl0.ids_block_regular_grid[:, slice, :].T, origin="lower")
    plt.colorbar()
    plt.show()

    # Lith - Only Level 1
    lvl1 = octree_list[1].output_centers
    shape = octree_list[1].grid_centers.regular_grid_shape
    empty_regular_grid = np.zeros(shape,  dtype=float)
    bool_array = octree_list[1].grid_centers.regular_grid.active_cells

    w_ = np.where(bool_array)

    empty_regular_grid[w_] = lvl1.ids_block

    plt.imshow(empty_regular_grid[:, slice, :].T, origin="lower")
    plt.colorbar()
    plt.show()

    # Lith - Only Level 2
    lvl2 = octree_list[2].output_centers
    shape = octree_list[2].grid_centers.regular_grid_shape
    empty_regular_grid = np.zeros(shape, dtype=float)

    bool_array = octree_list[2].grid_centers.regular_grid.active_cells.ravel()
    w_1 = np.where(bool_array)

    bool_regular_grid2 = octree_list[1].grid_centers.regular_grid.active_cells
    bool_regular_grid2 = np.repeat(bool_regular_grid2, 2, axis=0)
    bool_regular_grid2 = np.repeat(bool_regular_grid2, 2, axis=1)
    bool_regular_grid2 = np.repeat(bool_regular_grid2, 2, axis=2)
    w_2 = np.where(bool_regular_grid2)

    empty_regular_grid[(w_2[0][w_1], w_2[1][w_1], w_2[2][w_1])] = lvl2.ids_block

    plt.imshow(empty_regular_grid[:, 0, :].T, origin="lower")
    plt.colorbar()
    plt.show()

    # Plot raveling
    w_3 = np.where(bool_regular_grid2.ravel())
    empty_regular_grid2 = empty_regular_grid.copy().ravel()
    empty_regular_grid2[w_3[0][w_1]] = lvl2.ids_block
    plt.imshow(empty_regular_grid2.reshape(shape)[:, 0, :].T, origin="lower")
    plt.colorbar()
    plt.show()


def test_octree_lvl_collapse(simple_model, simple_grid_3d_octree):
    octree_list = test_octree_api(simple_model, simple_grid_3d_octree)
    slice = 2

    for i in range(7):
        # # Level 0
        shape = octree_list[i].grid_centers.regular_grid_shape
        regular_grid_values = get_regular_grid_for_level(octree_list, i)
        plt.imshow(regular_grid_values.reshape(shape)[:, int(shape[1] / slice), :].T, origin="lower")
        plt.colorbar()
        plt.show()
    #
    # # # Level 0
    # shape = octree_list[0].grid_centers.regular_grid_shape
    # regular_grid_values = get_regular_grid_for_level(octree_list, 0)
    # plt.imshow(regular_grid_values.reshape(shape)[:, int(shape[1]/slice), :].T, origin="lower")
    # plt.colorbar()
    # plt.show()
    #
    # # # # Level 1
    # shape = octree_list[1].grid_centers.regular_grid_shape
    # regular_grid_values = get_regular_grid_for_level(octree_list, 1)
    # plt.imshow(regular_grid_values.reshape(shape)[:, int(shape[1]/slice), :].T, origin="lower")
    # plt.colorbar()
    # plt.show()
    # #
    # # # # Level 2
    # shape = octree_list[2].grid_centers.regular_grid_shape
    # regular_grid_values = get_regular_grid_for_level(octree_list, 2)
    # plt.imshow(regular_grid_values.reshape(shape)[:, int(shape[1]/slice), :].T, origin="lower")
    # plt.colorbar()
    # plt.show()
    #
    # # Level 3
    # shape = octree_list[3].grid_centers.regular_grid_shape
    # regular_grid_values = get_regular_grid_for_level(octree_list, 3)
    # plt.imshow(regular_grid_values.reshape(shape)[:, int(shape[1]/slice), :].T, origin="lower")
    # plt.colorbar()
    # plt.show()
    #
    #
    # # Level 4
    # shape = octree_list[6].grid_centers.regular_grid_shape
    # regular_grid_values = get_regular_grid_for_level(octree_list, 6)
    # plt.imshow(regular_grid_values.reshape(shape)[:, int(shape[1] / slice), :].T, origin="lower")
    # plt.colorbar()
    # plt.show()



# def test_octree_lvls_collapse(simple_model, simple_grid_3d_octree):
#     debug_vals, output_0_centers = test_octree_and_topo_root(simple_model, simple_grid_3d_octree, pyvista_plot=False)
#     xyz = debug_vals[0]
#     bool_array = debug_vals[3]
#
#     spi, ori_i, options, data_shape = simple_model
#     ids = np.array([1, 2])
#
#     # Create Next level interpolation input
#
#     regular_grid_1 = RegularGrid.init_regular_grid(
#         simple_grid_3d_octree.regular_grid.extent,
#         simple_grid_3d_octree.regular_grid.resolution * 2
#     )
#     grid_1_centers = Grid(xyz, regular_grid=regular_grid_1)
#
#     # Interpolate level 1
#     interpolation_input = InterpolationInput(spi, ori_i, grid_1_centers, ids)
#     output_1_centers = interp.interpolate_single_scalar(interpolation_input, options, data_shape)
#
#     if True:
#         slice = 0
#         # Lith - Only Level 0
#         lvl0 = output_0_centers
#         plt.imshow(lvl0.ids_block_regular_grid[:, slice, :].T, origin="lower")
#         plt.colorbar()
#         plt.show()
#
#
#         # Bool - level 1
#         plt.imshow(bool_array.reshape(2,3,3)[:, slice, :].T, origin="lower")
#         plt.colorbar()
#         plt.show()
#
#         # Lith - Only Level 1
#         empty_regular_grid = np.zeros(grid_1_centers.regular_grid_shape, dtype=float)  # TODO: Add this a regular grid property
#         bool_regular_grid = bool_array.reshape(2,3,3) # TODO: This is not going to be possible in the next level
#         bool_regular_grid = np.repeat(bool_regular_grid, 2, axis=0)
#         bool_regular_grid = np.repeat(bool_regular_grid, 2, axis=1)
#         bool_regular_grid = np.repeat(bool_regular_grid, 2, axis=2)
#
#         w_ = np.where(bool_regular_grid)
#         empty_regular_grid[w_] = output_1_centers.ids_block
#         plt.imshow(empty_regular_grid[:, slice, :].T, origin="lower")
#         plt.colorbar()
#         plt.show()
#
#         # Lith - Level 0 and Level 1
#         lvl0_id_block = lvl0.ids_block_regular_grid
#         lvl0_id_block = np.repeat(lvl0_id_block, 2, axis=0)
#         lvl0_id_block = np.repeat(lvl0_id_block, 2, axis=1)
#         lvl0_id_block = np.repeat(lvl0_id_block, 2, axis=2)
#         lvl0_id_block *= ~bool_regular_grid
#         lvl0_lvl1 = empty_regular_grid + lvl0_id_block
#         plt.imshow(lvl0_lvl1[:, slice, :].T, origin="lower")
#         plt.colorbar()
#         plt.show()


# ===============================================================================

def test_octree_multiple_levels(simple_model, simple_grid_3d_more_points_grid):
    octree_list = _compute_two_octtree_levels(simple_grid_3d_more_points_grid, simple_model)

    print(octree_list)

    if True:
        slice = 2
        plt.contourf(
            octree_list[0].id_block[:, slice, :].T, N=40, cmap="viridis",
            extent=(0.25, .75, 0.25, .75)
        )

        new_xyz = octree_list[1].xyz_coords
        plt.scatter(new_xyz[:, 0], new_xyz[:, 2], c="w", s=.5)

        plt.scatter(octree_list[-1].xyz_coords[:, 0], octree_list[-1].xyz_coords[:, 2], c="r", s=.3)

        plt.colorbar()
        plt.show()


def test_octree_high_res_grid_collapse(simple_model, simple_grid_3d_octree: Grid):
    octree_list = _compute_two_octtree_levels(simple_grid_3d_octree, simple_model)
    slice = 0

    # Lith - Only Level 0
    lvl0 = octree_list[0]
    plt.imshow(lvl0.id_block[:, slice, :].T, origin="lower")
    plt.show()

    lvl1 = octree_list[1]

    # Bool - level 1
    plt.imshow(lvl1.grid.regular_grid.active_cells[:, slice, :].T, origin="lower")
    plt.show()

    # Lith - Only Level 1
    empty_regular_grid = np.zeros_like(lvl1.grid.regular_grid.active_cells,
                                       dtype=float)  # TODO: Add this a regular grid property
    w_ = np.where(lvl1.grid.regular_grid.active_cells)
    empty_regular_grid[w_] = lvl1.id_block_centers
    plt.imshow(empty_regular_grid[:, slice, :].T, origin="lower")
    plt.show()

    # # Lith - Level 0 and Level 1
    # lvl0_id_block = lvl0.id_block
    # lvl0_id_block = np.repeat(lvl0_id_block, 2, axis=0)
    # lvl0_id_block = np.repeat(lvl0_id_block, 2, axis=1)
    # lvl0_id_block = np.repeat(lvl0_id_block, 2, axis=2)
    # lvl0_id_block *= ~lvl1.grid.regular_grid.active_cells
    # lvl0_lvl1 = empty_regular_grid + lvl0_id_block
    # plt.imshow(lvl0_lvl1[:, slice, :].T, origin="lower")
    # plt.show()
    #
    # # Lith - Only Level 2
    # lvl2 = octree_list[2]
    # empty_regular_grid = np.zeros_like(lvl2.grid.regular_grid.active_cells,
    #                                    dtype=float)  # TODO: Add this a regular grid property
    # w_ = np.where(lvl2.grid.regular_grid.active_cells)
    # empty_regular_grid[w_] = lvl2.id_block_centers
    # plt.imshow(empty_regular_grid[:, slice, :].T, origin="lower")
    # plt.show()


def test_grid_from_previous_regular_grid(simple_grid_3d_octree: Grid):
    new_values = np.linspace(0, 5, 9).reshape(-1, 3)

    new_grid = Grid(values=new_values,
                    len_grids=[0, 3],
                    regular_grid_shape=2 * simple_grid_3d_octree.regular_grid_shape,
                    dxdydz=simple_grid_3d_octree.dxdydz / 2
                    )

    print(new_grid.values)
    with pytest.raises(ValueError):
        a = new_grid.regular_grid_values


def _compute_two_octtree_levels(simple_grid_3d_more_points_grid, simple_model):
    grid = simple_grid_3d_more_points_grid
    ids = np.array([1, 2])
    unit_values = ids

    interp_input, output = _interpolate_to_grid(grid, ids, simple_model)

    octree_lvl0 = OctreeLevel(grid, output.ids_block_regular_grid,
                              output.exported_fields_regular_grid,
                              is_root=True)

    octree_lvl1 = compute_octree_root(octree_lvl0, compute_topology=True)
    n_levels = 4  # TODO: Move to options
    octree_list = [octree_lvl0, octree_lvl1]
    for i in range(2, n_levels):
        next_octree = interp.compute_octree_level_n(octree_list[-1], interp_input, output, unit_values, i)
        octree_list.append(next_octree)
    return octree_list


def _interpolate_to_grid(grid, ids, simple_model) -> (SolverInput, InterpOutput):
    surface_points = simple_model[0]
    orientations = simple_model[1]
    options = simple_model[2]
    data_shape = simple_model[3]
    grid_internal, ori_internal, sp_internal = interp.input_preprocess(data_shape, grid, orientations,
                                                                       surface_points)
    interp_input = SolverInput(sp_internal, ori_internal, options)
    weights = interp.solve_interpolation(interp_input)
    exported_fields = interp._evaluate_sys_eq(grid_internal, interp_input, weights)
    scalar_at_surface_points = interp._get_scalar_field_at_surface_points(
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
    return interp_input, output


def test_octree_multiple_levels_pyvista_sanity_check(simple_model, simple_grid_3d_octree):
    surface_points = simple_model[0]
    orientations = simple_model[1]
    options = simple_model[2]
    data_shape = simple_model[3]
    grid = simple_grid_3d_octree

    ids = np.array([1, 2])
    unit_values = ids

    grid_internal, ori_internal, sp_internal = interp.input_preprocess(data_shape, grid, orientations,
                                                                       surface_points)
    interp_input = SolverInput(sp_internal, ori_internal, options)

    weights = interp.solve_interpolation(interp_input)

    exported_fields = interp._evaluate_sys_eq(grid_internal, interp_input, weights)

    scalar_at_surface_points = interp._get_scalar_field_at_surface_points(
        exported_fields.scalar_field, data_shape.nspv, surface_points.n_points)

    # -----------------
    # Export and Masking operations can happen even in parallel
    # TODO: [~X] Export block
    values_block: np.ndarray = activate_formation_block(exported_fields.scalar_field, scalar_at_surface_points,
                                                        ids, sigmoid_slope=50000)

    resolution = [20, 20, 20]
    mesh = compute_actual_mesh(simple_model, ids, grid, resolution, scalar_at_surface_points, weights)

    output = InterpOutput()
    output.grid = grid
    output.exported_fields = exported_fields
    output.weights = weights
    output.scalar_field_at_sp = scalar_at_surface_points
    output.values_block = values_block

    octree_lvl0 = OctreeLevel(grid, output.ids_block_regular_grid,
                              output.exported_fields_regular_grid,
                              is_root=True)

    octree_lvl1 = compute_octree_root(octree_lvl0, compute_topology=True)

    n_levels = 4  # TODO: Move to options
    octree_list = [octree_lvl0, octree_lvl1]
    for i in range(2, n_levels):
        next_octree = interp.compute_octree_level_n(octree_list[-1], interp_input, output, unit_values, i)
        octree_list.append(next_octree)

    print(octree_list)

    import pyvista as pv
    p = pv.Plotter()
    p.add_mesh(mesh, show_edges=True, line_width=1)
    p.add_mesh(pv.PolyData(octree_list[0].xyz_coords), color="black", point_size=12.0)
    p.add_mesh(pv.PolyData(octree_list[1].xyz_coords), color="w", point_size=8.0, render_points_as_spheres=False)

    vertex_redux_ = octree_list[2].xyz_coords
    b1 = vertex_redux_[:, 0] > 0.24
    b2 = vertex_redux_[:, 0] < 0.76
    b3 = vertex_redux_[:, 1] > 0.24
    b4 = vertex_redux_[:, 1] < 0.76
    vertex_redux = vertex_redux_[b1 * b2 * b3 * b4]

    p.add_mesh(pv.PolyData(octree_list[2].xyz_coords), color="r", point_size=5.0, render_points_as_spheres=False)

    vertex_redux_ = octree_list[3].xyz_coords
    b1 = vertex_redux_[:, 0] > 0.24
    b2 = vertex_redux_[:, 0] < 0.76
    b3 = vertex_redux_[:, 1] > 0.24
    b4 = vertex_redux_[:, 1] < 0.76
    vertex_redux = vertex_redux_[b1 * b2 * b3 * b4]
    # p.add_mesh(pv.PolyData(vertex_redux), color="b", point_size=5.0)
    p.show()


def compute_actual_mesh(simple_model, ids, grid, resolution, scalar_at_surface_points, weights):
    surface_points = simple_model[0]
    orientations = simple_model[1]
    options = simple_model[2]
    data_shape = simple_model[3]
    grid_internal, ori_internal, sp_internal = interp.input_preprocess(data_shape, grid, orientations,
                                                                       surface_points)
    interp_input = SolverInput(sp_internal, ori_internal, options)
    values_block_high_res, scalar_high_res, dxdydz = compute_high_res_model(data_shape, ids, interp_input, orientations,
                                                                            resolution, scalar_at_surface_points,
                                                                            surface_points, weights)
    from skimage.measure import marching_cubes
    import pyvista as pv
    vert, edges, _, _ = marching_cubes(scalar_high_res.scalar_field[:-7].reshape(resolution),
                                       scalar_at_surface_points[0],
                                       spacing=dxdydz)
    loc_0 = np.array([0.25, .25, .25]) + np.array(dxdydz) / 2
    vert += np.array(loc_0).reshape(1, 3)
    mesh = pv.PolyData(vert, np.insert(edges, 0, 3, axis=1).ravel())
    return mesh


def test_octree_last_levels(simple_model, simple_grid_3d_octree):
    surface_points = simple_model[0]
    orientations = simple_model[1]
    options = simple_model[2]
    data_shape = simple_model[3]
    grid = simple_grid_3d_octree

    ids = np.array([1, 2])
    unit_values = ids

    grid_internal, ori_internal, sp_internal = interp.input_preprocess(data_shape, grid, orientations, surface_points)

    interp_input = SolverInput(sp_internal, ori_internal, options)
    weights = interp.solve_interpolation(interp_input)

    exported_fields = interp._evaluate_sys_eq(grid_internal, interp_input, weights)

    scalar_at_surface_points = interp._get_scalar_field_at_surface_points(
        exported_fields.scalar_field, data_shape.nspv, surface_points.n_points)

    # -----------------
    # Export and Masking operations can happen even in parallel
    # TODO: [~X] Export block
    values_block = activate_formation_block(exported_fields.scalar_field, scalar_at_surface_points,
                                            ids, sigmoid_slope=50000)

    # Create very high res grid:
    resolution = [100, 1, 100]
    values_block_high_res = compute_high_res_model(data_shape, ids, interp_input, orientations, resolution,
                                                   scalar_at_surface_points, surface_points, weights)

    output = InterpOutput()
    output.grid = grid
    output.exported_fields = exported_fields
    output.weights = weights
    output.scalar_field_at_sp = scalar_at_surface_points
    output.values_block = values_block

    grid = simple_grid_3d_octree

    octree_lvl0 = OctreeLevel(grid.values, output.ids_block_regular_grid,
                              output.exported_fields_regular_grid,
                              is_root=True)

    octree_lvl1 = compute_octree_root(octree_lvl0, grid.regular_grid_values, grid.dxdydz, compute_topology=True)
    octree_list = [octree_lvl0, octree_lvl1]

    n_levels = 4  # TODO: Move to options
    octree_list = [octree_lvl0, octree_lvl1]
    for i in range(2, n_levels):
        next_octree = interp.compute_octree_level_n(octree_list[-1], interp_input, output, unit_values, grid.dxdydz, i)
        octree_list.append(next_octree)

    last_octree = interp.compute_octree_last_level(octree_list[-1], interp_input, output, unit_values)
    octree_list.append(last_octree)
    if True:
        slice = 0
        plt.contourf(
            # exported_fields.scalar_field[:-7].reshape(grid.regular_grid_shape)[:, slice, :].T, N=40, cmap="autumn",
            values_block_high_res[0, :-7].reshape(resolution)[:, slice, :].T, N=40, cmap="viridis",
            # octree_lvl0.id_block[:, slice, :].T, N=40, cmap="viridis",

            extent=(0.25, .75, 0.25, .75)
        )
        plt.scatter(octree_list[0].xyz_coords[:, 0], octree_list[0].xyz_coords[:, 2], c="black", s=.5)
        plt.scatter(octree_list[1].xyz_coords[:, 0], octree_list[1].xyz_coords[:, 2], c="w", s=.5)
        plt.scatter(octree_list[2].xyz_coords[:, 0], octree_list[2].xyz_coords[:, 2], c="r", s=.4)
        plt.scatter(octree_list[3].xyz_coords[:, 0], octree_list[3].xyz_coords[:, 2], c="g", s=.3)
        # plt.scatter(octree_list[4].xyz_coords[:, 0], octree_list[4].xyz_coords[:, 2], c="b", s=.2)

        plt.colorbar()
        plt.show()


def compute_high_res_model(data_shape, ids, interp_input, orientations, resolution, scalar_at_surface_points,
                           surface_points, weights):
    from test.fixtures.simple_models import create_regular_grid
    from gempy_engine.core.data.grid import Grid
    g, dx, dy, dz = create_regular_grid([0.25, .75, 0.25, .75, 0.25, .75], resolution)
    grid_high_res = Grid(g, [g.shape[0]], resolution, [dx, dy, dz])
    grid_internal_high_res, ori_internal, sp_internal = interp.input_preprocess(data_shape, grid_high_res, orientations,
                                                                                surface_points)
    exported_fields_high_res = interp._evaluate_sys_eq(grid_internal_high_res, interp_input, weights)
    values_block_high_res = activate_formation_block(exported_fields_high_res.scalar_field, scalar_at_surface_points,
                                                     ids, sigmoid_slope=50000)
    return values_block_high_res, exported_fields_high_res, [dx, dy, dz]
