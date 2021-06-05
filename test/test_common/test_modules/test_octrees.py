import numpy as np
import pytest

from gempy_engine.core.data.exported_structs import OctreeLevel, InterpOutput
from gempy_engine.core.data.grid import Grid, RegularGrid
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.integrations.interp_single.interp_single_interface import input_preprocess, solve_interpolation, \
    _evaluate_sys_eq, _get_scalar_field_at_surface_points, compute_octree_level_n, compute_octree_last_level
from gempy_engine.modules.activator.activator_interface import activate_formation_block
from gempy_engine.modules.octrees_topology.octrees_topology_interface import  compute_octree_root
import matplotlib.pyplot as plt


def test_regular_grid_preparation(simple_grid_3d_more_points_grid):
    engine_grid = simple_grid_3d_more_points_grid
    print(engine_grid.regular_grid_values[45, 2, 4, 2])
    np.testing.assert_almost_equal(engine_grid.regular_grid_values[45, 2, 4, 2], .295)


def test_regular_grid_point_generation(simple_grid_3d_octree: Grid):
    rg = simple_grid_3d_octree.regular_grid

    import pyvista as pv

    dims = np.asarray((rg.regular_grid_shape)) + 1
    grid = pv.ExplicitStructuredGrid(dims, rg.corners_values)
    grid.compute_connectivity()
    #grid.plot(show_edges=True)

    p = pv.Plotter()
    p.add_mesh(grid)
    p.add_mesh(pv.PolyData(rg.values), color="black", point_size=12.0, render_points_as_spheres=False)
    p.add_mesh(pv.PolyData(rg.corners_values), color="blue", point_size=7.0, render_points_as_spheres=False)
    p.add_mesh(pv.PolyData(rg.faces_values), color="g", point_size=5.0, render_points_as_spheres=False)
    p.show()

    # p = pv.Plotter()
    # p.add_mesh(pv.PolyData(rg.values), color="black", point_size=12.0, render_points_as_spheres=False)
    # p.add_mesh(pv.PolyData(rg.corners_values), color="blue", point_size=7.0, render_points_as_spheres=False)
    # p.add_mesh(pv.PolyData(rg.faces_values), color="g", point_size=5.0, render_points_as_spheres=False)
    #p.show()



def test_octree_and_topo_root(simple_model, simple_grid_3d_octree):
    grid_0_centers = simple_grid_3d_octree
    ids = np.array([1, 2])
    unit_values = ids

    # interpolate level 0 - center
    interp_input, output_0_centers = _interpolate_to_grid(grid_0_centers, ids, simple_model)

    # Compute actual mesh
    resolution = [20, 20, 20]
    mesh = compute_actual_mesh(simple_model, ids, grid_0_centers, resolution,
                               output_0_centers.scalar_field_at_sp, output_0_centers.weights)

    # Interpolate level 0 - faces
    grid_0_faces = Grid(grid_0_centers.regular_grid.faces_values, regular_grid= grid_0_centers.regular_grid)
    interp_input,  output_0_faces = _interpolate_to_grid(grid_0_faces, ids, simple_model)
    ids_block = output_0_faces.ids_block

    # Create octree level 0
    octree_lvl0 = OctreeLevel(grid_0_faces, output_0_faces, is_root=True)

    # Generate grid_1_centers
    xyz, anch, select = compute_octree_root(octree_lvl0, compute_topology=False)

    import pyvista as pv


    rg = grid_0_centers.regular_grid


    p = pv.Plotter()
    dims = np.asarray((rg.regular_grid_shape)) + 1
    grid = pv.ExplicitStructuredGrid(dims, rg.corners_values)
    grid.compute_connectivity()
   # p.add_mesh(grid, opacity=.5, color=True)
    p.add_mesh(mesh, opacity=.8, silhouette=True)
    p.add_mesh(pv.PolyData(grid_0_centers.values), color="black", point_size=12.0, render_points_as_spheres=False)
    p.add_mesh(pv.PolyData(rg.corners_values), color="blue", point_size=10.0, render_points_as_spheres=False)
    p.add_mesh(pv.PolyData(rg.faces_values), color="g", point_size=8.0, render_points_as_spheres=False)

    z_left =  grid_0_faces.values.reshape((6, -1, 3))[4][select[2]]
    z_right = grid_0_faces.values.reshape((6, -1, 3))[5][select[2]]
    try:
        p.add_mesh(pv.PolyData(z_left), color="c", point_size=6.0, render_points_as_spheres=False)
        p.add_mesh(pv.PolyData(z_right), color="y", point_size=5.0, render_points_as_spheres=False)
    except:
        pass

    x_left =  grid_0_faces.values.reshape((6, -1, 3))[0][select[0]]
    x_right = grid_0_faces.values.reshape((6, -1, 3))[1][select[0]]
    try:
        p.add_mesh(pv.PolyData(x_left), color="c", point_size=6.0, render_points_as_spheres=False)
        p.add_mesh(pv.PolyData(x_right), color="y", point_size=5.0, render_points_as_spheres=False)
    except:
        pass
    y_left =  grid_0_faces.values.reshape((6, -1, 3))[2][select[1]]
    y_right = grid_0_faces.values.reshape((6, -1, 3))[3][select[1]]
    try:
        p.add_mesh(pv.PolyData(y_left), color="c", point_size=6.0, render_points_as_spheres=False)
        p.add_mesh(pv.PolyData(y_right), color="y", point_size=5.0, render_points_as_spheres=False)
    except:
        pass
    try:
        p.add_mesh(pv.PolyData(anch), color="r", point_size=4.0, render_points_as_spheres=False)
        p.add_mesh(pv.PolyData(xyz), color="w", point_size=3.0, render_points_as_spheres=False)
    except:
        pass
    p.show()


    pass


    #
    # octree_lvl0 = OctreeLevel(grid, output.ids_block_regular_grid,
    #                           output.exported_fields_regular_grid,
    #                           is_root=True)
    #
    # octree_lvl1 = compute_octree_root(octree_lvl0, compute_topology=True)
    #
    # np.testing.assert_array_almost_equal(octree_lvl0.count_edges, np.array([413, 32], dtype=int))
    #
    # print(f"Edges id: {octree_lvl0.edges_id}")
    # print(f"Count edges: {octree_lvl0.count_edges}")
    #
    # if False:
    #     slice = 2
    #     plt.contourf(
    #         octree_lvl0.id_block[:, slice, :].T, N=40, cmap="viridis",
    #         extent=(0.25, .75, 0.25, .75)
    #     )
    #
    #     new_xyz = octree_lvl1.xyz_coords
    #     plt.scatter(new_xyz[:, 0], new_xyz[:, 2], c="w", s=.5)
    #
    #     plt.colorbar()
    #     plt.show()
    #
    # import pyvista as pv
    # p = pv.Plotter()
    # p.add_mesh(pv.PolyData(octree_lvl0.grid.regular_grid_values), color="black", point_size=12.0, render_points_as_spheres=False)
    # p.add_mesh(pv.PolyData(octree_lvl1.grid.regular_grid_values), color="blue", point_size=7.0, render_points_as_spheres=False)
    # p.show()

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
    empty_regular_grid = np.zeros_like(lvl1.grid.regular_grid.active_cells, dtype=float) # TODO: Add this a regular grid property
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
        next_octree = compute_octree_level_n(octree_list[-1], interp_input, output, unit_values, i)
        octree_list.append(next_octree)
    return octree_list


def _interpolate_to_grid(grid, ids, simple_model) -> (SolverInput, InterpOutput):
    surface_points = simple_model[0]
    orientations = simple_model[1]
    options = simple_model[2]
    data_shape = simple_model[3]
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
    return interp_input, output


def test_octree_multiple_levels_pyvista_sanity_check(simple_model, simple_grid_3d_octree):
    surface_points = simple_model[0]
    orientations = simple_model[1]
    options = simple_model[2]
    data_shape = simple_model[3]
    grid = simple_grid_3d_octree

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
        next_octree = compute_octree_level_n(octree_list[-1], interp_input, output, unit_values, i)
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
    grid_internal, ori_internal, sp_internal = input_preprocess(data_shape, grid, orientations,
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

    grid_internal, ori_internal, sp_internal = input_preprocess(data_shape, grid, orientations, surface_points)

    interp_input = SolverInput(sp_internal, ori_internal, options)
    weights = solve_interpolation(interp_input)

    exported_fields = _evaluate_sys_eq(grid_internal, interp_input, weights)

    scalar_at_surface_points = _get_scalar_field_at_surface_points(
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
        next_octree = compute_octree_level_n(octree_list[-1], interp_input, output, unit_values, grid.dxdydz, i)
        octree_list.append(next_octree)

    last_octree = compute_octree_last_level(octree_list[-1], interp_input, output, unit_values)
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
    grid_internal_high_res, ori_internal, sp_internal = input_preprocess(data_shape, grid_high_res, orientations,
                                                                         surface_points)
    exported_fields_high_res = _evaluate_sys_eq(grid_internal_high_res, interp_input, weights)
    values_block_high_res = activate_formation_block(exported_fields_high_res.scalar_field, scalar_at_surface_points,
                                                     ids, sigmoid_slope=50000)
    return values_block_high_res, exported_fields_high_res, [dx, dy, dz]
