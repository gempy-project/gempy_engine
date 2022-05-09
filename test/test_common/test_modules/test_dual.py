import copy

import gempy_engine.integrations.interp_single._interp_single_internals
from gempy_engine.core.data.grid import Grid
import numpy as np
import os
import matplotlib.pyplot as plt
import pytest

from gempy_engine.integrations.dual_contouring.dual_contouring import get_intersection_on_edges, \
    compute_dual_contouring
from gempy_engine.modules.activator.activator_interface import activate_formation_block
import gempy_engine.integrations.interp_single.interp_single_interface as interp
from gempy_engine.core.data.internal_structs import SolverInput

from gempy_engine.core.data.exported_structs import OctreeLevel
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.integrations.interp_single.interp_single_interface import compute_n_octree_levels, \
    interpolate_and_segment
from gempy_engine.modules.dual_contouring.dual_contouring_interface import QEF, \
    find_intersection_on_edge
from gempy_engine.modules.octrees_topology.octrees_topology_interface import \
    get_regular_grid_for_level

dir_name = os.path.dirname(__file__)

plot_pyvista = False
try:
    # noinspection PyPackageRequirements
    import pyvista as pv
except ImportError:
    plot_pyvista = False


def test_find_edges_intersection_step_by_step(simple_model, simple_grid_3d_octree):
    # region Test find_intersection_on_edge
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = copy.deepcopy(simple_grid_3d_octree)
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = compute_n_octree_levels(3, interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[2]

    sfsp = last_octree_level.output_corners.scalar_field_at_sp

    dc_data = find_intersection_on_edge(
        last_octree_level.grid_corners.values,
        last_octree_level.output_corners.exported_fields.scalar_field,
        sfsp
    )

    xyz_on_edge, valid_edges = dc_data.xyz_on_edge, dc_data.valid_edges
    # endregion

    # region Get Normals

    interpolation_input.grid = Grid(xyz_on_edge)
    output_on_edges = interpolate_and_segment(interpolation_input, options, data_shape)
    # stack gradients output_on_edges.exported_fields.gx_field
    gradients = np.stack(
        (output_on_edges.exported_fields.gx_field,
         output_on_edges.exported_fields.gy_field,
         output_on_edges.exported_fields.gz_field), axis=0).T

    # endregion

    # region Prepare data for vectorized QEF

    n_edges = valid_edges.shape[0]

    # Coordinates for all posible edges (12) and 3 dummy normals in the center
    xyz = np.zeros((n_edges, 15, 3))
    normals = np.zeros((n_edges, 15, 3))

    xyz[:, :12][valid_edges] = xyz_on_edge
    normals[:, :12][valid_edges] = gradients

    BIAS_STRENGTH = 0.1

    xyz_aux = np.copy(xyz[:, :12])

    # Numpy zero values to nans
    xyz_aux[np.isclose(xyz_aux, 0)] = np.nan
    # Mean ignoring nans
    mass_points = np.nanmean(xyz_aux, axis=1)

    # mass_points = np.mean(xyz[:, :12], axis= 1)
    xyz[:, 12] = mass_points
    xyz[:, 13] = mass_points
    xyz[:, 14] = mass_points

    normals[:, 12] = np.array([BIAS_STRENGTH, 0, 0])
    normals[:, 13] = np.array([0, BIAS_STRENGTH, 0])
    normals[:, 14] = np.array([0, 0, BIAS_STRENGTH])

    # Remove unused voxels
    bo = valid_edges.sum(axis=1, dtype=bool)
    xyz = xyz[bo]
    normals = normals[bo]

    # NOTE(miguel): I leave the code for the first voxel here to understand what is happening bellow
    # Compute first QEF

    qef = QEF.make_3d(xyz[0], normals[0])
    residual, v_pro = qef.solve()

    # Linear list square fitting of A and B

    A = normals[0]
    B = xyz[0]
    BB = (A * B).sum(axis=1)
    v_pro = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, BB))

    # endregion

    # Compute QEF
    v_mesh = []
    indeces = valid_edges.sum(axis=1).cumsum()
    indices = np.unique(indeces)[0:2]

    for i in range(len(indices) - 1):
        i_0 = indices[i]
        i_1 = indices[i + 1]

        a = xyz_on_edge[i_0:i_1]
        b = gradients[i_0:i_1]

        mass_point = np.mean(a, axis=0)
        BIAS_STRENGTH = 0.1

        a = np.vstack((a, mass_point))
        a = np.vstack((a, mass_point))
        a = np.vstack((a, mass_point))
        b = np.vstack((b, np.array([BIAS_STRENGTH, 0, 0])))
        b = np.vstack((b, np.array([0, BIAS_STRENGTH, 0])))
        b = np.vstack((b, np.array([0, 0, BIAS_STRENGTH])))

        qef = QEF.make_3d(a, b)
        residual, v = qef.solve()

        v_mesh.append(v)

    if plot_pyvista or False:
        _plot_pyvista(last_octree_level, octree_list, simple_model, ids, grid_0_centers,
                      xyz_on_edge, gradients, a, b, v_mesh, v_pro, plot_label=False)

    return xyz_on_edge, gradients


def test_compute_dual_contouring_api(simple_model, simple_grid_3d_octree):
    # region Test find_intersection_on_edge
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = compute_n_octree_levels(2, interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[-1]

    dc_data = get_intersection_on_edges(last_octree_level)
    interpolation_input.grid = Grid(dc_data.xyz_on_edge)
    output_on_edges = interp.interpolate_single_field(interpolation_input, options,
                                                      data_shape)
    dc_data.gradients = output_on_edges.exported_fields

    mesh = compute_dual_contouring(dc_data, data_shape.n_surfaces)

    if plot_pyvista or False:
        pv.global_theme.show_edges = True
        p = pv.Plotter()

        from test.helper_functions import plot_dc_meshes
        plot_dc_meshes(p, mesh[0])
        p.show()


def test_compute_dual_contouring_several_meshes(simple_model_3_layers, simple_grid_3d_octree):
    # region Test find_intersection_on_edge
    interpolation_input, options, data_shape = simple_model_3_layers
    ids = np.array([1, 2, 3, 4])
    grid_0_centers = simple_grid_3d_octree
    # interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = compute_n_octree_levels(4, interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[-1]

    dc_data = get_intersection_on_edges(last_octree_level)
    interpolation_input.grid = Grid(dc_data.xyz_on_edge)
    output_on_edges = interp.interpolate_single_field(interpolation_input, options,
                                                      data_shape)
    dc_data.gradients = output_on_edges.exported_fields

    mesh = compute_dual_contouring(dc_data, data_shape.n_surfaces)

    if plot_pyvista or False:
        _plot_pyvista(last_octree_level, octree_list, simple_model_3_layers,
                      ids, grid_0_centers,
                      dc_data.xyz_on_edge,
                      dc_data.gradients,
                      v_pro = mesh[0].vertices,
                      indices = mesh[0].edges,
                      plot_label=False, plot_marching_cubes=False)

    # if plot_pyvista or True:
    #     pv.global_theme.show_edges = True
    #     p = pv.Plotter()
    #
    #     from test.helper_functions import plot_dc_meshes
    #     plot_dc_meshes(p, mesh[0])
    #     p.show()


def test_find_edges_intersection_pro(simple_model, simple_grid_3d_octree):
    # region Test find_intersection_on_edge
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = compute_n_octree_levels(2, interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[-1]

    sfsp = last_octree_level.output_corners.scalar_field_at_sp
    sfsp = np.append(sfsp, -0.1)
    dc_data = find_intersection_on_edge(
        last_octree_level.grid_corners.values,
        last_octree_level.output_corners.exported_fields.scalar_field,
        sfsp
    )
    xyz_on_edge, valid_edges = dc_data.xyz_on_edge, dc_data.valid_edges
    # endregion

    # region Get Normals

    interpolation_input.grid = Grid(xyz_on_edge)
    output_on_edges = interpolate_and_segment(interpolation_input, options, data_shape)
    # stack gradients output_on_edges.exported_fields.gx_field
    gradients = np.stack(
        (output_on_edges.exported_fields.gx_field,
         output_on_edges.exported_fields.gy_field,
         output_on_edges.exported_fields.gz_field), axis=0).T

    # endregion

    # region Prepare data for vectorized QEF

    n_edges = valid_edges.shape[0]

    # Coordinates for all posible edges (12) and 3 dummy normals in the center
    xyz = np.zeros((n_edges, 15, 3))
    normals = np.zeros((n_edges, 15, 3))

    xyz[:, :12][valid_edges] = xyz_on_edge
    normals[:, :12][valid_edges] = gradients

    BIAS_STRENGTH = 0.1

    xyz_aux = np.copy(xyz[:, :12])

    # Numpy zero values to nans
    xyz_aux[np.isclose(xyz_aux, 0)] = np.nan
    # Mean ignoring nans
    mass_points = np.nanmean(xyz_aux, axis=1)

    xyz[:, 12] = mass_points
    xyz[:, 13] = mass_points
    xyz[:, 14] = mass_points

    normals[:, 12] = np.array([BIAS_STRENGTH, 0, 0])
    normals[:, 13] = np.array([0, BIAS_STRENGTH, 0])
    normals[:, 14] = np.array([0, 0, BIAS_STRENGTH])

    # Remove unused voxels
    bo = valid_edges.sum(axis=1, dtype=bool)
    xyz = xyz[bo]
    normals = normals[bo]

    # Compute LSTSQS in all voxels at the same time
    A1 = normals
    b1 = xyz
    bb1 = (A1 * b1).sum(axis=2)
    s1 = np.einsum("ijk, ilj->ikl", A1, np.transpose(A1, (0, 2, 1)))
    s2 = np.linalg.inv(s1)
    s3 = np.einsum("ijk,ik->ij", np.transpose(A1, (0, 2, 1)), bb1)
    v_pro = np.einsum("ijk, ij->ik", s2, s3)

    np.testing.assert_array_almost_equal(
        v_pro[::10],
        np.array(
            [[0.522, 0.369, 0.388],
             [0.515, 0.368, 0.371],
             [0.181, 0.568, 0.332]]
        ), 3
    )

    # endregion

    # endregion

    if plot_pyvista or False:
        # region Compute triangles
        # For each edge that exhibits a sign change, generate a quad
        # connecting the minimizing vertices of the four cubes containing the edge.
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(v_pro)
            indices = hull.simplices
        except ImportError:
            indices = None
        _plot_pyvista(last_octree_level, octree_list, simple_model, ids, grid_0_centers,
                      xyz_on_edge, gradients, v_pro=v_pro, indices=indices)

    return xyz_on_edge, gradients


# =======================


def _plot_pyvista(last_octree_level, octree_list, simple_model, ids, grid_0_centers,
                  xyz_on_edge, gradients, a=None, b=None, v_mesh=None, v_pro=None, indices=None,
                  plot_label=False, plot_marching_cubes = True
                  ):
    n = 1
    p = pv.Plotter()

    # Plot Actual mesh (from marching cubes)
    if plot_marching_cubes:
        output_1_centers = last_octree_level.output_centers
        resolution = [20, 20, 20]
        mesh = _compute_actual_mesh(simple_model, ids, grid_0_centers, resolution,
                                    output_1_centers.scalar_field_at_sp, output_1_centers.weights)

        p.add_mesh(mesh, opacity=.8, silhouette=True)

    # Plot Regular grid Octree
    regular_grid_values = octree_list[n].grid_centers.regular_grid.values_vtk_format
    regular_grid_scalar = get_regular_grid_for_level(octree_list, n)

    shape = octree_list[n].grid_centers.regular_grid_shape
    grid_3d = regular_grid_values.reshape(*(shape + 1), 3).T
    regular_grid_mesh = pv.StructuredGrid(*grid_3d)
    regular_grid_mesh["lith"] = regular_grid_scalar.ravel()
    foo = regular_grid_mesh.threshold([0, 10])

    p.add_mesh(foo, show_edges=True, opacity=.5, cmap="tab10")

    # Plot gradients
    poly = pv.PolyData(xyz_on_edge)
    poly['vectors'] = gradients
    arrows = poly.glyph(orient='vectors', scale=False, factor=.05)
    p.add_mesh(arrows, color="k", point_size=10.0, render_points_as_spheres=False)

    if plot_label:
        p.add_point_labels(xyz_on_edge, list(range(xyz_on_edge.shape[0])), point_size=20, font_size=36)

    if a is not None and b is not None:
        poly = pv.PolyData(a)
        poly['vectors'] = b

        arrows = poly.glyph(orient='vectors', scale=False, factor=.05)

        p.add_mesh(arrows, color="green", point_size=10.0, render_points_as_spheres=False)

    # Plot QEF
    if v_mesh is not None:
        p.add_mesh(pv.PolyData(v_mesh), color="b", point_size=15.0, render_points_as_spheres=False)

    if v_pro is not None:
        p.add_mesh(pv.PolyData(v_pro), color="w", point_size=15.0, render_points_as_spheres=True)

    if indices is not None:
        dual_mesh = pv.PolyData(v_pro, np.insert(indices, 0, 3, axis=1).ravel())
        p.add_mesh(dual_mesh, opacity=1, silhouette=True, color="green")

    p.add_axes()
    p.show()


def _compute_actual_mesh(simple_model, ids, grid, resolution, scalar_at_surface_points, weights):
    def _compute_high_res_model(data_shape, ids, interp_input, orientations, resolution,
                                scalar_at_surface_points,
                                surface_points, weights):
        from gempy_engine.core.data.grid import Grid, RegularGrid

        grid_high_res = Grid.from_regular_grid(
            RegularGrid([0.25, .75, 0.25, .75, 0.25, .75], resolution))
        grid_internal_high_res, ori_internal, sp_internal = gempy_engine.integrations.interp_single._interp_single_internals._input_preprocess(
            data_shape, grid_high_res, orientations, surface_points)
        exported_fields_high_res = gempy_engine.integrations.interp_single._interp_single_internals._evaluate_sys_eq(
            grid_internal_high_res, interp_input, weights)

        exported_fields_high_res.n_points_per_surface = data_shape.reference_sp_position
        exported_fields_high_res.n_surface_points = surface_points.n_points

        values_block_high_res = activate_formation_block(exported_fields_high_res,
                                                         ids, sigmoid_slope=50000)

        return values_block_high_res, exported_fields_high_res, grid_high_res.dxdydz

    surface_points = simple_model[0]
    orientations = simple_model[1]
    options = simple_model[2]
    data_shape = simple_model[3]
    grid_internal, ori_internal, sp_internal = gempy_engine.integrations.interp_single._interp_single_internals._input_preprocess(
        data_shape, grid, orientations, surface_points)
    interp_input = SolverInput(sp_internal, ori_internal, options)
    values_block_high_res, scalar_high_res, dxdydz = _compute_high_res_model(data_shape, ids,
                                                                             interp_input,
                                                                             orientations,
                                                                             resolution,
                                                                             scalar_at_surface_points,
                                                                             surface_points,
                                                                             weights)
    from skimage.measure import marching_cubes
    import pyvista as pv
    vert, edges, _, _ = marching_cubes(scalar_high_res.scalar_field.reshape(resolution),
                                       scalar_at_surface_points[0],
                                       spacing=dxdydz)
    loc_0 = np.array([0.25, .25, .25]) + np.array(dxdydz) / 2
    vert += np.array(loc_0).reshape(1, 3)
    mesh = pv.PolyData(vert, np.insert(edges, 0, 3, axis=1).ravel())
    return mesh
