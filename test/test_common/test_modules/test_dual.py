import copy
from typing import List

import pytest

from gempy_engine.API.interp_single._interp_scalar_field import _evaluate_sys_eq
from gempy_engine.API.interp_single._interp_single_feature import input_preprocess
from gempy_engine.API.interp_single._multi_scalar_field_manager import interpolate_all_fields
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data.grid import Grid, RegularGrid
import numpy as np
import os

import gempy_engine.API.interp_single.interp_features as interp

from gempy_engine.API.dual_contouring._dual_contouring import get_intersection_on_edges, compute_dual_contouring
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.modules.activator.activator_interface import activate_formation_block
from gempy_engine.core.data.internal_structs import SolverInput

from gempy_engine.core.data.solutions import Solutions
from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.core.data.dual_contouring_data import DualContouringData
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.API.interp_single.interp_features import interpolate_n_octree_levels, interpolate_and_segment
from gempy_engine.modules.dual_contouring.dual_contouring_interface import QEF, find_intersection_on_edge, triangulate_dual_contouring
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level
from test import helper_functions_pyvista
from test.conftest import TEST_SPEED

dir_name = os.path.dirname(__file__)

plot_pyvista = False
try:
    # noinspection PyPackageRequirements
    import pyvista as pv
except ImportError:
    plot_pyvista = False


def test_compute_dual_contouring_api(simple_model, simple_grid_3d_octree):
    # region Test find_intersection_on_edge
    spi, ori_i, options, data_shape = simple_model
    options.compute_scalar_gradient = True
    
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = interpolate_n_octree_levels(interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[-1]

    intersection_xyz, valid_edges = get_intersection_on_edges(last_octree_level, last_octree_level.outputs_corners[0])
    interpolation_input.grid = Grid(intersection_xyz)
    output_on_edges: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, data_shape)

    dc_data = DualContouringData(
        xyz_on_edge=intersection_xyz,
        valid_edges=valid_edges,
        xyz_on_centers=last_octree_level.grid_centers.values,
        dxdydz=last_octree_level.grid_centers.dxdydz,
        exported_fields_on_edges=output_on_edges[0].exported_fields,
        n_surfaces=data_shape.tensors_structure.n_surfaces
    )

    gradients = dc_data.gradients

    dc_meshes: List[DualContouringMesh] = compute_dual_contouring(dc_data)

    dc_data = dc_meshes[0].dc_data
    valid_voxels = dc_data.valid_voxels

    # Mark active voxels
    temp_ids = octree_list[-1].last_output_center.ids_block  # ! I need this because setters in python sucks
    temp_ids[valid_voxels] = 5
    octree_list[-1].last_output_center.ids_block = temp_ids  # paint valid voxels

    if plot_pyvista or False:
        output_corners: InterpOutput = last_octree_level.outputs_corners[-1]
        vertices = output_corners.grid.values
        intersection_points = intersection_xyz
        center_mass = dc_data.bias_center_mass
        normals = dc_data.bias_normals
        helper_functions_pyvista.plot_pyvista(octree_list, dc_meshes=dc_meshes, gradients=gradients,
                                              a=center_mass, b=normals,
                                              xyz_on_edge=intersection_xyz,
                                              v_just_points=vertices, vertices=intersection_points)
    # endregion


def test_compute_dual_contouring_fancy_triangulation(simple_model, simple_grid_3d_octree):
    from gempy_engine.modules.dual_contouring.fancy_triangulation import get_left_right_array, triangulate
    def simple_grid_3d_octree_regular():
        import dataclasses
        resolution = [2, 2, 2]
        extent = [0.25, .75, 0.25, .75, 0.25, .75]

        regular_grid = RegularGrid(extent, resolution)
        grid = Grid.from_regular_grid(regular_grid)
        return dataclasses.replace(grid)

    # region Test find_intersection_on_edge
    spi, ori_i, options, data_shape = simple_model

    options.number_octree_levels = 5
    options.compute_scalar_gradient = True

    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree_regular()
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = interpolate_n_octree_levels(interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[-1]

    intersection_xyz, valid_edges = get_intersection_on_edges(last_octree_level, last_octree_level.outputs_corners[0])
    interpolation_input.grid = Grid(intersection_xyz)
    output_on_edges: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, data_shape)

    dc_data = DualContouringData(
        xyz_on_edge=intersection_xyz,
        valid_edges=valid_edges,
        xyz_on_centers=last_octree_level.grid_centers.values,
        dxdydz=last_octree_level.grid_centers.dxdydz,
        exported_fields_on_edges=output_on_edges[0].exported_fields,
        n_surfaces=data_shape.tensors_structure.n_surfaces
    )

    dc_meshes: List[DualContouringMesh] = compute_dual_contouring(dc_data)
    dc_data = dc_meshes[0].dc_data
    valid_voxels = dc_data.valid_voxels

    # Mark active voxels
    temp_ids = last_octree_level.last_output_center.ids_block  # ! I need this because setters in python sucks
    temp_ids[valid_voxels] = 5
    last_octree_level.last_output_center.ids_block = temp_ids  # paint valid voxels

    # * ---- New code Here ----

    stacked = get_left_right_array(octree_list)
    validated_stacked = stacked[valid_voxels]
    validated_edges = valid_edges[valid_voxels]
    indices_array: np.ndarray = triangulate(validated_stacked, validated_edges, options.number_octree_levels)

    # endregion

    # TODO: Plot the edges
    n_edges = valid_edges.shape[0]
    edges_xyz = np.zeros((n_edges, 15, 3))
    edges_xyz[:, :12][valid_edges] = intersection_xyz

    if plot_pyvista or False:
        intersection_points = intersection_xyz
        center_mass = dc_data.bias_center_mass
        p = helper_functions_pyvista.plot_pyvista(
            octree_list,
            # dc_meshes=dc_meshes, Uncomment to see the OG mesh
            a=center_mass,
            xyz_on_edge=intersection_xyz,
            vertices=intersection_points,
            plot=False
        )

        for e, indices_array_ in enumerate(indices_array):
            # paint the triangles in different colors
            color = ["b", "r", "m", "y", "k", "w"][e % 6]

            fancy_mesh_complete = pv.PolyData(dc_meshes[0].vertices, np.insert(indices_array_, 0, 3, axis=1).ravel())
            p.add_mesh(fancy_mesh_complete, silhouette=False, color=color, show_edges=True)

        p.show()
        # endregion


def test_compute_dual_contouring_complex(unconformity_complex_one_layer, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex_one_layer

    options.debug = True

    options.number_octree_levels = n_oct_levels
    solutions: Solutions = compute_model(interpolation_input, options, structure)
    dc_data = solutions.dc_meshes[0].dc_data

    if plot_pyvista or False:
        output_corners: InterpOutput = solutions.octrees_output[-1].outputs_corners[-1]
        vertices = output_corners.grid.values

        intersection_xyz = dc_data.xyz_on_edge
        gradients = dc_data.gradients

        center_mass = dc_data.bias_center_mass
        normals = dc_data.bias_normals
        helper_functions_pyvista.plot_pyvista(solutions.octrees_output,
                                              dc_meshes=solutions.dc_meshes,
                                              xyz_on_edge=intersection_xyz, gradients=gradients,
                                              a=center_mass, b=normals,
                                              # v_just_points=vertices
                                              )


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_compute_dual_contouring_several_meshes(simple_model_3_layers, simple_grid_3d_octree):
    # region Test find_intersection_on_edge
    interpolation_input, options, data_shape = simple_model_3_layers
    options.compute_scalar_gradient = True
    
    ids = np.array([1, 2, 3, 4])
    grid_0_centers = simple_grid_3d_octree

    octree_list = interpolate_n_octree_levels(interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[-1]

    intersection_xyz, valid_edges = get_intersection_on_edges(last_octree_level, last_octree_level.outputs_corners[0])
    interpolation_input.grid = Grid(intersection_xyz)
    output_on_edges = interp.interpolate_single_field(interpolation_input, options, data_shape.tensors_structure)

    dc_data = DualContouringData(
        xyz_on_edge=intersection_xyz,
        valid_edges=valid_edges,
        xyz_on_centers=last_octree_level.grid_centers.values,
        dxdydz=last_octree_level.grid_centers.dxdydz,
        exported_fields_on_edges=output_on_edges.exported_fields,
        n_surfaces=data_shape.tensors_structure.n_surfaces
    )

    mesh = compute_dual_contouring(dc_data)

    if plot_pyvista or False:
        _plot_pyvista(last_octree_level, octree_list, simple_model_3_layers,
                      ids, grid_0_centers,
                      dc_data.xyz_on_edge,
                      dc_data.gradients,
                      v_pro=mesh[0].vertices,
                      indices=mesh[0].edges,
                      plot_label=False, plot_marching_cubes=False)


def test_find_edges_intersection_step_by_step(simple_model, simple_grid_3d_octree):
    # region Test find_intersection_on_edge
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = copy.deepcopy(simple_grid_3d_octree)
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    options.number_octree_levels = 5
    octree_list = interpolate_n_octree_levels(interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[2]

    sfsp = last_octree_level.last_output_corners.scalar_field_at_sp

    xyz_on_edge, valid_edges = find_intersection_on_edge(last_octree_level.grid_corners.values, last_octree_level.output_corners.exported_fields.scalar_field, sfsp, )

    # endregion

    # region Get Normals

    interpolation_input.grid = Grid(xyz_on_edge)
    output_on_edges = interpolate_and_segment(interpolation_input, options, data_shape.tensors_structure)

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

    # * NOTE(miguel): I leave the code for the first voxel here to understand what is happening bellow
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
    indices = np.unique(indeces)  # [0:2]  # * show only 2 vertices

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
                      xyz_on_edge, gradients, a, b, v_pro, np.array(v_mesh), indices=None,
                      plot_label=False, plot_marching_cubes=False)

    return xyz_on_edge, gradients


def test_find_edges_intersection_pro(simple_model, simple_grid_3d_octree):
    # region Test find_intersection_on_edge
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = interpolate_n_octree_levels(interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[-1]

    sfsp = last_octree_level.output_corners.scalar_field_at_sp
    # sfsp = np.append(sfsp, -0.1)
    xyz_on_edge, valid_edges = find_intersection_on_edge(last_octree_level.grid_corners.values, last_octree_level.output_corners.exported_fields.scalar_field, sfsp, )
    # endregion

    # region Get Normals

    interpolation_input.grid = Grid(xyz_on_edge)
    output_on_edges = interpolate_and_segment(interpolation_input, options, data_shape.tensors_structure)
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

    # endregion

    # endregion

    # region triangulate
    grid_centers = last_octree_level.grid_centers
    valid_voxels = valid_edges.sum(axis=1, dtype=bool)

    temp_ids = octree_list[-1].last_output_center.ids_block  # ! I need this because setters in python sucks
    temp_ids[valid_voxels] = 5
    octree_list[-1].last_output_center.ids_block = temp_ids  # paint valid voxels

    dc_data = DualContouringData(
        xyz_on_edge=xyz_on_edge,
        xyz_on_centers=grid_centers.values,
        dxdydz=grid_centers.dxdydz,
        valid_edges=valid_edges,
        exported_fields_on_edges=None,
        n_surfaces=data_shape.tensors_structure.n_surfaces
    )

    indices = triangulate_dual_contouring(dc_data)
    # endregion

    if plot_pyvista or False:
        # ! I leave this test for the assert as comparison to the other implementation. The model looks bad
        # ! with this level of BIAS
        center_mass = xyz[:, 12:].reshape(-1, 3)
        normals = normals[:, 12:].reshape(-1, 3)
        _plot_pyvista(last_octree_level, octree_list, simple_model, ids, grid_0_centers,
                      xyz_on_edge, gradients, a=center_mass, b=normals,
                      v_pro=v_pro, indices=indices, plot_marching_cubes=True
                      )

    return xyz_on_edge, gradients


def test_find_edges_intersection_bias_on_center_of_the_cell(simple_model, simple_grid_3d_octree):
    """This looks works that taking the center of gravity of the edges intersections. I leave this test here as 
    documentation"""

    # region Test find_intersection_on_edge
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = interpolate_n_octree_levels(interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[-1]

    sfsp = last_octree_level.output_corners.scalar_field_at_sp
    xyz_on_edge, valid_edges = find_intersection_on_edge(last_octree_level.grid_corners.values, last_octree_level.output_corners.exported_fields.scalar_field, sfsp, )
    valid_voxels = valid_edges.sum(axis=1, dtype=bool)

    # endregion

    # region Get Normals

    interpolation_input.grid = Grid(xyz_on_edge)
    output_on_edges = interpolate_and_segment(interpolation_input, options, data_shape.tensors_structure)

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

    BIAS_STRENGTH = 1

    mass_points = last_octree_level.grid_centers.values

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

    # endregion

    # endregion

    # region triangulate
    grid_centers = last_octree_level.grid_centers

    temp_ids = octree_list[-1].output_centers.ids_block  # ! I need this because setters in python sucks
    temp_ids[valid_voxels] = 5
    octree_list[-1].output_centers.ids_block = temp_ids  # paint valid voxels

    dc_data = DualContouringData(
        xyz_on_edge=xyz_on_edge,
        xyz_on_centers=grid_centers.values,
        dxdydz=grid_centers.dxdydz,
        valid_edges=valid_edges,
        exported_fields_on_edges=None,
        n_surfaces=data_shape.tensors_structure.n_surfaces
    )

    indices = triangulate_dual_contouring(dc_data)
    # endregion

    if plot_pyvista or False:
        center_mass = xyz[:, 12:].reshape(-1, 3)
        normals = normals[:, 12:].reshape(-1, 3)
        _plot_pyvista(last_octree_level, octree_list, simple_model, ids, grid_0_centers,
                      xyz_on_edge, gradients, a=center_mass, b=normals,
                      v_pro=v_pro, indices=indices, plot_marching_cubes=True
                      )

    return xyz_on_edge, gradients


# =======================


def _plot_pyvista(last_octree_level, octree_list, simple_model, ids, grid_0_centers,
                  xyz_on_edge, gradients, a=None, b=None, v_mesh=None, v_pro=None, indices=None,
                  plot_label=False, plot_marching_cubes=True
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
    regular_grid_scalar = get_regular_grid_value_for_level(octree_list, n)

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
    surface_points = simple_model[0]
    orientations = simple_model[1]
    options = simple_model[2]
    shape: InputDataDescriptor = simple_model[3]

    interpolation_input = InterpolationInput(surface_points, orientations, grid, ids)

    from gempy_engine.core.data.grid import Grid, RegularGrid

    # region interpolate high res grid
    grid_high_res = Grid.from_regular_grid(RegularGrid([0.25, .75, 0.25, .75, 0.25, .75], resolution))
    interpolation_input.grid = grid_high_res
    input1: SolverInput = input_preprocess(shape.tensors_structure, interpolation_input)
    exported_fields_high_res = _evaluate_sys_eq(input1, weights, options)

    exported_fields_high_res.set_structure_values(
        reference_sp_position=shape.tensors_structure.reference_sp_position,
        slice_feature=interpolation_input.slice_feature,
        grid_size=interpolation_input.grid.len_all_grids)

    res = activate_formation_block(exported_fields_high_res, ids, sigmoid_slope=50000)
    result = res, exported_fields_high_res, grid_high_res.dxdydz
    values_block_high_res, scalar_high_res, dxdydz = result
    # endregion

    from skimage.measure import marching_cubes
    import pyvista as pv
    vert, edges, _, _ = marching_cubes(scalar_high_res.scalar_field.reshape(resolution),
                                       scalar_at_surface_points[0],
                                       spacing=dxdydz)
    loc_0 = np.array([0.25, .25, .25]) + np.array(dxdydz) / 2
    vert += np.array(loc_0).reshape(1, 3)
    mesh = pv.PolyData(vert, np.insert(edges, 0, 3, axis=1).ravel())
    return mesh
