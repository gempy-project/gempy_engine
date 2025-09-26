import numpy as np
import os
import pytest
from typing import List

import gempy_engine.API.interp_single.interp_features as interp
from gempy_engine.modules.dual_contouring._dual_contouring import compute_dual_contouring
from gempy_engine.API.interp_single._multi_scalar_field_manager import interpolate_all_fields
from gempy_engine.API.interp_single.interp_features import interpolate_n_octree_levels, interpolate_and_segment
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.dual_contouring_data import DualContouringData
from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.core.data.engine_grid import EngineGrid, RegularGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.modules.dual_contouring.dual_contouring_interface import QEF, find_intersection_on_edge
from gempy_engine.modules.dual_contouring._triangulate import triangulate_dual_contouring
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level
from gempy_engine.plugins.plotting import helper_functions_pyvista
from tests.conftest import TEST_SPEED, plot_pyvista

dir_name = os.path.dirname(__file__)

try:
    # noinspection PyPackageRequirements
    import pyvista as pv
except ImportError:
    plot_pyvista = False


def _grab_xyz_edges(last_octree_level: OctreeLevel) -> tuple:
    corners = last_octree_level.outputs_centers[0]
    # First find xyz on edges:
    xyz, edges = find_intersection_on_edge(
        _xyz_corners=last_octree_level.grid_centers.corners_grid.values,
        scalar_field_on_corners=corners.exported_fields.scalar_field[corners.grid.corners_grid_slice],
        scalar_at_sp=corners.scalar_field_at_sp,
        masking=None
    )
    return xyz, edges


def test_compute_dual_contouring_api(simple_model, simple_grid_3d_octree):
    # region Test find_intersection_on_edge
    spi, ori_i, options, data_shape = simple_model
    options.compute_scalar_gradient = True
    options.evaluation_options.number_octree_levels = 2

    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = interpolate_n_octree_levels(interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[-1]

    intersection_xyz, valid_edges = _grab_xyz_edges(last_octree_level)
    dc_data = _gen_dc_data(
        octree_level=last_octree_level,
        interpolation_input=interpolation_input,
        data_shape=data_shape,
        options=options,
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
                                              gradient_pos=intersection_xyz,
                                              v_just_points=vertices, vertices=intersection_points)
    # endregion


def _gen_dc_data(octree_level: OctreeLevel, interpolation_input: InterpolationInput,
                 options: InterpolationOptions, data_shape: InputDataDescriptor) -> DualContouringData:
    intersection_xyz, valid_edges = _grab_xyz_edges(octree_level)

    interpolation_input.set_temp_grid(EngineGrid.from_xyz_coords(intersection_xyz))

    output_on_edges: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, data_shape)

    dc_data = DualContouringData(
        xyz_on_edge=intersection_xyz,
        valid_edges=valid_edges,
        xyz_on_centers=octree_level.grid_centers.octree_grid.values,
        dxdydz=octree_level.grid_centers.octree_dxdydz,
        exported_fields_on_edges=output_on_edges[0].exported_fields,
        n_surfaces_to_export=data_shape.tensors_structure.n_surfaces
    )
    return dc_data


@pytest.mark.skipif(BackendTensor.engine_backend != AvailableBackends.numpy, reason="Only numpy supported")
def test_compute_mesh_extraction_fancy_triangulation(simple_model, simple_grid_3d_octree):
    from gempy_engine.modules.dual_contouring.fancy_triangulation import get_left_right_array, triangulate

    def _simple_grid_3d_octree_regular():
        import dataclasses
        resolution = [2, 2, 2]
        extent = [0.25, .75, 0.25, .75, 0.25, .75]

        regular_grid = RegularGrid(extent, resolution)
        grid = EngineGrid.from_regular_grid(regular_grid)
        return dataclasses.replace(grid)

    # region Test find_intersection_on_edge
    spi, ori_i, options, data_shape = simple_model

    options.number_octree_levels = 5
    options.number_octree_levels_surface = 5
    options.compute_scalar_gradient = True

    ids = np.array([1, 2])
    grid_0_centers = _simple_grid_3d_octree_regular()
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = interpolate_n_octree_levels(interpolation_input, options, data_shape)

    octree_level_for_surface: OctreeLevel = octree_list[options.number_octree_levels_surface - 1]

    # corners = octree_level_for_surface.outputs_corners[0]
    # # First find xyz on edges:
    # xyz, edges = find_intersection_on_edge(
    #     _xyz_corners=octree_level_for_surface.grid_corners.values,
    #     scalar_field_on_corners=corners.exported_fields.scalar_field,
    #     scalar_at_sp=corners.scalar_field_at_sp,
    #     masking=None
    # )
    # result = xyz, edges
    # intersection_xyz, valid_edges = result
    # 
    # last_octree_level: OctreeLevel = octree_list[-1]
    # intersection_xyz, valid_edges = _grab_xyz_edges(octree_level_for_surface)
    # interpolation_input.set_temp_grid(EngineGrid.from_xyz_coords(intersection_xyz))
    # 
    # output_on_edges: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, data_shape)
    # 
    # dc_data = DualContouringData(
    #     xyz_on_edge=intersection_xyz,
    #     valid_edges=valid_edges,
    #     xyz_on_centers=octree_level_for_surface.grid_centers.values,
    #     dxdydz=octree_level_for_surface.grid_centers.octree_dxdydz,
    #     exported_fields_on_edges=output_on_edges[0].exported_fields,
    #     n_surfaces_to_export=data_shape.tensors_structure.n_surfaces
    # )

    # dc_data = _gen_dc_data(data_shape, interpolation_input, intersection_xyz, last_octree_level, options, valid_edges)

    dc_data = _gen_dc_data(
        octree_level=octree_level_for_surface,
        interpolation_input=interpolation_input,
        data_shape=data_shape,
        options=options,
    )
    
    intersection_xyz, valid_edges = _grab_xyz_edges(octree_level_for_surface)
    
    dc_meshes: List[DualContouringMesh] = compute_dual_contouring(dc_data)
    dc_data = dc_meshes[0].dc_data
    valid_voxels = dc_data.valid_voxels

    # Mark active voxels
    temp_ids = octree_level_for_surface.last_output_center.ids_block  # ! I need this because setters in python sucks
    temp_ids[valid_voxels] = 5
    octree_level_for_surface.last_output_center.ids_block = temp_ids  # paint valid voxels

    # * ---- New code Here ----

    stacked = get_left_right_array(octree_list)
    validated_stacked = stacked[valid_voxels]
    validated_edges = valid_edges[valid_voxels]

    edges_normals = np.zeros((valid_edges.shape[0], 12, 3))
    edges_normals[:] = np.nan
    edges_normals[valid_edges] = dc_data.gradients

    voxel_normal = np.nanmean(edges_normals, axis=1)
    voxel_normal = voxel_normal[(~np.isnan(voxel_normal).any(axis=1))]  # drop nans

    indices_array: np.ndarray = triangulate(
        left_right_array=validated_stacked,
        valid_edges=validated_edges,
        tree_depth=options.number_octree_levels,
        voxel_normals=voxel_normal
    )

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
            gradient_pos=intersection_xyz,
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

    options.evaluation_options.number_octree_levels = n_oct_levels
    options.evaluation_options.number_octree_levels_surface = n_oct_levels

    solutions: Solutions = compute_model(interpolation_input, options, structure)
    dc_data = solutions.dc_meshes[0].dc_data

    if plot_pyvista or False:
        output_corners: InterpOutput = solutions.octrees_output[-1].outputs_centers[-1]
        vertices = output_corners.grid.values

        intersection_xyz = dc_data.xyz_on_edge
        gradients = dc_data.gradients

        center_mass = dc_data.bias_center_mass
        normals = dc_data.bias_normals
        helper_functions_pyvista.plot_pyvista(
            octree_list=solutions.octrees_output,
            dc_meshes=solutions.dc_meshes,
            gradient_pos=intersection_xyz,
            gradients=gradients,
            a=center_mass,
            b=normals,
            # v_just_points=vertices
        )


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_compute_dual_contouring_several_meshes(simple_model_3_layers, simple_grid_3d_octree):
    # region Test find_intersection_on_edge
    interpolation_input, options, data_shape = simple_model_3_layers
    interpolation_input.surface_points.__post_init__()  # ! This is a weird hack to be able to run all the tests at the same time. I have no idea why the change of backend does not work with this specific case
    options.compute_scalar_gradient = True

    ids = np.array([1, 2, 3, 4])
    grid_0_centers = simple_grid_3d_octree

    octree_list = interpolate_n_octree_levels(interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[-1]

    corners = last_octree_level.outputs_centers[0]
    # First find xyz on edges:
    xyz, edges = find_intersection_on_edge(
        _xyz_corners=last_octree_level.grid_centers.corners_grid.values,
        scalar_field_on_corners=corners.exported_fields.scalar_field[corners.grid.corners_grid_slice],
        scalar_at_sp=corners.scalar_field_at_sp,
        masking=None
    )
    result = xyz, edges
    intersection_xyz, valid_edges = result
    interpolation_input.set_temp_grid(EngineGrid.from_xyz_coords(intersection_xyz))

    output_on_edges = interp.interpolate_single_field(interpolation_input, options, data_shape.tensors_structure)

    dc_data = DualContouringData(
        xyz_on_edge=intersection_xyz,
        valid_edges=valid_edges,
        xyz_on_centers=last_octree_level.grid_centers.octree_grid.values,
        dxdydz=last_octree_level.grid_centers.octree_dxdydz,
        exported_fields_on_edges=output_on_edges.exported_fields,
        n_surfaces_to_export=data_shape.tensors_structure.n_surfaces
    )

    meshes: List[DualContouringMesh] = compute_dual_contouring(dc_data)

    if plot_pyvista or False:
        _plot_pyvista(
            last_octree_level=last_octree_level,
            octree_list=octree_list,
            simple_model=simple_model_3_layers,
            ids=ids,
            grid_0_centers=grid_0_centers,
            xyz_on_edge=dc_data.xyz_on_edge,
            gradients=dc_data.gradients,
            v_pro=meshes[0].vertices,
            indices=meshes[0].edges,
            plot_label=False,
            plot_marching_cubes=False,
            n=0
        )


@pytest.mark.skipif(BackendTensor.engine_backend != AvailableBackends.numpy, reason="Only numpy supported")
def test_find_edges_intersection_pro(simple_model, simple_grid_3d_octree):
    # region Test find_intersection_on_edge
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    options.number_octree_levels = 5
    options.number_octree_levels_surface = 5
    options.compute_scalar_gradient = True

    octree_list = interpolate_n_octree_levels(interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[-1]

    sfsp = last_octree_level.last_output_center.scalar_field_at_sp

    xyz_on_edge, valid_edges = find_intersection_on_edge(last_octree_level.grid_centers.corners_grid.values, last_octree_level.last_output_center.exported_fields.scalar_field[last_octree_level.last_output_center.grid.corners_grid_slice], sfsp, )

    # endregion

    # region Get Normals

    interpolation_input.set_temp_grid(EngineGrid.from_xyz_coords(xyz_on_edge))

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




@pytest.mark.skipif(BackendTensor.engine_backend != AvailableBackends.numpy, reason="Only numpy supported")
def test_find_edges_intersection_bias_on_center_of_the_cell(simple_model, simple_grid_3d_octree):
    """This looks works that taking the center of gravity of the edges intersections. I leave this test here as 
    documentation"""

    # region Test find_intersection_on_edge
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    options.compute_scalar_gradient = True
    octree_list = interpolate_n_octree_levels(interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[-1]

    sfsp = last_octree_level.last_output_center.scalar_field_at_sp
    xyz_on_edge, valid_edges = find_intersection_on_edge(last_octree_level.grid_centers.corners_grid.values, last_octree_level.last_output_center.exported_fields.scalar_field[last_octree_level.last_output_center.grid.corners_grid_slice], sfsp, )
    valid_voxels = valid_edges.sum(axis=1, dtype=bool)

    # endregion

    # region Get Normals

    interpolation_input.set_temp_grid(EngineGrid.from_xyz_coords(xyz_on_edge))
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

    mass_points = last_octree_level.grid_centers.octree_grid.values

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

    temp_ids = octree_list[-1].last_output_center.ids_block  # ! I need this because setters in python sucks
    temp_ids[valid_voxels] = 5
    octree_list[-1].last_output_center.ids_block = temp_ids  # paint valid voxels

    dc_data = DualContouringData(
        xyz_on_edge=xyz_on_edge,
        xyz_on_centers=grid_centers.octree_grid.values,
        dxdydz=grid_centers.octree_dxdydz,
        valid_edges=valid_edges,
        exported_fields_on_edges=None,
        n_surfaces_to_export=data_shape.tensors_structure.n_surfaces
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


# * ======================= Private functions =======================


def _plot_pyvista(last_octree_level, octree_list, simple_model, ids, grid_0_centers,
                  xyz_on_edge, gradients, a=None, b=None, v_mesh=None, v_pro=None, indices=None,
                  plot_label=False, plot_marching_cubes=True, n=1
                  ):
    p = pv.Plotter()

    # Plot Regular grid Octree
    regular_grid_values = octree_list[n].grid_centers.octree_grid.values_vtk_format
    regular_grid_scalar = get_regular_grid_value_for_level(octree_list, n)

    shape = octree_list[n].grid_centers.octree_grid_shape
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

