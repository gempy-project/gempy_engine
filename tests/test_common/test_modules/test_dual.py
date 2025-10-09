import os

import numpy as np

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level
from gempy_engine.plugins.plotting import helper_functions_pyvista
from tests.conftest import plot_pyvista

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
