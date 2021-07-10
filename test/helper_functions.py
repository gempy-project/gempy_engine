from typing import List
import numpy as np

from gempy_engine.core.data.exported_structs import OctreeLevel, DualContouringMesh
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.modules.octrees_topology.octrees_topology_interface import \
    get_regular_grid_for_level

import matplotlib.pyplot as plt

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
except ImportError:
    plot_pyvista = False


def plot_octree_pyvista(p: pv.Plotter, octree_list: List[OctreeLevel], n_octree: int):

    n = n_octree

    shape = octree_list[n].grid_centers.regular_grid_shape
    regular_grid_values = octree_list[n].grid_centers.regular_grid.values_vtk_format
    regular_grid_scalar = get_regular_grid_for_level(octree_list, n)

    grid_3d = regular_grid_values.reshape(*(shape + 1), 3).T
    regular_grid_mesh = pv.StructuredGrid(*grid_3d)
    regular_grid_mesh["lith"] = regular_grid_scalar.ravel()
    foo = regular_grid_mesh.threshold([0, 10])

    p.add_mesh(foo, show_edges=False, opacity=.5, cmap="tab10")


    p.add_axes()


def plot_dc_meshes(p: pv.Plotter, dc_mesh: DualContouringMesh, plot_labels=False):

    vtk_edges = np.insert(dc_mesh.edges.reshape(-1, 3), 0, 3, axis=1).ravel()
    dual_mesh = pv.PolyData(dc_mesh.vertices, vtk_edges)
    p.add_mesh(dual_mesh, opacity=1, silhouette=False, color="green")

    vertex = pv.PolyData(dc_mesh.vertices)
    p.add_mesh(vertex, color="w", point_size=5.0, render_points_as_spheres=True)
    if plot_labels:
        p.add_point_labels(vertex, list(range(dc_mesh.vertices.shape[0])), point_size=20, font_size=36)


def plot_points(p: pv.Plotter, xyz: np.ndarray, plot_labels=False):
    coords = pv.PolyData(xyz)
    p.add_mesh(coords, color="w", point_size=10.0, render_points_as_spheres=True)
    if plot_labels:
        p.add_point_labels(coords, list(range(xyz.shape[0])), point_size=20,
                           font_size=36)


def plot_vector(p: pv.Plotter, xyz, gradients):
    poly = pv.PolyData(xyz)
    poly['vectors'] = gradients

    arrows = poly.glyph(orient='vectors', scale=False, factor=.05)

    p.add_mesh(arrows, color="green", point_size=10.0, render_points_as_spheres=False)

def plot_2d_scalar_y_direction(interpolation_input: InterpolationInput, Z_x):

    resolution = interpolation_input.grid.regular_grid.resolution
    extent = interpolation_input.grid.regular_grid.extent

    plt.contourf(Z_x.reshape(resolution)[:, resolution[1]//2, :].T, N=40, cmap="autumn",
                 extent=extent[[0,1,4,5]]
                 )

    xyz = interpolation_input.surface_points.sp_coords
    plt.plot(xyz[:, 0], xyz[:, 2], "o")
    plt.colorbar()

    plt.quiver(interpolation_input.orientations.dip_positions[:, 0],
               interpolation_input.orientations.dip_positions[:, 2],
               interpolation_input.orientations.dip_gradients[:, 0],
               interpolation_input.orientations.dip_gradients[:, 2],
               scale=10
               )

    # plt.quiver(
    #      gx.reshape(50, 5, 50)[:, 2, :].T,
    #      gz.reshape(50, 5, 50)[:, 2, :].T,
    #      scale=1
    #  )

    plt.savefig("foo")
    plt.show()



def calculate_gradient(dip, az, pol):
    """Calculates the gradient from dip, azimuth and polarity values."""
    g_x = np.sin(np.deg2rad(dip)) * np.sin(np.deg2rad(az)) * pol
    g_y = np.sin(np.deg2rad(dip)) * np.cos(np.deg2rad(az)) * pol
    g_z = np.cos(np.deg2rad(dip)) * pol
    return g_x, g_y, g_z


