from typing import List
import numpy as np

from gempy_engine.core.data.exported_structs import OctreeLevel
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_for_level
from gempy_engine.core.data.exported_structs import DualContouringMesh
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

