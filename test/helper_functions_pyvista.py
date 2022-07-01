from typing import List
import numpy as np

from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level
from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
except ImportError:
    plot_pyvista = False


def plot_octree_pyvista(p: pv.Plotter, octree_list: List[OctreeLevel], n_octree: int):
    n = n_octree

    shape = octree_list[n].grid_centers.regular_grid_shape
    regular_grid_values = octree_list[n].grid_centers.regular_grid.values_vtk_format
    regular_grid_scalar = get_regular_grid_value_for_level(octree_list, n)

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


def plot_pyvista(octree_list=None, dc_meshes: List[DualContouringMesh] = None, vertices=None, indices=None,
                 xyz_on_edge=None, gradients=None, a=None, b=None, v_just_points=None,
                 plot_label=False, delaunay_3d=False):
    n = len(octree_list) - 1
    p = pv.Plotter()

    # Plot Regular grid Octree
    if octree_list is not None:
        regular_grid_values = octree_list[n].grid_centers.regular_grid.values_vtk_format
        regular_grid_scalar = get_regular_grid_value_for_level(octree_list, n)
        print("regular_grid_scalar.shape", regular_grid_scalar.shape)
        
        shape = octree_list[n].grid_centers.regular_grid_shape
        grid_3d = regular_grid_values.reshape(*(shape + 1), 3).T
        regular_grid_mesh = pv.StructuredGrid(*grid_3d)
        regular_grid_mesh["lith"] = regular_grid_scalar.ravel()
        foo = regular_grid_mesh.threshold([0, 10])

        p.add_mesh(foo, show_edges=False, opacity=.2, cmap="tab10")

    # Plot gradients
    if gradients is not None and xyz_on_edge is not None:
        poly = pv.PolyData(xyz_on_edge)
        poly['vectors'] = gradients
        arrows = poly.glyph(orient='vectors', scale=False, factor=.3)
        p.add_mesh(arrows, color="k", point_size=10.0, render_points_as_spheres=False)

    if plot_label:
        p.add_point_labels(xyz_on_edge, list(range(xyz_on_edge.shape[0])), point_size=20, font_size=36)

    if a is not None and b is not None:
        poly = pv.PolyData(a)
        poly['vectors'] = b

        arrows = poly.glyph(orient='vectors', scale=False, factor=.1)

        p.add_mesh(arrows, color="gray", point_size=10.0, render_points_as_spheres=False)

    # Plot QEF
    if v_just_points is not None:
        p.add_mesh(pv.PolyData(v_just_points), color="b", point_size=5.0, render_points_as_spheres=False)

    if vertices is not None:
        data = pv.PolyData(vertices)
        if delaunay_3d:
            triangulated = data.delaunay_3d(alpha=0.5)
            p.add_mesh(triangulated, color="w", point_size=4.0, render_points_as_spheres=True, show_edges=True)
        else:
            p.add_mesh(data, color="w", point_size=4.0, render_points_as_spheres=True)

    if indices is not None:
        dual_mesh = pv.PolyData(vertices, np.insert(indices, 0, 3, axis=1).ravel())
        p.add_mesh(dual_mesh, opacity=1, silhouette=True, color="green", show_edges=True)

    colors = ['green', 'blue', 'red', 'yellow', 'pink', 'brown', 'purple']
    if dc_meshes is not None:
        for e, mesh in enumerate(dc_meshes):
            vertices = mesh.vertices
            indices = mesh.edges

            if vertices.shape[0] == 0: continue

            dual_mesh = pv.PolyData(vertices, np.insert(indices, 0, 3, axis=1).ravel())
            p.add_mesh(dual_mesh, opacity=1, silhouette=True, color=colors[e], show_edges=True)

            p.add_mesh(pv.PolyData(vertices), color=colors[e], point_size=1.0, render_points_as_spheres=True)

    p.add_axes()
    p.show()
