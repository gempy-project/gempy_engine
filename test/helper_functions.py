from typing import List
import numpy as np

from gempy_engine.core.data.exported_structs import OctreeLevel, DualContouringMesh
from gempy_engine.modules.octrees_topology.octrees_topology_interface import \
    get_regular_grid_for_level

from .conftest import plot_pyvista

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
except ImportError:
    plot_pyvista = False


def plot_octree_pyvista(p: pv.Plotter, octree_list: List[OctreeLevel], n_octree: int):
    if plot_pyvista is False: return

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


def plot_dc_meshes(p: pv.Plotter, dc_mesh: DualContouringMesh):

    vtk_edges = np.insert(dc_mesh.edges, 0, 3, axis=1).ravel()
    dual_mesh = pv.PolyData(dc_mesh.vertices, vtk_edges)
    p.add_mesh(dual_mesh, opacity=1, silhouette=True, color="green")

