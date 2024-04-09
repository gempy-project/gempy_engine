import numpy as np
import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from ...conftest import plot_pyvista, REQUIREMENT_LEVEL, Requirements

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
    from gempy_engine.plugins.plotting.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_points, plot_vector
except ImportError:
    plot_pyvista = False

pytestmark = pytest.mark.skipif(REQUIREMENT_LEVEL.value < Requirements.OPTIONAL.value, reason="This test needs higher requirements.")

def test_output_to_subsurface(simple_model_interpolation_input, n_oct_levels=3):
    import subsurface
    interpolation_input, options, structure = simple_model_interpolation_input
    print(interpolation_input)

    options.number_octree_levels = n_oct_levels
    solutions = compute_model(interpolation_input, options, structure)

    meshes: list[DualContouringMesh] = solutions.dc_meshes
    vertex_array = np.concatenate([meshes[i].vertices for i in range(len(meshes))])
    simplex_array = np.concatenate([meshes[i].edges for i in range(len(meshes))])

    unstructured_data = subsurface.UnstructuredData.from_array(
        vertex=vertex_array,
        cells=simplex_array,
        #       cells_attr=pd.DataFrame(ids_array, columns=['id']) # TODO: We have to create an array with the shape of simplex array with the id of each simplex
    )

    if plot_pyvista or False:  # Plot using subsurface
        trisurf = subsurface.TriSurf(unstructured_data)
        s = subsurface.visualization.to_pyvista_mesh(trisurf)
        subsurface.visualization.pv_plot([s], image_2d=False)
