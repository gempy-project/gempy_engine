from gempy_engine import compute_model
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.options import MeshExtractionMaskingOptions
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.plugins.plotting import helper_functions_pyvista
from tests.conftest import plot_pyvista


def test_dual_contouring_on_fault_model(one_fault_model, n_oct_levels=5):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model

    import numpy as np
    interpolation_input.surface_points.sp_coords[:, 2] += np.random.uniform(-0.02, 0.02, interpolation_input.surface_points.sp_coords[:, 2].shape)
    options.compute_scalar_gradient = False
    options.evaluation_options.dual_contouring = True
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
    options.evaluation_options.mesh_extraction_fancy = True

    options.evaluation_options.number_octree_levels = n_oct_levels
    options.evaluation_options.number_octree_levels_surface = n_oct_levels

    solutions: Solutions = compute_model(interpolation_input, options, structure)

    outputs: list[OctreeLevel] = solutions.octrees_output

    if plot_pyvista or False:
        helper_functions_pyvista.plot_pyvista(
            # octree_list=solutions.octrees_output,
            octree_list=None,
            dc_meshes=solutions.dc_meshes
        )
