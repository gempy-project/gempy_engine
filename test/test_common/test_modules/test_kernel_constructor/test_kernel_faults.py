import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.core.data.interp_output import InterpOutput
from test import helper_functions_pyvista
from test.conftest import TEST_SPEED

PLOT = False


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_fault_kernel(unconformity_complex, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    solutions: Solutions = compute_model(interpolation_input, options, structure)

    # TODO: Grab second scalar and create fault kernel
    output: InterpOutput = solutions.octrees_output[0].outputs_centers[1]
    output.values_block
    
    
    if PLOT or True:
        helper_functions_pyvista.plot_pyvista(solutions.octrees_output, dc_meshes=solutions.dc_meshes)
