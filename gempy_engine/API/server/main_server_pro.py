import fastapi
from fastapi import FastAPI
from fastapi.responses import Response

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.kernel_classes.server.input_parser import GemPyInput
from gempy_engine.core.data.solutions import Solutions
from ._process_output import process_output
from ._server_functions import process_input, setup_logger

# Optional visualization dependencies
try:
    import pyvista as pv
    from test.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_pyvista

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Create FastAPI application
gempy_engine_App = FastAPI(debug=True)
logger = setup_logger()

# Default interpolation options
range_ = 1
default_interpolation_options: InterpolationOptions = InterpolationOptions.from_args(
    range=range_,
    c_o=(range_ ** 2) / 14 / 3,
    number_octree_levels=4,
    kernel_function=AvailableKernelFunctions.cubic,
    mesh_extraction=True
)


@gempy_engine_App.post("/")
def compute_gempy_model(gempy_input: GemPyInput) -> Response:
    """
    Process GemPy input model and return serialized results.

    Args:
        gempy_input: Input data for the GemPy model

    Returns:
        Binary response containing serialized model output
    """
    logger.info("Running GemPy Engine")


    interpolation_input: InterpolationInput
    input_data_descriptor: InputDataDescriptor
    # Process input data
    input_data_descriptor, interpolation_input, n_stack = process_input(
        gempy_input=gempy_input,
        logger=logger
    )

    # Compute model
    solutions = _compute_model(
        interpolation_input=interpolation_input,
        options=default_interpolation_options,
        structure=input_data_descriptor
    )
    logger.info("Finished computing model")

    # Process output
    body = process_output(
        meshes=solutions.dc_meshes,
        n_stack=n_stack,
        solutions=solutions,
        logger=logger
    )

    logger.info("Finished running GemPy Engine")
    return fastapi.Response(content=body, media_type='application/octet-stream')


def _compute_model(
        interpolation_input: InterpolationInput,
        options: InterpolationOptions,
        structure: InputDataDescriptor
) -> Solutions:
    """
    Compute the GemPy model using the provided inputs.

    Args:
        interpolation_input: Input data for interpolation
        options: Interpolation options
        structure: Data descriptor for model structure

    Returns:
        Computed solutions
    """
    solutions = compute_model(interpolation_input, options, structure)

    # Optional visualization (disabled by default)
    if VISUALIZATION_AVAILABLE and False:
        from test import helper_functions_pyvista
        helper_functions_pyvista.plot_pyvista(
            octree_list=None,
            dc_meshes=solutions.dc_meshes,
            gradients=interpolation_input.orientations.dip_gradients,
            gradient_pos=interpolation_input.orientations.dip_gradients,
            v_just_points=interpolation_input.surface_points.sp_coords
        )

    return solutions