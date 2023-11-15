import fastapi

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from ._process_output import process_output
from ._server_functions import process_input, setup_logger
from ...core.data.kernel_classes.server.input_parser import GemPyInput
from ...core.data.options import MeshExtractionMaskingOptions
from ...core.data.solutions import Solutions

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
    from test.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_pyvista
except ImportError:
    plot_pyvista = False

try:
    from fastapi import FastAPI
except ImportError:
    print("FastAPI is not installed. No server capabilities will be available.")
    
try:
    from pydantic import BaseModel, Field
except ImportError:
    print("Pydantic is not installed. No server capabilities will be available.")
    BaseModel = object

app = FastAPI()
# Start the server: uvicorn gempy_engine.API.server.main_server:app


# region InterpolationOptions

range_ = 1
default_interpolation_options: InterpolationOptions = InterpolationOptions(
    range= range_,  # TODO: have constructor from RegularGrid
    c_o=( range_ ** 2 ) / 14 / 3,
    number_octree_levels=4,
    kernel_function=AvailableKernelFunctions.cubic,
    mesh_extraction=True
)


# endregion

logger = setup_logger()

# BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=False, pykeops_enabled=True)

@app.post("/")
def compute_gempy_model(gempy_input: GemPyInput):
    logger.info("Running GemPy Engine")

    input_data_descriptor, interpolation_input, n_stack = process_input(
        gempy_input=gempy_input,
        logger=logger
    )

    # region Set new fancy triangulation TODO: This has to be move to the interpolation options coming from the client
    FANCY_TRIANGULATION = True
    if FANCY_TRIANGULATION:
        default_interpolation_options.mesh_extraction_fancy = True
        # default_interpolation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.RAW  # * To Date only raw making is supported
    # endregion

    default_interpolation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.RAW  # * To Date only raw making is supported
    
    solutions: Solutions = _compute_model(
        interpolation_input=interpolation_input,
        options=default_interpolation_options,
        structure=input_data_descriptor
    )

    logger.info("Finished computing model")
    # logger.debug("first mesh vertices:", solutions.dc_meshes[0].vertices)

    body = process_output(
        meshes=solutions.dc_meshes,
        n_stack=n_stack,
        solutions=solutions,
        logger=logger
    )

    logger.info("Finished running GemPy Engine")
    response = fastapi.Response(content=body, media_type='application/octet-stream')
    return response


# noinspection DuplicatedCode
def _compute_model(interpolation_input: InterpolationInput, options: InterpolationOptions,
                   structure: InputDataDescriptor) \
        -> Solutions:
    n_oct_levels = options.number_octree_levels
    solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista and False:
        #pv.global_theme.show_edges = True

        from test import helper_functions_pyvista
        helper_functions_pyvista.plot_pyvista(
            octree_list=None,
            dc_meshes=solutions.dc_meshes,
            gradients=interpolation_input.orientations.dip_gradients,
            gradient_pos=interpolation_input.orientations.dip_gradients,
            v_just_points=interpolation_input.surface_points.sp_coords
        )
    return solutions
