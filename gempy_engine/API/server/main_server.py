import json

import fastapi
import numpy as np
from fastapi.openapi.models import Response
from pydantic import BaseModel
from starlette.responses import FileResponse, StreamingResponse

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import InterpolationOptions, SurfacePoints, Orientations
from gempy_engine.core.data.grid import Grid, RegularGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor, TensorsStructure, StacksStructure, StackRelationType
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.kernel_classes.server.input_parser import GemPyInput
from ...core.data.solutions import Solutions

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
    from test.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_pyvista
except ImportError:
    plot_pyvista = False

from typing import Union

from fastapi import FastAPI

app = FastAPI()
# Start the server: uvicorn gempy_engine.API.server.main_server:app


# region InterpolationInput
surface_points: SurfacePoints = SurfacePoints(
    sp_coords=np.array([
        [0.25010, 0.50010, 0.37510],
        [0.50010, 0.50010, 0.37510],
        [0.66677, 0.50010, 0.41677],
        [0.70843, 0.50010, 0.47510],
        [0.75010, 0.50010, 0.54177],
        [0.58343, 0.50010, 0.39177],
        [0.73343, 0.50010, 0.50010],
    ]))

orientations: Orientations = Orientations(
    dip_positions=np.array([
        [0.25010, 0.50010, 0.54177],
        [0.66677, 0.50010, 0.62510],
    ]),
    dip_gradients=np.array([[0, 0, 1],
                            [-.6, 0, .8]])
)

regular_grid = RegularGrid(
    #extent=[0.25, .75, 0.25, .75, 0.25, .75],
    extent=[0, 10,0,10,0,10],
    regular_grid_shape=[2, 2, 3]
)
default_grid: Grid = Grid.from_regular_grid(regular_grid)

default_interpolation_input: InterpolationInput = InterpolationInput(
    surface_points=surface_points,
    orientations=orientations,
    grid=default_grid
)

# endregion

# region InterpolationOptions

default_interpolation_options: InterpolationOptions = InterpolationOptions(
    range=40.166666666667,  # TODO: have constructor from RegularGrid
    c_o=1.1428571429,  # TODO: This should be a property
    number_octree_levels=4,
    kernel_function=AvailableKernelFunctions.cubic,
    dual_contouring=True
)

# endregion

# region InputDataDescriptor
tensor_struct: TensorsStructure = TensorsStructure(
    number_of_points_per_surface=np.array([7])
)

stack_structure: StacksStructure = StacksStructure(
    number_of_points_per_stack=np.array([7]),
    number_of_orientations_per_stack=np.array([2]),
    number_of_surfaces_per_stack=np.array([1]),
    masking_descriptor=[StackRelationType.ERODE]
)

default_input_data_descriptor: InputDataDescriptor = InputDataDescriptor(
    tensors_structure=tensor_struct,
    stack_structure=stack_structure
)
# endregion


@app.post("/")
def compute_gempy_model(input_json: GemPyInput):
    import subsurface
    import pandas as pd
    print("Running GemPy Engine")
    
    input_json.interpolation_input.grid = default_grid # ! Hack inject default grid:
    
    interpolation_input: InterpolationInput = InterpolationInput.from_schema(input_json.interpolation_input)
    input_data_descriptor: InputDataDescriptor = InputDataDescriptor.from_schema(input_json.input_data_descriptor)

    solutions: Solutions = _compute_model(interpolation_input, default_interpolation_options, input_data_descriptor)

    meshes: list[DualContouringMesh] = solutions.dc_meshes
    n_meshes = len(meshes)
    print(f"Number of meshes: {n_meshes}")
    
    vertex_array = np.concatenate([meshes[i].vertices for i in range(n_meshes)])
    simplex_array = np.concatenate([meshes[i].edges for i in range(n_meshes)])
    ids_array = np.ones(simplex_array.shape[0])
    l0 = 0
    id = 1
    
    for mesh in meshes:
        l1 = l0 + mesh.edges.shape[0]
        ids_array[l0:l1] = id
        l0 = l1
        id += 1
    
    print("ids_array", ids_array)
    
    unstructured_data = subsurface.UnstructuredData.from_array(
        vertex=vertex_array,
        cells=simplex_array,
        cells_attr=pd.DataFrame(ids_array, columns=['id'])  # TODO: We have to create an array with the shape of simplex array with the id of each simplex
    )

    body, header = unstructured_data.to_binary()
    
    # encode json header and insert it into the binary body
    header_json = json.dumps(header)
    header_json_bytes = header_json.encode('utf-8')
    header_json_length = len(header_json_bytes)
    header_json_length_bytes = header_json_length.to_bytes(4, byteorder='little')
    body = header_json_length_bytes + header_json_bytes + body
        
    response = fastapi.Response(content=body, media_type='application/octet-stream')
    return response


# noinspection DuplicatedCode
def _compute_model(interpolation_input: InterpolationInput, options: InterpolationOptions, structure: InputDataDescriptor) \
        -> Solutions:
    n_oct_levels = options.number_octree_levels
    solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or False:
        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
        plot_dc_meshes(p, solutions.dc_meshes[0])
        p.show()

    return solutions
