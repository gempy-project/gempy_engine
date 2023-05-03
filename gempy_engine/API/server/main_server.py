import json

import fastapi
import numpy as np
import pandas as pd
import subsurface.visualization
from fastapi.openapi.models import Response
from pydantic import BaseModel
from starlette.responses import FileResponse, StreamingResponse
from subsurface import UnstructuredData

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import InterpolationOptions, SurfacePoints, Orientations
from gempy_engine.core.data.grid import Grid, RegularGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor, TensorsStructure, StacksStructure, StackRelationType
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.kernel_classes.server.input_parser import GemPyInput
from ...core.data.options import DualContouringMaskingOptions
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
    extent=[0., .75, 0., .75, 0., .75],
    # extent=[0.25, .75, 0.25, .75, 0.25, .75],
    # extent=[0, 20, 0, 20, 0, 20],
    regular_grid_shape=[2, 2, 2]
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

    input_json.interpolation_input.grid = default_grid  # ! Hack inject default grid:

    interpolation_input: InterpolationInput = InterpolationInput.from_schema(input_json.interpolation_input)
    input_data_descriptor: InputDataDescriptor = InputDataDescriptor.from_schema(input_json.input_data_descriptor)
    n_stack = len(input_data_descriptor.stack_structure.masking_descriptor)

    print(input_data_descriptor.stack_structure.masking_descriptor)
    print(input_data_descriptor.stack_structure)

    # region Set new fancy triangulation
    FANCY_TRIANGULATION = True
    if FANCY_TRIANGULATION:
        default_interpolation_options.dual_contouring_fancy = True
        default_interpolation_options.dual_contouring_masking_options = DualContouringMaskingOptions.RAW  # * To Date only raw making is supported
    # endregion

    solutions: Solutions = _compute_model(interpolation_input, default_interpolation_options, input_data_descriptor)

    print(solutions.dc_meshes[0].vertices)

    meshes: list[DualContouringMesh] = solutions.dc_meshes

    if MESHES_TO_UNSTRUT := True:
        unstructured_data_meshes: UnstructuredData = meshes_to_unstruct(meshes, n_stack)
        if PLOT_SUBSURFACE_OBJECT := False:
            plot_subsurface_object(unstructured_data)

    # region encode octrees
    grid_centers_xyz = solutions.octrees_output[0].grid_centers.values
    grid_centers_val = solutions.octrees_output[0].last_output_center.ids_block

    unstructured_data_volume = subsurface.UnstructuredData.from_array(
        vertex=grid_centers_xyz,
        cells="points",
        cells_attr=pd.DataFrame(grid_centers_val, columns=['id'])  # TODO: We have to create an array with the shape of simplex array with the id of each simplex
    )
    # endregion

    # TODO: Add xarray_attrs to differentiate between the type of data
    body_meshes, header_meshes = unstructured_data_meshes.to_binary()

    # encode json header and insert it into the binary body
    header_json = json.dumps(header_meshes)
    header_json_bytes = header_json.encode('utf-8')
    header_json_length = len(header_json_bytes)
    header_json_length_bytes = header_json_length.to_bytes(4, byteorder='little')
    body_meshes = header_json_length_bytes + header_json_bytes + body_meshes

    # TODO: Serialize unstructured_data_volume and add Global Header for the size of each data
    body_volume, header_volume = unstructured_data_volume.to_binary()
    
    # encode json header and insert it into the binary body
    header_json = json.dumps(header_volume)
    header_json_bytes = header_json.encode('utf-8')
    header_json_length = len(header_json_bytes)
    header_json_length_bytes = header_json_length.to_bytes(4, byteorder='little')
    body_volume = header_json_length_bytes + header_json_bytes + body_volume
    
    body = body_meshes + body_volume
    
    global_header = json.dumps({"mesh_size": len(body_meshes), "octree_size": len(body_volume)})
    global_header_bytes = global_header.encode('utf-8')
    global_header_length = len(global_header_bytes)
    global_header_length_bytes = global_header_length.to_bytes(4, byteorder='little')
    body = global_header_length_bytes + global_header_bytes + body
    
    response = fastapi.Response(content=body, media_type='application/octet-stream')
    return response


def plot_subsurface_object(unstructured_data):
    print(unstructured_data)
    obj = subsurface.TriSurf(unstructured_data)
    pv_unstruct = subsurface.visualization.to_pyvista_mesh(obj)
    print(pv_unstruct)
    subsurface.visualization.pv_plot([pv_unstruct])


def meshes_to_unstruct(meshes: list[DualContouringMesh], n_stack) -> UnstructuredData:
    n_meshes = len(meshes)
    print(f"Number of meshes: {n_meshes}")
    # print("Mesh1Vert" + str(meshes[0].vertices))
    # print("Mesh2Vert" + str(meshes[1].vertices))
    # 
    # print("Mesh1Tri" + str(meshes[0].edges))
    # print("Mesh2Tri" + str(meshes[1].edges))
    print("Mesh1TriShape" + str(meshes[0].edges.shape))
    # print("Mesh2TriShape" + str(meshes[1].edges.shape))
    vertex_array = np.concatenate([meshes[i].vertices for i in range(n_meshes)])
    simplex_array = np.concatenate([meshes[i].edges for i in range(n_meshes)])
    unc, count = np.unique(simplex_array, axis=0, return_counts=True)
    print(f"edges shape {simplex_array.shape}")
    # print(f"UNC COUNT {unc[count > 1][0]}")
    if n_stack > 1:  # if unc[count > 1][0][0] == 0:
        simplex_array = meshes[0].edges
        for i in range(n_meshes):
            adder = 0
            meshes_are_contiguous = meshes[i].edges[0, 0] == meshes[0].edges[0, 0]  # * this is how Jan was doing it for the old triangulation
            meshes_are_contiguous = True  # *  For now I compute it always
            if i == 0:
                continue
            elif meshes_are_contiguous:
                adder = np.max(meshes[i - 1].edges) + 1
                print("adder" + str(adder))
                addmesh = meshes[i].edges + adder
                simplex_array = np.append(simplex_array, addmesh, axis=0)
    print(f"edges shape {simplex_array.shape}")
    ids_array = np.ones(simplex_array.shape[0])
    l0 = 0
    id = 1
    for mesh in meshes:
        l1 = l0 + mesh.edges.shape[0]
        print(f"l0 {l0} l1 {l1} id {id}")
        ids_array[l0:l1] = id
        l0 = l1
        id += 1
    print("ids_array count" + str(np.unique(ids_array)))
    unstructured_data = subsurface.UnstructuredData.from_array(
        vertex=vertex_array,
        cells=simplex_array,
        cells_attr=pd.DataFrame(ids_array, columns=['id'])  # TODO: We have to create an array with the shape of simplex array with the id of each simplex
    )
    return unstructured_data


# noinspection DuplicatedCode
def _compute_model(interpolation_input: InterpolationInput, options: InterpolationOptions, structure: InputDataDescriptor) \
        -> Solutions:
    n_oct_levels = options.number_octree_levels
    solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista and False:
        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
        for e, mesh in enumerate(solutions.dc_meshes):
            colors = ["red", "green", "blue", "yellow", "orange", "purple", "black", "white"]
            plot_dc_meshes(p, dc_mesh=mesh, color=colors[e])
        p.show()

    return solutions
