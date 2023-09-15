import json
import logging

import numpy as np
import pandas as pd
import subsurface
from subsurface import UnstructuredData

from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.core.data.solutions import Solutions

PLOT_SUBSURFACE_OBJECT = False


def process_output(meshes: list[DualContouringMesh], n_stack: int, solutions: Solutions, logger: logging.Logger) \
        -> bytes:
    # * serialize meshes
    unstructured_data_meshes: UnstructuredData = _meshes_to_unstruct(meshes, logger)
    if PLOT_SUBSURFACE_OBJECT:
        _plot_subsurface_object(unstructured_data_meshes)
    body_meshes, header_meshes = unstructured_data_meshes.to_binary()
    header_json = json.dumps(header_meshes)
    header_json_bytes = header_json.encode('utf-8')
    header_json_length = len(header_json_bytes)
    header_json_length_bytes = header_json_length.to_bytes(4, byteorder='little')
    body_meshes = header_json_length_bytes + header_json_bytes + body_meshes

    # * serialize octrees
    unstructured_data_volume = subsurface.UnstructuredData.from_array(
        vertex=solutions.octrees_output[0].grid_centers.values,
        cells="points",
        cells_attr=pd.DataFrame(
            data=solutions.octrees_output[0].last_output_center.ids_block,
            columns=['id']
        )  # TODO: We have to create an array with the shape of simplex array with the id of each simplex
    )
    
    body_volume, header_volume = unstructured_data_volume.to_binary()
    header_json = json.dumps(header_volume)
    header_json_bytes = header_json.encode('utf-8')
    header_json_length = len(header_json_bytes)
    header_json_length_bytes = header_json_length.to_bytes(4, byteorder='little')
    body_volume = header_json_length_bytes + header_json_bytes + body_volume

    # * serialize global header
    body = body_meshes + body_volume
    global_header = json.dumps({"mesh_size": len(body_meshes), "octree_size": len(body_volume)})
    global_header_bytes = global_header.encode('utf-8')
    global_header_length = len(global_header_bytes)
    global_header_length_bytes = global_header_length.to_bytes(4, byteorder='little')
    body = global_header_length_bytes + global_header_bytes + body
    return body


def _meshes_to_unstruct(meshes: list[DualContouringMesh], logger: logging.Logger) -> UnstructuredData:
    # ? I added this function to the Solutions class
    n_meshes = len(meshes)
    logger.debug(f"Number of meshes: {n_meshes}")
    logger.debug("Mesh1TriShape" + str(meshes[0].edges.shape))

    vertex_array = np.concatenate([meshes[i].vertices for i in range(n_meshes)])
    simplex_array = np.concatenate([meshes[i].edges for i in range(n_meshes)])
    unc, count = np.unique(simplex_array, axis=0, return_counts=True)
    logger.debug(f"edges shape {simplex_array.shape}")

    # * Prepare the simplex array
    simplex_array = meshes[0].edges
    for i in range(1,n_meshes):
        adder = np.max(meshes[i - 1].edges) + 1
        logger.debug("triangle counts adder:" + str(adder))
        add_mesh = meshes[i].edges + adder
        simplex_array = np.append(simplex_array, add_mesh, axis=0)

    logger.debug(f"edges shape {simplex_array.shape}")

    # * Prepare the cells_attr array
    ids_array = np.ones(simplex_array.shape[0])
    l0 = 0
    id = 1
    for mesh in meshes:
        l1 = l0 + mesh.edges.shape[0]
        logger.debug(f"l0 {l0} l1 {l1} id {id}")
        ids_array[l0:l1] = id
        l0 = l1
        id += 1
    logger.debug("ids_array count" + str(np.unique(ids_array)))

    # * Create the unstructured data
    unstructured_data = subsurface.UnstructuredData.from_array(
        vertex=vertex_array,
        cells=simplex_array,
        cells_attr=pd.DataFrame(ids_array, columns=['id'])  # TODO: We have to create an array with the shape of simplex array with the id of each simplex
    )

    return unstructured_data


def _plot_subsurface_object(unstructured_data):
    print(unstructured_data)
    obj = subsurface.TriSurf(unstructured_data)
    pv_unstruct = subsurface.visualization.to_pyvista_mesh(obj)
    print(pv_unstruct)
    subsurface.visualization.pv_plot([pv_unstruct])
