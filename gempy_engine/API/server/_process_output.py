import json
import logging
from typing import List

import numpy as np
import pandas as pd
import subsurface
from subsurface import UnstructuredData

from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.core.data.solutions import Solutions

PLOT_SUBSURFACE_OBJECT = False


def process_output(
    meshes: List[DualContouringMesh],
    n_stack: int,
    solutions: Solutions,
    logger: logging.Logger
) -> bytes:
    """
    Process model outputs into binary format for transmission.
    
    Args:
        meshes: List of dual contouring meshes
        n_stack: Number of stacks in the model
        solutions: Model solutions containing octree data
        logger: Logger instance
        
    Returns:
        Binary data containing serialized meshes and octrees
    """
    # Serialize meshes
    unstructured_data_meshes: UnstructuredData = _meshes_to_unstruct(meshes, logger)
    if PLOT_SUBSURFACE_OBJECT:
        _plot_subsurface_object(unstructured_data_meshes)
    body_meshes = unstructured_data_meshes.to_binary()
    
    # Serialize octrees
    unstructured_data_volume = subsurface.UnstructuredData.from_array(
        vertex=solutions.octrees_output[0].grid_centers.values,
        cells="points",
        cells_attr=pd.DataFrame(
            data=solutions.octrees_output[0].last_output_center.ids_block,
            columns=['id']
        )
    )
    body_volume = unstructured_data_volume.to_binary()
    
    # Serialize global header and combine data
    body = body_meshes + body_volume
    global_header = json.dumps({"mesh_size": len(body_meshes), "octree_size": len(body_volume)})
    global_header_bytes = global_header.encode('utf-8')
    global_header_length = len(global_header_bytes)
    global_header_length_bytes = global_header_length.to_bytes(4, byteorder='little')
    
    return global_header_length_bytes + global_header_bytes + body


def _meshes_to_unstruct(
    meshes: List[DualContouringMesh], 
    logger: logging.Logger
) -> UnstructuredData:
    """
    Convert a list of dual contouring meshes to an unstructured data format.
    
    Args:
        meshes: List of dual contouring meshes
        logger: Logger instance
        
    Returns:
        Unstructured data representation of the meshes
    """
    n_meshes = len(meshes)
    logger.debug(f"Number of meshes: {n_meshes}")
    logger.debug(f"Mesh1TriShape: {meshes[0].edges.shape}")

    # Prepare the vertex array
    vertex_array = np.concatenate([meshes[i].vertices for i in range(n_meshes)])
    
    # Check for edge uniqueness (debugging)
    simplex_array = np.concatenate([meshes[i].edges for i in range(n_meshes)])
    unc, count = np.unique(simplex_array, axis=0, return_counts=True)
    logger.debug(f"edges shape {simplex_array.shape}")

    # Prepare the simplex array with proper indexing
    simplex_array = meshes[0].edges
    for i in range(1, n_meshes):
        adder = np.max(meshes[i - 1].edges) + 1
        logger.debug(f"triangle counts adder: {adder}")
        add_mesh = meshes[i].edges + adder
        simplex_array = np.append(simplex_array, add_mesh, axis=0)

    logger.debug(f"edges shape {simplex_array.shape}")

    # Prepare the cells_attr array
    ids_array = np.ones(simplex_array.shape[0])
    l0 = 0
    id_value = 1
    for mesh in meshes:
        l1 = l0 + mesh.edges.shape[0]
        logger.debug(f"l0 {l0} l1 {l1} id {id_value}")
        ids_array[l0:l1] = id_value
        l0 = l1
        id_value += 1
    logger.debug(f"ids_array count: {np.unique(ids_array)}")

    # Create the unstructured data
    unstructured_data = subsurface.UnstructuredData.from_array(
        vertex=vertex_array,
        cells=simplex_array,
        cells_attr=pd.DataFrame(ids_array, columns=['id'])
    )

    return unstructured_data


def _plot_subsurface_object(unstructured_data: UnstructuredData) -> None:
    """
    Visualize the unstructured data using subsurface.
    
    Args:
        unstructured_data: The unstructured data to visualize
    """
    print(unstructured_data)
    obj = subsurface.TriSurf(unstructured_data)
    pv_unstruct = subsurface.visualization.to_pyvista_mesh(obj)
    print(pv_unstruct)
    subsurface.visualization.pv_plot([pv_unstruct])