import warnings

import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor


def _surface_slicer(surface_i: int, valid_edges_per_surface) -> slice:
    next_surface_edge_idx: int = valid_edges_per_surface[:surface_i + 1].sum()
    if surface_i == 0:
        last_surface_edge_idx = 0
    else:
        last_surface_edge_idx: int = valid_edges_per_surface[:surface_i].sum()
    slice_object: slice = slice(last_surface_edge_idx, next_surface_edge_idx)

    return slice_object

def _calc_mesh_normals(vertices_numpy, indices_numpy):
    # Calculate face normals for each triangle
    # Get the three vertices of each triangle
    v0 = vertices_numpy[indices_numpy[:, 0]]
    v1 = vertices_numpy[indices_numpy[:, 1]]
    v2 = vertices_numpy[indices_numpy[:, 2]]

    # Calculate edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Calculate face normals using cross product
    mesh_normals = np.cross(edge1, edge2)

    # Normalize the normals
    norms = np.linalg.norm(mesh_normals, axis=1, keepdims=True)
    # Avoid division by zero
    # norms = np.where(norms == 0, 1, norms)
    # mesh_normals = mesh_normals / norms

    return mesh_normals

def _correct_normals(vertices_numpy, indices_numpy, edges_normals):
    # Calculate face normals for each triangle
    # Get the three vertices of each triangle
    v0 = vertices_numpy[indices_numpy[:, 0]]
    v1 = vertices_numpy[indices_numpy[:, 1]]
    v2 = vertices_numpy[indices_numpy[:, 2]]

    # Calculate edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Calculate face normals using cross product
    mesh_normals = BackendTensor.t.cross(edge1, edge2)

    edge_normals_avg = edges_normals

    # For each triangle, we need to match it to the corresponding edge normal
    # Assuming the triangle index corresponds to the voxel/edge index
    # If the shapes match, we can directly compare
    n_triangles = indices_numpy.shape[0]
    n_edge_normals = edge_normals_avg.shape[0]

    reference_normals = edge_normals_avg

    # Normalize reference normals

    # Calculate dot product between mesh normals and reference normals
    # Positive dot product means they point in roughly the same direction
    dot_products = BackendTensor.t.sum(mesh_normals * reference_normals, axis=1)

    # Find triangles where normals point in opposite directions (dot product < 0)
    flip_mask = dot_products < 0

    # Flip triangles by swapping vertex indices 1 and 2
    indices_corrected = BackendTensor.t.copy(indices_numpy)
    indices_corrected[flip_mask, 1], indices_corrected[flip_mask, 2] = BackendTensor.t.copy(indices_numpy[flip_mask, 2]), BackendTensor.t.copy(indices_numpy[flip_mask, 1])

    return indices_corrected, mesh_normals