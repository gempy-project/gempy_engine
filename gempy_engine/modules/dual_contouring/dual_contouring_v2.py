import concurrent.futures
import os
from typing import List

from ._gen_vertices import generate_dual_contouring_vertices
from ...config import AvailableBackends
from ...core.backend_tensor import BackendTensor
from ...core.data.dual_contouring_data import DualContouringData
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.utils import gempy_profiler_decorator
from ...modules.dual_contouring.fancy_triangulation import triangulate


@gempy_profiler_decorator
def compute_dual_contouring_v2(dc_data_list: list[DualContouringData], max_workers: int = None) -> List[DualContouringMesh]:
    n_surfaces = len(dc_data_list)
    if n_surfaces == 0:
        return []
    # --- PYTORCH BUG WORKAROUND ---
    # Force the lazy initialization of the linalg module on the main thread 
    # to prevent race conditions during parallel voxel processing.
    if BackendTensor.engine_backend == AvailableBackends.PYTORCH and BackendTensor.use_gpu:
        import torch
        _dummy_matrix = torch.ones((1, 1), device=BackendTensor.device, dtype=BackendTensor.dtype_obj)
        _ = torch.linalg.inv(_dummy_matrix)

    if os.getenv("DUAL_CONTOURING_MULTITHREAD", "False") == "True":
        # Use ThreadPoolExecutor to parallelize surface processing
        # This is similar to the approach in _weighted_qef_setup_multicore.py
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [
                    executor.submit(_process_one_surface, dc_data, dc_data.left_right_codes)
                    for dc_data in dc_data_list
            ]

            # Collect results in order
            stack_meshes: List[DualContouringMesh] = []
            for future in futures:
                stack_meshes.append(future.result())

    else:
        stack_meshes = [_process_one_surface(dc_data, dc_data.left_right_codes) for dc_data in dc_data_list]

    return stack_meshes


def _process_one_surface(dc_data: DualContouringData, left_right_codes) -> DualContouringMesh:
    vertices = generate_dual_contouring_vertices(dc_data, slice_surface=None, debug=False)

    if os.getenv("GEMPY_SKIP_TRIANGULATION", "0").lower() in ("true", "1", "t", "y", "yes"):
        mesh = DualContouringMesh(vertices, BackendTensor.t.array([[], []]), dc_data)
        return mesh

    # * Average gradient for the edges
    valid_edges = dc_data.valid_edges
    edges_normals = BackendTensor.t.zeros((valid_edges.shape[0], 12, 3), dtype=BackendTensor.dtype_obj)
    edges_normals[:] = 0
    edges_normals[valid_edges] = dc_data.gradients

    indices = _compute_triangulation(
        dc_data_per_surface=dc_data,
        left_right_codes=left_right_codes,
        edges_normals=edges_normals,
        vertex=vertices
    )

    mesh = DualContouringMesh(vertices, indices, dc_data)
    return mesh


def _compute_triangulation(dc_data_per_surface: DualContouringData,
                           left_right_codes, edges_normals, vertex):
    """Compute triangulation indices for a specific surface."""

    # * Fancy triangulation 👗
    valid_voxels = dc_data_per_surface.valid_voxels

    left_right_per_surface = left_right_codes[valid_voxels]
    valid_voxels_per_surface = dc_data_per_surface.valid_edges[valid_voxels]
    tree_depth_per_surface = dc_data_per_surface.tree_depth

    voxels_normals = edges_normals[valid_voxels]

    indices = triangulate(
        left_right_array=left_right_per_surface,
        valid_edges=valid_voxels_per_surface,
        tree_depth=tree_depth_per_surface,
        voxel_normals=voxels_normals,
        vertex=vertex,
        base_number=dc_data_per_surface.base_number
    )

    # @on
    return indices
