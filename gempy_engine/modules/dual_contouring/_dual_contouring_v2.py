import os
import numpy as np
import concurrent.futures
from typing import List

from ._gen_vertices import generate_dual_contouring_vertices
from ._parallel_triangulation import _should_use_parallel_processing, _init_worker
from ._sequential_triangulation import _compute_triangulation
from ... import optional_dependencies
from ...core.backend_tensor import BackendTensor
from ...core.data.dual_contouring_data import DualContouringData
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.utils import gempy_profiler_decorator

# Multiprocessing imports
try:
    import torch.multiprocessing as mp

    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    import multiprocessing as mp

    MULTIPROCESSING_AVAILABLE = False

# Import trimesh once at module level
TRIMESH_AVAILABLE = False
try:
    trimesh = optional_dependencies.require_trimesh()
    TRIMESH_AVAILABLE = True
except ImportError:
    trimesh = None


@gempy_profiler_decorator
def compute_dual_contouring_v2(dc_data_list: list[DualContouringData], max_workers: int = None) -> List[DualContouringMesh]:
    n_surfaces = len(dc_data_list)
    if n_surfaces == 0:
        return []

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

    return stack_meshes


def _process_one_surface(dc_data: DualContouringData, left_right_codes) -> DualContouringMesh:
    vertices = generate_dual_contouring_vertices(dc_data, slice_surface=None, debug=False)

    if os.environ.get('GEMPY_SKIP_TRIANGULATION', '0') == '1':
        vertices_numpy = BackendTensor.t.to_numpy(vertices)
        mesh = DualContouringMesh(vertices_numpy, np.array([]), dc_data)
        return mesh

    # * Average gradient for the edges
    valid_edges = dc_data.valid_edges
    edges_normals = BackendTensor.t.zeros((valid_edges.shape[0], 12, 3), dtype=BackendTensor.dtype_obj)
    edges_normals[:] = 0
    edges_normals[valid_edges] = dc_data.gradients

    indices_numpy = _compute_triangulation(
        dc_data_per_surface=dc_data,
        left_right_codes=left_right_codes,
        edges_normals=edges_normals,
        vertex=vertices
    )

    vertices_numpy = BackendTensor.t.to_numpy(vertices)
    mesh = DualContouringMesh(vertices_numpy, indices_numpy, dc_data)
    return mesh
