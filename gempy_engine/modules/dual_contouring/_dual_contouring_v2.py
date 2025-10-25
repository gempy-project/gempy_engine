import os
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
def compute_dual_contouring_v2(dc_data_list: list[DualContouringData], ) -> List[DualContouringMesh]:
    parallel_results = _parallel_process(dc_data_list)

    if parallel_results is not None:
        return parallel_results


    # Fall back to sequential processing
    print(f"Using sequential processing for {len(dc_data_list)} surfaces")
    stack_meshes: List[DualContouringMesh] = []

    for dc_data in dc_data_list:
        mesh = _process_one_surface(dc_data, dc_data.left_right_codes)
        stack_meshes.append(mesh)
    return stack_meshes


def _parallel_process(dc_data_list: list[DualContouringData]):
    # Check if we should use parallel processing

    return None  # * Not clear wins
    
    n_surfaces_to_export = len(dc_data_list)
    use_parallel = _should_use_parallel_processing(n_surfaces_to_export, BackendTensor.engine_backend)
    parallel_results = None

    if use_parallel and False:  # ! (Miguel Sep 25) I do not see a speedup
        print(f"Using parallel processing for {n_surfaces_to_export} surfaces")
        parallel_results = _parallel_process_surfaces_v2(dc_data_list)

    return parallel_results


def _parallel_process_surfaces_v2(dc_data_list: list[DualContouringData], num_workers=None, chunk_size=2):
    """Process surfaces in parallel using multiprocessing."""
    n_surfaces = len(dc_data_list)

    if num_workers is None:
        num_workers = max(1, min(os.cpu_count() // 2, n_surfaces // 2))
        num_workers=3

    # Prepare data for serialization - convert each DualContouringData to dict
    dc_data_dicts = []
    for dc_data in dc_data_list:
        dc_data_dict = {
                'xyz_on_edge'         : dc_data.xyz_on_edge,
                'valid_edges'         : dc_data.valid_edges,
                'xyz_on_centers'      : dc_data.xyz_on_centers,
                'dxdydz'              : dc_data.dxdydz,
                'gradients'           : dc_data.gradients,
                'left_right_codes'    : dc_data.left_right_codes,
                'n_surfaces_to_export': dc_data.n_surfaces_to_export,
                'tree_depth'          : dc_data.tree_depth
        }
        dc_data_dicts.append(dc_data_dict)

    # Create surface index chunks
    surface_indices = list(range(n_surfaces))
    chunks = [surface_indices[i:i + chunk_size] for i in range(0, len(surface_indices), chunk_size)]

    try:
        # Use spawn context for better PyTorch compatibility
        ctx = mp.get_context("fork") if MULTIPROCESSING_AVAILABLE else mp

        with ctx.Pool(processes=num_workers, initializer=_init_worker) as pool:
            # Submit all chunks
            async_results = []
            for chunk in chunks:
                result = pool.apply_async(
                    _process_surface_batch_v2,
                    (chunk, dc_data_dicts )
                )
                async_results.append(result)

            # Collect results
            all_results = []
            for async_result in async_results:
                batch_results = async_result.get()
                all_results.extend(batch_results)

        return all_results

    except Exception as e:
        print(f"Parallel processing failed: {e}. Falling back to sequential processing.")
        return None


def _process_surface_batch_v2(surface_indices, dc_data_dicts, left_right_codes):
    """Process a batch of surfaces. This function runs in worker processes."""
    results = []

    for idx in surface_indices:
        dc_data_dict = dc_data_dicts[idx]

        # Reconstruct DualContouringData from dict
        dc_data = DualContouringData(
            xyz_on_edge=dc_data_dict['xyz_on_edge'],
            valid_edges=dc_data_dict['valid_edges'],
            xyz_on_centers=dc_data_dict['xyz_on_centers'],
            dxdydz=dc_data_dict['dxdydz'],
            gradients=dc_data_dict['gradients'],
            left_right_codes=dc_data_dict['left_right_codes'],
            n_surfaces_to_export=dc_data_dict['n_surfaces_to_export'],
            tree_depth=dc_data_dict['tree_depth']
        )
        # Process the surface
        mesh = _process_one_surface(dc_data, dc_data.left_right_codes)
        results.append(mesh)

    return results
def _process_one_surface(dc_data: DualContouringData, left_right_codes) -> DualContouringMesh:
    vertices = generate_dual_contouring_vertices(dc_data, slice_surface=None, debug=False)
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
    if TRIMESH_LAST_PASS := True:
        vertices_numpy, indices_numpy = _last_pass(vertices_numpy, indices_numpy)

    mesh = DualContouringMesh(vertices_numpy, indices_numpy, dc_data)
    return mesh


def _last_pass(vertices, indices):
    """Apply trimesh post-processing if available."""
    if not TRIMESH_AVAILABLE:
        return vertices, indices
    
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=indices)
        mesh.fill_holes()
        return mesh.vertices, mesh.faces
    except Exception as e:
        print(f"Warning: Trimesh post-processing failed: {e}")
        return vertices, indices
