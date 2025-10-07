import os
from typing import List

from ... import optional_dependencies
from ...core.backend_tensor import BackendTensor
from ...core.data.dual_contouring_data import DualContouringData
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.utils import gempy_profiler_decorator
from ._parallel_triangulation import _should_use_parallel_processing, _process_surface_batch, _init_worker
from ._sequential_triangulation import _sequential_triangulation
from ._gen_vertices import _compute_vertices

# Multiprocessing imports
try:
    import torch.multiprocessing as mp

    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    import multiprocessing as mp

    MULTIPROCESSING_AVAILABLE = False


@gempy_profiler_decorator
def compute_dual_contouring(dc_data_per_stack: DualContouringData,
                            left_right_codes=None, debug: bool = False) -> List[DualContouringMesh]:
    valid_edges_per_surface = dc_data_per_stack.valid_edges.reshape((dc_data_per_stack.n_surfaces_to_export, -1, 12))

    # Check if we should use parallel processing
    use_parallel = _should_use_parallel_processing(dc_data_per_stack.n_surfaces_to_export, BackendTensor.engine_backend)
    parallel_results = None

    if use_parallel and True:  # ! (Miguel Sep 25) I do not see a speedup
        print(f"Using parallel processing for {dc_data_per_stack.n_surfaces_to_export} surfaces")
        parallel_results = _parallel_process_surfaces(dc_data_per_stack, left_right_codes, debug)

    # Fall back to sequential processing
    print(f"Using sequential processing for {dc_data_per_stack.n_surfaces_to_export} surfaces")
    stack_meshes: List[DualContouringMesh] = []

    for i in range(dc_data_per_stack.n_surfaces_to_export):
        # @off
        if parallel_results is not None:
            _, vertices_numpy = _compute_vertices(
                dc_data_per_stack=dc_data_per_stack,
                debug=debug,
                surface_i=i,
                valid_edges_per_surface=valid_edges_per_surface
            )
            indices_numpy = parallel_results[i]
        else:
            dc_data_per_surface, indices_numpy, vertices_numpy = _sequential_triangulation(
                dc_data_per_stack=dc_data_per_stack,
                debug=debug,
                i=i,
                valid_edges_per_surface=valid_edges_per_surface,
                left_right_codes=left_right_codes
            )
            
            # Handle None left_right_codes case
            if left_right_codes is not None:
                valid_left_right_codes = left_right_codes[dc_data_per_surface.valid_voxels]
            else:
                valid_left_right_codes = None

        if TRIMESH_LAST_PASS := True:
            vertices_numpy, indices_numpy = _last_pass(vertices_numpy, indices_numpy)

        stack_meshes.append(
            DualContouringMesh(
                vertices_numpy,
                indices_numpy,
                dc_data_per_stack
            )
        )
    return stack_meshes


def _parallel_process_surfaces(dc_data_per_stack, left_right_codes, debug, num_workers=None, chunk_size=2):
    """Process surfaces in parallel using multiprocessing."""
    if num_workers is None:
        num_workers = max(1, min(os.cpu_count() // 2, dc_data_per_stack.n_surfaces_to_export // 2))

    # Prepare data for serialization
    dc_data_dict = {
            'xyz_on_edge'             : dc_data_per_stack.xyz_on_edge,
            'valid_edges'             : dc_data_per_stack.valid_edges,
            'xyz_on_centers'          : dc_data_per_stack.xyz_on_centers,
            'dxdydz'                  : dc_data_per_stack.dxdydz,
            'gradients': dc_data_per_stack.gradients,
            'n_surfaces_to_export'    : dc_data_per_stack.n_surfaces_to_export,
            'tree_depth'              : dc_data_per_stack.tree_depth,
    }

    # Create surface index chunks
    surface_indices = list(range(dc_data_per_stack.n_surfaces_to_export))
    chunks = [surface_indices[i:i + chunk_size] for i in range(0, len(surface_indices), chunk_size)]

    # Handle None left_right_codes case - ensure we pass a serializable value
    serializable_left_right_codes = left_right_codes

    try:
        # Use spawn context for better PyTorch compatibility
        ctx = mp.get_context("spawn") if MULTIPROCESSING_AVAILABLE else mp

        with ctx.Pool(processes=num_workers, initializer=_init_worker) as pool:
            # Submit all chunks
            async_results = []
            for chunk in chunks:
                result = pool.apply_async(
                    _process_surface_batch,
                    (chunk, dc_data_dict, serializable_left_right_codes, debug)
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


def _last_pass(vertices, indices):
    # Check if trimesh is available
    try:
        trimesh = optional_dependencies.require_trimesh()
        mesh = trimesh.Trimesh(vertices=vertices, faces=indices)
        mesh.fill_holes()
        return mesh.vertices, mesh.faces
    except ImportError:
        return vertices, indices
