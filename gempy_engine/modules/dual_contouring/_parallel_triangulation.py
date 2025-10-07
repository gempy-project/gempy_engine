import numpy as np
import os
import warnings

from gempy_engine.config import AvailableBackends
from ...core.backend_tensor import BackendTensor
from ...core.data.dual_contouring_data import DualContouringData
from ._triangulate import triangulate_dual_contouring
from ...modules.dual_contouring.fancy_triangulation import triangulate

# Multiprocessing imports
try:
    import torch.multiprocessing as mp
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    import multiprocessing as mp
    MULTIPROCESSING_AVAILABLE = False


def _should_use_parallel_processing(n_surfaces: int, backend: AvailableBackends) -> bool:
    """Determine if parallel processing should be used."""
    # Only use parallel processing for PyTorch CPU backend with sufficient surfaces
    if backend == AvailableBackends.PYTORCH and MULTIPROCESSING_AVAILABLE:
        # Check if we're on CPU (not GPU)
        try:
            import torch
            if torch.cuda.is_available():
                # If CUDA is available, check if default tensor type is CPU
                dummy = BackendTensor.t.zeros(1)
                is_cpu = dummy.device.type == 'cpu' if hasattr(dummy, 'device') else True
            else:
                is_cpu = True

            # Use parallel processing if we have CPU tensors and enough surfaces to justify overhead
            return is_cpu and n_surfaces >= 4
        except ImportError:
            return False
    return False


def _init_worker():
    """Initialize worker process to avoid thread oversubscription."""
    # Set environment variables for NumPy/OpenMP/MKL
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    # For PyTorch, set environment variables before import
    os.environ['TORCH_NUM_THREADS'] = '1'
    os.environ['TORCH_NUM_INTEROP_THREADS'] = '1'

    # Now import torch in the worker process
    try:
        import torch
        # These calls might still work if torch hasn't done any parallel work yet in this process
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # If the above fails, the environment variables should handle it
            pass
    except ImportError:
        pass


def _process_surface_batch(surface_indices_batch, dc_data_dict, left_right_codes, debug):
    """Process a batch of surfaces in a worker process."""
    _init_worker()

    # Reconstruct dc_data_per_stack from dictionary
    dc_data_per_stack = DualContouringData(**dc_data_dict)
    valid_edges_per_surface = dc_data_per_stack.valid_edges.reshape((dc_data_per_stack.n_surfaces_to_export, -1, 12))

    batch_results = []

    for i in surface_indices_batch:
        result = _process_single_surface(
            i, dc_data_per_stack, valid_edges_per_surface, left_right_codes, debug
        )
        batch_results.append(result)

    return batch_results

def _process_single_surface(i, dc_data_per_stack, valid_edges_per_surface, left_right_codes, debug):
    """Process a single surface and return vertices and indices."""
    try:
        valid_edges = valid_edges_per_surface[i]

        # Calculate edge indices for this surface
        last_surface_edge_idx = sum(valid_edges_per_surface[j].sum() for j in range(i))
        next_surface_edge_idx = valid_edges.sum() + last_surface_edge_idx
        slice_object = slice(last_surface_edge_idx, next_surface_edge_idx)

        dc_data_per_surface = DualContouringData(
            xyz_on_edge=dc_data_per_stack.xyz_on_edge,
            valid_edges=valid_edges,
            xyz_on_centers=dc_data_per_stack.xyz_on_centers,
            dxdydz=dc_data_per_stack.dxdydz,
            gradients=dc_data_per_stack.gradients,
            n_surfaces_to_export=dc_data_per_stack.n_surfaces_to_export,
            tree_depth=dc_data_per_stack.tree_depth
        )

        if left_right_codes is None:
            # Legacy triangulation
            indices = triangulate_dual_contouring(dc_data_per_surface)
        else:
            edges_normals = BackendTensor.t.zeros((valid_edges.shape[0], 12, 3), dtype=BackendTensor.dtype_obj)
            if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
                edges_normals[:] = float('nan')  # Use Python float nan instead of np.nan
            else:
                edges_normals[:] = np.nan

            # Get gradient data
            gradient_data = dc_data_per_stack.gradients[slice_object]

            # Fix dtype mismatch by ensuring compatible dtypes
            if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
                if hasattr(gradient_data, 'dtype') and hasattr(edges_normals, 'dtype'):
                    if gradient_data.dtype != edges_normals.dtype:
                        gradient_data = gradient_data.to(edges_normals.dtype)

            edges_normals[valid_edges] = gradient_data

            if BackendTensor.engine_backend != AvailableBackends.PYTORCH:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    voxel_normal = np.nanmean(edges_normals, axis=1)
                    voxel_normal = voxel_normal[(~np.isnan(voxel_normal).any(axis=1))]
            else:
                # PyTorch tensor operations
                nan_mask = BackendTensor.t.isnan(edges_normals)
                valid_count = (~nan_mask).sum(dim=1)
                safe_normals = edges_normals.clone()
                safe_normals[nan_mask] = 0
                sum_normals = BackendTensor.t.sum(safe_normals, 1)
                voxel_normal = sum_normals / valid_count.clamp(min=1)
                voxel_normal = voxel_normal[valid_count > 0].reshape(-1, 3)

            valid_voxels = dc_data_per_surface.valid_voxels
            left_right_per_surface = left_right_codes[valid_voxels]
            valid_voxels_per_surface = dc_data_per_surface.valid_edges[valid_voxels]
            tree_depth_per_surface = dc_data_per_surface.tree_depth

            indices = triangulate(
                left_right_array=left_right_per_surface,
                valid_edges=valid_voxels_per_surface,
                tree_depth=tree_depth_per_surface,
                voxel_normals=voxel_normal,
                vertex=vertex
            )
            indices = BackendTensor.t.concatenate(indices, axis=0)

        # vertices_numpy = BackendTensor.t.to_numpy(vertices)
        indices_numpy = BackendTensor.t.to_numpy(indices)
        return indices_numpy

    except Exception as e:
        print(f"ERROR in _process_single_surface for surface {i}: {e}")
        import traceback
        traceback.print_exc()
        raise

