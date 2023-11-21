import warnings
from typing import List

import numpy as np

from gempy_engine.config import AvailableBackends

from ...core.backend_tensor import BackendTensor
from ...core.data.dual_contouring_data import DualContouringData
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.utils import gempy_profiler_decorator
from ...modules.dual_contouring.dual_contouring_interface import triangulate_dual_contouring, generate_dual_contouring_vertices
from ...modules.dual_contouring.fancy_triangulation import triangulate


@gempy_profiler_decorator
def compute_dual_contouring(dc_data_per_stack: DualContouringData, left_right_codes=None, debug: bool = False) -> List[DualContouringMesh]:
    valid_edges_per_surface = dc_data_per_stack.valid_edges.reshape((dc_data_per_stack.n_surfaces_to_export, -1, 12))

    # ? Is  there a way to cut also the vertices?

    stack_meshes: List[DualContouringMesh] = []

    last_surface_edge_idx = 0
    for i in range(dc_data_per_stack.n_surfaces_to_export):
        # @off
        valid_edges          : np.ndarray = valid_edges_per_surface[i]
        next_surface_edge_idx: int        = valid_edges.sum() + last_surface_edge_idx
        slice_object         : slice      = slice(last_surface_edge_idx, next_surface_edge_idx)
        last_surface_edge_idx: int        = next_surface_edge_idx

        dc_data_per_surface = DualContouringData(
            xyz_on_edge              = dc_data_per_stack.xyz_on_edge,
            valid_edges              = valid_edges,
            xyz_on_centers           = dc_data_per_stack.xyz_on_centers,
            dxdydz                   = dc_data_per_stack.dxdydz,
            exported_fields_on_edges = dc_data_per_stack.exported_fields_on_edges,
            n_surfaces_to_export     = dc_data_per_stack.n_surfaces_to_export,
            tree_depth               = dc_data_per_stack.tree_depth

        )
        vertices: np.ndarray = generate_dual_contouring_vertices(
            dc_data_per_stack = dc_data_per_surface,
            slice_surface     = slice_object,
            debug             = debug
        )
        
        if left_right_codes is None:
            # * Legacy triangulation
            indices = triangulate_dual_contouring(dc_data_per_surface)
        else:
            # * Fancy triangulation 👗
            
            # * Average gradient for the edges
            edges_normals = BackendTensor.t.zeros((valid_edges.shape[0], 12, 3), dtype=BackendTensor.dtype_obj)
            edges_normals[:] = np.nan
            edges_normals[valid_edges] = dc_data_per_stack.gradients[slice_object]
                
            # if LEGACY:=True:
            if BackendTensor.engine_backend != AvailableBackends.PYTORCH:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    voxel_normal  = np.nanmean(edges_normals, axis=1)
                    voxel_normal  = voxel_normal[(~np.isnan(voxel_normal).any(axis=1))]  # drop nans
                    pass 
            else:
                # Assuming edges_normals is a PyTorch tensor
                nan_mask = BackendTensor.t.isnan(edges_normals)
                valid_count = (~nan_mask).sum(dim=1)

                # Replace NaNs with 0 for sum calculation
                safe_normals = edges_normals.clone()
                safe_normals[nan_mask] = 0

                # Compute the sum of non-NaN elements
                sum_normals = BackendTensor.t.sum(safe_normals, 1)

                # Calculate the mean, avoiding division by zero
                voxel_normal = sum_normals / valid_count.clamp(min=1)

                # Remove rows where all elements were NaN (and hence valid_count is 0)
                voxel_normal = voxel_normal[valid_count > 0].reshape(-1, 3)
                

            valid_voxels = dc_data_per_surface.valid_voxels
            indices = triangulate(
                left_right_array = left_right_codes[valid_voxels],
                valid_edges      = dc_data_per_surface.valid_edges[valid_voxels],
                tree_depth       = dc_data_per_surface.tree_depth,
                voxel_normals     = voxel_normal 
            )
            indices = np.vstack(indices)
            
        # @on
        stack_meshes.append(
            DualContouringMesh(
                BackendTensor.t.to_numpy(vertices),
                indices,
                dc_data_per_stack
            )
        )
    return stack_meshes
