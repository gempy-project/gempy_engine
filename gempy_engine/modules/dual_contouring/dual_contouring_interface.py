import warnings
from typing import Tuple, List

import numpy as np

from ._vertex_overlap import find_repeated_voxels_across_stacks, _apply_fault_relations_to_overlaps
from .fancy_triangulation import get_left_right_array
from ...core.backend_tensor import BackendTensor
from ...core.data import InterpolationOptions
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.interp_output import InterpOutput
from ...core.data.octree_level import OctreeLevel
from ...core.data.options import MeshExtractionMaskingOptions
from ...core.data.stack_relation_type import StackRelationType
from ...core.data.stacks_structure import StacksStructure


# region edges
def find_intersection_on_edge(_xyz_corners, scalar_field_on_corners,
                              scalar_at_sp, masking=None) -> Tuple:
    """This function finds all the intersections for multiple layers per series
    
    - The shape of valid edges is n_surfaces * xyz_corners. Where xyz_corners is 8 * the octree leaf
    - The shape of intersection_xyz really depends on the number of intersections per voxel
    
    
    """
    scalar_8_ = scalar_field_on_corners
    scalar_8 = scalar_8_.reshape((1, -1, 8))
    xyz_8 = _xyz_corners.reshape((-1, 8, 3))

    if masking is not None:
        ma_8 = masking
        xyz_8 = xyz_8[ma_8]
        scalar_8 = scalar_8[:, ma_8]

    scalar_at_sp = scalar_at_sp.reshape((-1, 1, 1))

    n_isosurface = scalar_at_sp.shape[0]
    xyz_8 = BackendTensor.tfnp.tile(xyz_8, (n_isosurface, 1, 1))  # TODO: Generalize

    # Compute distance of scalar field on the corners
    scalar_dx = scalar_8[:, :, :4] - scalar_8[:, :, 4:]
    scalar_d_y = scalar_8[:, :, [0, 1, 4, 5]] - scalar_8[:, :, [2, 3, 6, 7]]
    scalar_d_z = scalar_8[:, :, ::2] - scalar_8[:, :, 1::2]

    # Add a tiny value to avoid division by zero
    scalar_dx += 1e-10
    scalar_d_y += 1e-10
    scalar_d_z += 1e-10

    # Compute the weights
    weight_x = ((scalar_at_sp - scalar_8[:, :, 4:]) / scalar_dx).reshape(-1, 4, 1)
    weight_y = ((scalar_at_sp - scalar_8[:, :, [2, 3, 6, 7]]) / scalar_d_y).reshape(-1, 4, 1)
    weight_z = ((scalar_at_sp - scalar_8[:, :, 1::2]) / scalar_d_z).reshape(-1, 4, 1)

    # Calculate eucledian distance between the corners
    d_x = xyz_8[:, :4] - xyz_8[:, 4:]
    d_y = xyz_8[:, [0, 1, 4, 5]] - xyz_8[:, [2, 3, 6, 7]]
    d_z = xyz_8[:, ::2] - xyz_8[:, 1::2]

    # Compute the weighted distance
    intersect_dx = d_x[:, :, :] * weight_x[:, :, :]
    intersect_dy = d_y[:, :, :] * weight_y[:, :, :]
    intersect_dz = d_z[:, :, :] * weight_z[:, :, :]

    # Mask invalid edges
    valid_edge_x = BackendTensor.tfnp.logical_and(weight_x > -0.01, weight_x < 1.01)
    valid_edge_y = BackendTensor.tfnp.logical_and(weight_y > -0.01, weight_y < 1.01)
    valid_edge_z = BackendTensor.tfnp.logical_and(weight_z > -0.01, weight_z < 1.01)

    # * Note(miguel) From this point on the arrays become sparse
    xyz_8_edges = BackendTensor.tfnp.hstack([xyz_8[:, 4:], xyz_8[:, [2, 3, 6, 7]], xyz_8[:, 1::2]])
    intersect_segment = BackendTensor.tfnp.hstack([intersect_dx, intersect_dy, intersect_dz])
    valid_edges = BackendTensor.tfnp.hstack([valid_edge_x, valid_edge_y, valid_edge_z])[:, :, 0]
    valid_edges = valid_edges > 0

    intersection_xyz = xyz_8_edges[valid_edges] + intersect_segment[valid_edges]

    return intersection_xyz, valid_edges


# endregion

# region Triangulation Codes
def get_triangulation_codes(octree_list: List[OctreeLevel], options: InterpolationOptions) -> np.ndarray | None:
    """
    Determine the appropriate triangulation codes based on options and octree structure.
    
    Args:
        octree_list: List of octree levels
        options: Interpolation options
        
    Returns:
        Left-right codes array if fancy triangulation is enabled and supported, None otherwise
    """
    is_pure_octree = bool(np.all(octree_list[0].grid_centers.octree_grid_shape == 2))

    match (options.evaluation_options.mesh_extraction_fancy, is_pure_octree):
        case (True, True):
            return get_left_right_array(octree_list)
        case (True, False):
            warnings.warn(
                "Fancy triangulation only works with regular grid of resolution [2,2,2]. "
                "Defaulting to regular triangulation"
            )
            return None
        case (False, _):
            return None
        case _:
            raise ValueError("Invalid combination of options")


def get_masked_codes(left_right_codes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to left-right codes if both are available.
    
    Args:
        left_right_codes: Original left-right codes array
        mask: Boolean mask array
        
    Returns:
        Masked codes if both inputs are not None, otherwise original codes
    """
    if mask is not None and left_right_codes is not None:
        return left_right_codes[mask]
    return left_right_codes


# endregion

# region masking

def mask_generation(
        octree_leaves: OctreeLevel,
        masking_option: MeshExtractionMaskingOptions
) -> np.ndarray:
    """
    Generate masks for mesh extraction based on masking options and stack relations.
    
    Args:
        octree_leaves: Octree leaf level containing scalar field outputs
        masking_option: Mesh extraction masking configuration
        
    Returns:
        Matrix of boolean masks for each scalar field
        
    Raises:
        NotImplementedError: For unsupported masking options
        ValueError: For invalid option combinations
    """
    all_scalar_fields_outputs: List[InterpOutput] = octree_leaves.outputs_centers
    n_scalar_fields = len(all_scalar_fields_outputs)
    outputs_ = all_scalar_fields_outputs[0]
    slice_corners = outputs_.grid.corners_grid_slice
    grid_size = outputs_.cornersGrid_values.shape[0]

    mask_matrix = BackendTensor.t.zeros((n_scalar_fields, grid_size // 8), dtype=bool)
    onlap_chain_counter = 0

    for i in range(n_scalar_fields):
        stack_relation = all_scalar_fields_outputs[i].scalar_fields.stack_relation

        match (masking_option, stack_relation):
            case MeshExtractionMaskingOptions.RAW, _:
                mask_matrix[i] = BackendTensor.t.ones(grid_size // 8, dtype=bool)

            case MeshExtractionMaskingOptions.DISJOINT, _:
                raise NotImplementedError(
                    "Disjoint is not supported yet. Not even sure if there is anything to support"
                )

            case MeshExtractionMaskingOptions.INTERSECT, StackRelationType.ERODE:
                mask_array = all_scalar_fields_outputs[i + onlap_chain_counter].squeezed_mask_array
                x = mask_array[slice_corners].reshape((1, -1, 8))
                mask_matrix[i] = BackendTensor.t.sum(x, -1, bool)[0]
                onlap_chain_counter = 0

            case MeshExtractionMaskingOptions.INTERSECT, StackRelationType.BASEMENT:
                mask_array = all_scalar_fields_outputs[i].squeezed_mask_array
                x = mask_array[slice_corners].reshape((1, -1, 8))
                mask_matrix[i] = BackendTensor.t.sum(x, -1, bool)[0]
                onlap_chain_counter = 0

            case MeshExtractionMaskingOptions.INTERSECT, StackRelationType.ONLAP:
                mask_array = all_scalar_fields_outputs[i].squeezed_mask_array
                x = mask_array[slice_corners].reshape((1, -1, 8))
                mask_matrix[i] = BackendTensor.t.sum(x, -1, bool)[0]
                onlap_chain_counter += 1

            case _, StackRelationType.FAULT:
                mask_matrix[i] = BackendTensor.t.ones(grid_size // 8, dtype=bool)

            case _:
                raise ValueError("Invalid combination of options")

    return mask_matrix


# endregion
def apply_faults_vertex_overlap(all_meshes: list[DualContouringMesh],
                                stack_structure: StacksStructure, 
                                left_right_per_mesh: list[np.ndarray]):
    voxel_overlaps = find_repeated_voxels_across_stacks(left_right_per_mesh)
    
    if voxel_overlaps:
        print(f"Found voxel overlaps between stacks: {voxel_overlaps.keys()}")
        _apply_fault_relations_to_overlaps(all_meshes, voxel_overlaps, stack_structure)
