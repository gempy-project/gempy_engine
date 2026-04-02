from typing import List, Dict, TYPE_CHECKING
import numpy as np
from gempy_engine.config import DualContouringOverlap, DUAL_CONTOURING_VERTEX_OVERLAP
from .apply_mesh_modifications import apply_overlap_to_surface_pair

if TYPE_CHECKING:
    from ...core.data.dual_contouring_mesh import DualContouringMesh
    from ...core.data.stacks_structure import StacksStructure


def apply_relations_to_overlaps(
        all_meshes: List['DualContouringMesh'],
        voxel_overlaps: Dict[str, dict],
        stacks_structure: 'StacksStructure'
) -> None:
    """
    Apply fault, erosion, and onlap relations to voxel overlaps by updating meshes.

    Instead of sharing vertices across meshes, remove triangles on the destination
    surfaces where voxels overlap with the origin surfaces.

    Args:
        all_meshes: List of dual contouring meshes (ordered per-surface)
        voxel_overlaps: Dictionary containing overlap information between per-surface meshes
        stacks_structure: Structure containing relations and stack->surface index mapping
    """
    _apply_fault_logic(all_meshes, voxel_overlaps, stacks_structure)
    if DUAL_CONTOURING_VERTEX_OVERLAP == DualContouringOverlap.watertight:
        _apply_non_fault_overlap_logic(all_meshes, voxel_overlaps, stacks_structure)


def _apply_non_fault_overlap_logic(
        all_meshes: List['DualContouringMesh'],
        voxel_overlaps: Dict[str, dict],
        stacks_structure: 'StacksStructure'
) -> None:
    """
    Apply logic to all overlaps that are not faults.
    Pick one surface as master (the one with lower stack index) and apply the same logic as faults.
    """
    n_stacks = stacks_structure.n_stacks
    surfaces_per_stack = stacks_structure.number_of_surfaces_per_stack_vector
    faults_relations = stacks_structure.faults_relations

    for i in range(n_stacks):
        for j in range(i + 1, n_stacks):
            # Check if this pair is already handled by fault logic
            is_fault = False
            if faults_relations is not None:
                if faults_relations[i, j] or faults_relations[j, i]:
                    is_fault = True

            if is_fault:
                continue

            overlap_key = f"stack_{i}_vs_stack_{j}"
            if overlap_key in voxel_overlaps:
                # We pick stack 'i' as the master (origin) and 'j' as destination
                # If stack i is ERODE, it should be the master.
                # If stack j is ERODE, it should be the master.
                # Default to i as master if no clear relation.

                origin_stack = i
                destination_stack = j

                if stacks_structure.masking_descriptor is not None:
                    # If j is ERODE, it might be younger and should erode i?
                    # In GemPy, younger stacks usually have higher indices.
                    # If j is ERODE, it erodes everything below it (including i).
                    # So j would be the 'master' (origin) and i the 'destination'.
                    from gempy_engine.core.data.stack_relation_type import StackRelationType

                    relation_i = stacks_structure.masking_descriptor[i]
                    relation_j = stacks_structure.masking_descriptor[j]

                    if relation_j == StackRelationType.ERODE:
                        origin_stack = j
                        destination_stack = i
                    elif relation_i == StackRelationType.ERODE:
                        origin_stack = i
                        destination_stack = j
                    # If both are ONLAP, the younger one (j) usually onlaps onto the older one (i).
                    # So i is the master, j is destination.

                origin_surface_range = _get_surface_range(surfaces_per_stack, origin_stack)
                destination_surface_range = _get_surface_range(surfaces_per_stack, destination_stack)

                for origin_surface_idx in origin_surface_range:
                    for destination_surface_idx in destination_surface_range:
                        apply_overlap_to_surface_pair(
                            all_meshes=all_meshes,
                            origin_surface_idx=origin_surface_idx,
                            destination_surface_idx=destination_surface_idx,
                            voxel_overlaps=voxel_overlaps
                        )


def _apply_fault_logic(
        all_meshes: List['DualContouringMesh'],
        voxel_overlaps: Dict[str, dict],
        stacks_structure: 'StacksStructure'
) -> None:
    if stacks_structure.faults_relations is None:
        return

    faults_relations = stacks_structure.faults_relations
    n_stacks = stacks_structure.n_stacks
    surfaces_per_stack = stacks_structure.number_of_surfaces_per_stack_vector

    # For each fault relation, process all pairs of origin (fault) surfaces and destination (affected) surfaces
    for origin_stack, destination_stack in _get_fault_pairs(faults_relations, n_stacks):
        origin_surface_range = _get_surface_range(surfaces_per_stack, origin_stack)
        destination_surface_range = _get_surface_range(surfaces_per_stack, destination_stack)

        for origin_surface_idx in origin_surface_range:
            for destination_surface_idx in destination_surface_range:
                apply_overlap_to_surface_pair(
                    all_meshes=all_meshes,
                    origin_surface_idx=origin_surface_idx,
                    destination_surface_idx=destination_surface_idx,
                    voxel_overlaps=voxel_overlaps
                )


def _get_fault_pairs(faults_relations: np.ndarray, n_stacks: int):
    """Generate pairs of stacks that have fault relations."""
    for origin_stack in range(n_stacks):
        for destination_stack in range(n_stacks):
            if faults_relations[origin_stack, destination_stack]:
                yield origin_stack, destination_stack


def _get_surface_range(surfaces_per_stack: np.ndarray, stack_index: int) -> range:
    """Get the range of surfaces for a given stack."""
    return range(
        surfaces_per_stack[stack_index],
        surfaces_per_stack[stack_index + 1]
    )
