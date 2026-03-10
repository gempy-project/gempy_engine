from typing import List, Dict, Optional
import numpy as np

def find_repeated_voxels_across_stacks(all_left_right_codes: List[np.ndarray],
                                       base_numbers: tuple[int, int, int]) -> Dict[str, dict]:
    """
    Find repeated voxels using NumPy operations for efficient processing of large arrays.

    Args:
        all_left_right_codes: List of left_right_codes arrays, one per stack
        base_numbers: tuple of (nx, ny, nz)

    Returns:
        Dictionary with detailed overlap analysis between stack pairs
    """
    if not all_left_right_codes:
        return {}

    stack_codes = _generate_voxel_codes(all_left_right_codes, base_numbers)
    return _find_overlaps_between_stacks(stack_codes, all_left_right_codes)


def _generate_voxel_codes(all_left_right_codes: List[np.ndarray], base_numbers) -> List[np.ndarray]:
    """Generate stable voxel codes for each per-surface mesh using packed bit directions.

    Works for arbitrary base numbers (nx, ny, nz may differ).
    """
    from gempy_engine.modules.dual_contouring.fancy_triangulation import _get_pack_factors

    # Ensure we always compute pack factors once and reuse them for all surfaces
    pack_directions = _get_pack_factors(*base_numbers)

    stack_codes = []
    for left_right_array in all_left_right_codes:
        voxel_codes = (left_right_array * pack_directions).sum(axis=1)
        stack_codes.append(voxel_codes)

    return stack_codes


def _find_overlaps_between_stacks(
        stack_codes: List[np.ndarray],
        all_left_right_codes: List[np.ndarray]
) -> Dict[str, dict]:
    """Find overlaps between all pairs of per-surface meshes (order-sensitive)."""
    overlaps = {}

    for i in range(len(stack_codes)):
        for j in range(i + 1, len(stack_codes)):
            overlap_data = _process_stack_pair(
                codes_i=stack_codes[i],
                codes_j=stack_codes[j],
                left_right_i=all_left_right_codes[i],
                left_right_j=all_left_right_codes[j],
            )

            if overlap_data:
                overlaps[f"stack_{i}_vs_stack_{j}"] = overlap_data

    return overlaps


def _process_stack_pair(
        codes_i: np.ndarray,
        codes_j: np.ndarray,
        left_right_i: np.ndarray,
        left_right_j: np.ndarray,
) -> Optional[dict]:
    """Process a pair of per-surface meshes to find overlapping voxels."""
    if codes_i.size == 0 or codes_j.size == 0:
        return None

    # Use intersect1d with assume_unique=False for robustness; ensure same dtype
    common_codes = np.intersect1d(codes_i, codes_j, assume_unique=False)

    if common_codes.size == 0:
        return None

    # Find indices of common voxels in each surface mesh
    indices_i = np.isin(codes_i, common_codes)
    indices_j = np.isin(codes_j, common_codes)

    return {
            'common_voxel_codes'   : common_codes,
            'count'                : int(common_codes.size),
            'indices_in_stack_i'   : np.where(indices_i)[0],
            'indices_in_stack_j'   : np.where(indices_j)[0],
            'common_binary_codes_i': left_right_i[indices_i],
            'common_binary_codes_j': left_right_j[indices_j]
    }
