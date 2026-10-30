import itertools

import numpy as np
import warnings
from typing import List

from ._curvature_analysis import mark_highest_curvature_voxels
from ._octree_common import _generate_next_level_centers
from ._neighbor_closure import close_refinement_mask
from ... import config
from ...config import DEBUG_MODE
from ...core.backend_tensor import BackendTensor
from ...core.data.engine_grid import EngineGrid
from ...core.data.exported_fields import ExportedFields
from ...core.data.interp_output import InterpOutput
from ...core.data.octree_level import OctreeLevel
from ...core.data.options.evaluation_options import EvaluationOptions, OctreeRefinementMode
from ...core.data.regular_grid import RegularGrid


def compute_next_octree_locations(prev_octree: OctreeLevel, evaluation_options: EvaluationOptions,
                                  current_octree_level: int, debug: bool = False) -> EngineGrid:
    ids = prev_octree.litho_faults_ids_corners_grid
    uv_8 = ids.reshape((-1, 8))

    # Old octree
    voxel_select = _marked_voxels(uv_8)

    additional_voxel_selected_to_refinement = _additional_refinement_tests(
        voxel_select=voxel_select,
        current_octree_level=current_octree_level,
        evaluation_options=evaluation_options,
        prev_octree=prev_octree
    )

    surface_mask = voxel_select
    primary_mask = surface_mask | additional_voxel_selected_to_refinement
    mode = OctreeRefinementMode(evaluation_options.octree_refinement_mode)
    closure_enabled = (
        mode != OctreeRefinementMode.FAST
        and evaluation_options.mesh_extraction
        and current_octree_level < evaluation_options.number_octree_levels_surface - 1
    )
    support_mask, missing_count = close_refinement_mask(
        prev_octree.grid.octree_grid.integer_coordinates if closure_enabled else None,
        primary_mask, prev_octree.grid.octree_grid.regular_grid_shape,
        mode if closure_enabled else OctreeRefinementMode.FAST
    )
    voxel_select = primary_mask | support_mask
    if missing_count:
        warnings.warn(
            f"Octree level {current_octree_level}: {missing_count} in-domain support cells "
            "are absent from the sparse generation; neighbor closure cannot restore them.",
            RuntimeWarning, stacklevel=2
        )
    
    if compute_topology := False:  # TODO: Fix topology function
        raise NotImplementedError
        prev_octree.edges_id, prev_octree.count_edges = _calculate_topology(
            shift_select_xyz=shift_select_xyz,
            ids=prev_octree.id_block
        )

    # The diagnostics' counts come from one device read, which also sizes the active cells' index.
    diagnostics = debug or DEBUG_MODE or evaluation_options.verbose
    counts = (_host_counts(primary_mask, surface_mask, additional_voxel_selected_to_refinement, support_mask,
                           voxel_select) if diagnostics else None)
    active_index = _active_index(voxel_select, counts[-1] if counts else None)

    # New Octree
    dxdydz = prev_octree.dxdydz
    xyz_anchor = prev_octree.grid.octree_grid.values[active_index]
    xyz_coords, bool_idx = _generate_next_level_centers(xyz_anchor, dxdydz, level=1)

    grid_next_centers = EngineGrid(
        octree_grid=RegularGrid.from_octree_level(
            xyz_coords_octree=xyz_coords,
            previous_regular_grid=prev_octree.grid.octree_grid,
            active_cells=voxel_select,
            left_right=bool_idx,
            active_index=active_index
        ),
    )

    if diagnostics:
        primary_count, surface_count, additional_count, support_count, final_count = counts
        minimum_level = current_octree_level < evaluation_options.octree_min_level
        child_grid = grid_next_centers.octree_grid
        grid_next_centers.octree_grid.refinement_debug = dict(
            primary_surface_mask=surface_mask,
            additional_refinement_mask=additional_voxel_selected_to_refinement,
            support_only_mask=support_mask,
            current_cell_count=len(primary_mask),
            primary_surface_count=surface_count,
            additional_refinement_count=additional_count,
            minimum_level_count=len(primary_mask) if minimum_level else 0,
            curvature_count=0 if minimum_level else additional_count,
            support_only_count=support_count,
            final_refinement_count=final_count,
            closure_multiplier=final_count / primary_count if primary_count else 1.0,
            missing_current_neighbor_count=missing_count,
            generated_child_count=len(xyz_coords),
        )
        if config.EXPENSIVE_DIAGNOSTICS:
            grid_next_centers.octree_grid.refinement_debug['duplicate_corner_count'] = \
                _duplicate_corner_count(child_grid.integer_coordinates)

    if True:
        shift_select_xyz = _mark_voxel(uv_8)[0] if DEBUG_MODE else None
        grid_next_centers.debug_vals = (xyz_coords, xyz_anchor, shift_select_xyz, bool_idx, voxel_select, grid_next_centers)
        return grid_next_centers  # TODO: This is going to break the tests that were using this
    else:
        return grid_next_centers


def _duplicate_corner_count(integer_coordinates) -> int:
    """Stored cell-corner rows minus distinct lattice points they represent (diagnostics only)."""
    coordinates = BackendTensor.t.to_numpy(integer_coordinates).astype(np.int64)
    if len(coordinates) == 0:
        return 0
    coordinates -= coordinates.min(axis=0)
    extent = coordinates.max(axis=0) + 2  # Corners reach one node past the last cell.
    stored = 8 * len(coordinates)
    if np.prod(extent.astype(float)) > 2.0 ** 62:
        corners = (coordinates[:, None, :] + np.array(list(itertools.product((0, 1), repeat=3)))).reshape(-1, 3)
        return stored - len(np.unique(corners, axis=0))
    # One int64 code per lattice node; neighbouring corners are fixed code offsets.
    codes = (coordinates[:, 0] * extent[1] + coordinates[:, 1]) * extent[2] + coordinates[:, 2]
    offsets = [(x * extent[1] + y) * extent[2] + z for x, y, z in itertools.product((0, 1), repeat=3)]
    if np.prod(extent.astype(float)) <= 2 ** 28:
        present = np.zeros(int(np.prod(extent)), dtype=bool)  # Linear-time presence grid.
        for offset in offsets:
            present[codes + offset] = True
        return stored - int(np.count_nonzero(present))
    return stored - len(np.unique((codes[:, None] + np.array(offsets)).reshape(-1)))

def _marked_voxels(uv_8):
    """Cells whose 8 corner ids are not all equal: the cells ``_mark_voxel`` selects, since its 12
    edges connect all corners."""
    return (uv_8 != uv_8[:, :1]).any(1)


def _host_counts(*masks) -> list[int]:
    """The number of selected cells of each mask, from one device read."""
    t = BackendTensor.t
    return [int(n) for n in t.to_numpy(t.stack([mask.sum() for mask in masks]))]


def _active_index(mask, count=None):
    """Positions of the selected cells; with a known ``count`` without reading the mask back."""
    if isinstance(mask, np.ndarray):
        return np.flatnonzero(mask)
    import torch
    if count is not None:
        return torch.nonzero_static(mask, size=count).reshape(-1)
    return torch.nonzero(mask).reshape(-1)


def _mark_voxel(uv_8):
    list_ixd_select = []
    shift_x = uv_8[:, :4] - uv_8[:, 4:]
    shift_y = uv_8[:, [0, 1, 4, 5]] - uv_8[:, [2, 3, 6, 7]]
    shift_z = uv_8[:, ::2] - uv_8[:, 1::2]

    shift_x_select = BackendTensor.t.not_equal(shift_x, 0)
    shift_y_select = BackendTensor.t.not_equal(shift_y, 0)
    shift_z_select = BackendTensor.t.not_equal(shift_z, 0)
    shift_select_xyz = BackendTensor.t.stack([shift_x_select, shift_y_select, shift_z_select])

    idx_select_x = shift_x_select.sum(axis=1, dtype=bool)
    idx_select_y = shift_y_select.sum(axis=1, dtype=bool)
    idx_select_z = shift_z_select.sum(axis=1, dtype=bool)
    list_ixd_select.append(idx_select_x)
    list_ixd_select.append(idx_select_y)
    list_ixd_select.append(idx_select_z)

    voxel_select = (shift_x_select + shift_y_select + shift_z_select).sum(axis=1, dtype=bool)
    return shift_select_xyz, voxel_select


def _calculate_topology(shift_select_xyz: List[np.ndarray], ids: np.ndarray):
    """This is for the typology of level 0. Probably for the rest of octtrees
    levels it will be a bit different
    """
    raise NotImplementedError

    shift_x_select, shift_y_select, shift_z_select = shift_select_xyz

    x_l = ids[1:, :, :][shift_x_select]
    x_r = ids[:-1, :, :][shift_x_select]

    y_l = ids[:, 1:, :][shift_y_select]
    y_r = ids[:, :-1, :][shift_y_select]

    z_l = ids[:, :, 1:][shift_z_select]
    z_r = ids[:, :, :-1][shift_z_select]

    contiguous_voxels = np.vstack([np.hstack((x_l, y_l, z_l)), np.hstack((x_r, y_r, z_r))])
    edges_id, count_edges = np.unique(contiguous_voxels, return_counts=True, axis=1)

    return edges_id, count_edges


def _additional_refinement_tests(voxel_select, current_octree_level, evaluation_options, prev_octree):
    shape = voxel_select.shape[0]

    if current_octree_level < evaluation_options.octree_min_level:
        shape_ = shape
        additional_voxel_selected_to_refinement = BackendTensor.t.ones(shape_, dtype=bool)
        return BackendTensor.t.array(additional_voxel_selected_to_refinement)

    test_for_curvature = 0 <= evaluation_options.octree_curvature_threshold <= 1 and evaluation_options.compute_scalar_gradient

    additional_voxel_selected_to_refinement = BackendTensor.t.zeros(voxel_select.shape, dtype=bool)
    output: InterpOutput
    for output in prev_octree.outputs:
        slicer =output.grid.corners_grid_slice
        exported_fields = output.scalar_fields.exported_fields
        if test_for_curvature:
            additional_voxel_selected_to_refinement |= mark_highest_curvature_voxels(
                gx=(exported_fields.gx_field[slicer].reshape((-1, 8))),
                gy=(exported_fields.gy_field[slicer].reshape((-1, 8))),
                gz=(exported_fields.gz_field[slicer].reshape((-1, 8))),
                voxel_size=np.array(prev_octree.grid.octree_grid.dxdydz),
                curvature_threshold=evaluation_options.octree_curvature_threshold  # * This curvature assumes that 1 is the maximum curvature of any voxel
            )

    return BackendTensor.t.array(additional_voxel_selected_to_refinement)
