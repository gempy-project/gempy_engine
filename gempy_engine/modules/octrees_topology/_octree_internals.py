from typing import List, Optional

from ...core.backend_tensor import BackendTensor
from ...core.data.exported_fields import ExportedFields
from ...core.data.interp_output import InterpOutput
from ...core.data.octree_level import OctreeLevel
import numpy as np

from ...core.data.engine_grid import EngineGrid
from ...core.data.options.evaluation_options import EvaluationOptions
from ...core.data.regular_grid import RegularGrid
from ._curvature_analysis import mark_highest_curvature_voxels
from ._octree_common import _generate_next_level_centers
from ...core.data.scalar_field_output import ScalarFieldOutput


def compute_next_octree_locations(prev_octree: OctreeLevel, evaluation_options: EvaluationOptions,
                                  current_octree_level: int) -> EngineGrid:
    ids = prev_octree.last_output_corners.litho_faults_ids
    uv_8 = ids.reshape((-1, 8))

    # Old octree
    shift_select_xyz, voxel_select = _mark_voxel(uv_8)

    additional_voxel_selected_to_refinement = _additional_refinement_tests(
        voxel_select=voxel_select,
        current_octree_level=current_octree_level,
        evaluation_options=evaluation_options,
        prev_octree=prev_octree
    )

    voxel_select = voxel_select | additional_voxel_selected_to_refinement
    
    if compute_topology := False:  # TODO: Fix topology function
        raise NotImplementedError
        prev_octree.edges_id, prev_octree.count_edges = _calculate_topology(
            shift_select_xyz=shift_select_xyz,
            ids=prev_octree.id_block
        )

    # New Octree
    dxdydz = prev_octree.dxdydz
    xyz_anchor = prev_octree.grid_centers.octree_grid.values[voxel_select]
    xyz_coords, bool_idx = _generate_next_level_centers(xyz_anchor, dxdydz, level=1)

    grid_next_centers = EngineGrid(
        octree_grid=RegularGrid.from_octree_level(
            xyz_coords_octree=xyz_coords,
            previous_regular_grid=prev_octree.grid_centers.octree_grid,
            active_cells=voxel_select,
            left_right=bool_idx
        ),
    )

    if True:
        grid_next_centers.debug_vals = (xyz_coords, xyz_anchor, shift_select_xyz, bool_idx, voxel_select, grid_next_centers)
        return grid_next_centers  # TODO: This is going to break the tests that were using this
    else:
        return grid_next_centers


def _mark_voxel(uv_8):
    list_ixd_select = []
    shift_x = uv_8[:, :4] - uv_8[:, 4:]
    shift_y = uv_8[:, [0, 1, 4, 5]] - uv_8[:, [2, 3, 6, 7]]
    shift_z = uv_8[:, ::2] - uv_8[:, 1::2]

    shift_x_select = np.not_equal(shift_x, 0)
    shift_y_select = np.not_equal(shift_y, 0)
    shift_z_select = np.not_equal(shift_z, 0)
    shift_select_xyz = np.array([shift_x_select, shift_y_select, shift_z_select])

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
    exported_fields = prev_octree.last_output_corners.scalar_fields.exported_fields

    if current_octree_level < evaluation_options.min_octree_level:
        shape_ = shape
        additional_voxel_selected_to_refinement = np.ones(shape_, dtype=bool)
        return BackendTensor.t.array(additional_voxel_selected_to_refinement)

    test_for_curvature = 0 <= evaluation_options.curvature_threshold <= 1 and evaluation_options.compute_scalar_gradient
    test_for_error = evaluation_options.error_threshold > 0

    additional_voxel_selected_to_refinement = np.zeros_like(voxel_select)
    output: InterpOutput
    for output in prev_octree.outputs_corners:
        exported_fields = output.scalar_fields.exported_fields
        if test_for_curvature:
            additional_voxel_selected_to_refinement |= mark_highest_curvature_voxels(
                gx=(exported_fields.gx_field.reshape((-1, 8))),
                gy=(exported_fields.gy_field.reshape((-1, 8))),
                gz=(exported_fields.gz_field.reshape((-1, 8))),
                voxel_size=np.array(prev_octree.grid_centers.octree_grid.dxdydz),
                curvature_threshold=evaluation_options.curvature_threshold  # * This curvature assumes that 1 is the maximum curvature of any voxel
            )

        if test_for_error:
            additional_voxel_selected_to_refinement |= _test_refinement_on_stats(
                voxel_select_corners_eval=voxel_select,
                exported_fields=exported_fields,
                plot=evaluation_options.verbose
            )

    return BackendTensor.t.array(additional_voxel_selected_to_refinement)


def _test_refinement_on_stats(exported_fields: ExportedFields, voxel_select_corners_eval, plot=False):
    at_surface_points = exported_fields.scalar_field_at_surface_points
    n_surfaces = at_surface_points.shape[0]
    
    voxel_select_stats = np.zeros_like(voxel_select_corners_eval, dtype=bool)
    scalar_distance = (exported_fields.scalar_field - at_surface_points.reshape(-1, 1)).T
    # TODO: This is only applying the first isosurface of the scalar field. 
    for i in range(0, n_surfaces):
        scalar_distance_8 = scalar_distance[:, i].reshape(-1, 8)

        mean_scalar = scalar_distance_8.mean(axis=1)
        # foo3 = scalar_distance_8[voxel_select_corners_eval].std(axis=1)
        voxel_select_stats |= np.abs(mean_scalar - mean_scalar.mean()) < 1 * mean_scalar.std()

        if plot:
            import matplotlib.pyplot as plt
            # Color to .5 those values that are not in voxel select but are within 2 std of the mean
            c = np.zeros_like(mean_scalar)
            c[voxel_select_stats] = .5
            c[voxel_select_corners_eval] = 1
            # colorbar between 0 and 1
            mean_scalar = BackendTensor.t.to_numpy(mean_scalar)
            plt.scatter( y=mean_scalar, x=range(mean_scalar.size), c=c, cmap='viridis', vmin=0, vmax=1, alpha=.8,)
            plt.colorbar()
            plt.show()
            
    return voxel_select_stats
