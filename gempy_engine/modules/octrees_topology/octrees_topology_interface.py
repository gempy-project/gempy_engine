from typing import List, Optional

import numpy as np

from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.core.data.output.blocks_value_type import ValueType

from ._octree_internals import compute_next_octree_locations
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.engine_grid import EngineGrid


# TODO: [ ] Check if fortran order speeds up this function
# TODO: Substitute numpy for b.tfnp
# TODO: Remove all stack to be able to compile TF


def get_next_octree_grid(prev_octree: OctreeLevel, compute_topology=False) -> EngineGrid:
    # voxels_with_sp_within = _are_surface_points_within_voxel(
    #     points=surface_points.sp_coords,
    #     centers=prev_octree.grid_centers.octree_grid.values,
    #     size=prev_octree.grid_centers.octree_grid.dxdydz,
    #     scale_voxel_to_avoid_false_negatives=2.0 # TODO: Expose this parameter in the options
    # )


    ids = prev_octree.last_output_corners.litho_faults_ids
    corners_scalar_fields = prev_octree.last_output_corners.scalar_fields.exported_fields.scalar_field.reshape((-1, 8))
    corners_scalar_fields_gx = prev_octree.last_output_corners.scalar_fields.exported_fields.gx_field.reshape((-1, 8))
    corners_scalar_fields_gy = prev_octree.last_output_corners.scalar_fields.exported_fields.gy_field.reshape((-1, 8))
    corners_scalar_fields_gz = prev_octree.last_output_corners.scalar_fields.exported_fields.gz_field.reshape((-1, 8))
    # Calculate the variance over axis 1
    foo = compute_curvature(
        gx=corners_scalar_fields_gx,
        gy=corners_scalar_fields_gy,
        gz=corners_scalar_fields_gz,
        voxel_size=np.array(prev_octree.grid_centers.octree_grid.dxdydz)
    )
    
    bar = mark_highest_curvature_voxels(
        gx=corners_scalar_fields_gx,
        gy=corners_scalar_fields_gy,
        gz=corners_scalar_fields_gz,
        voxel_size=np.array(prev_octree.grid_centers.octree_grid.dxdydz),
        curvature_threshold=10
    )
    # corners_scalar_fields_variance = np.var(corners_scalar_fields, axis=1)
    # data_outliers = detect_outliers(
    #     data=corners_scalar_fields_variance,
    #     method='modified_zscore',
    #     threshold=.7
    # )
    # 
    center_scalar_fields = prev_octree.last_output_center.scalar_fields.exported_fields.scalar_field
    scalar_field_sp = prev_octree.last_output_center.scalar_field_at_sp
    

    octree_from_output: EngineGrid = compute_next_octree_locations(
        prev_octree=prev_octree, 
        union_voxel_select=bar,
        # union_voxel_select=None,
        # union_voxel_select=data_outliers,
        compute_topology=compute_topology
    )
    return octree_from_output


def trilinear_interpolation(corner_gradients, point):
    """
    Perform trilinear interpolation of gradients at a given point within the voxel.

    Parameters:
    - corner_gradients: list of gradients at the 8 corners of the voxel
    - point: (x, y, z) coordinates of the point within the voxel

    Returns:
    - interpolated_gradient: interpolated gradient at the given point
    """
    x, y, z = point
    g000, g001, g010, g011, g100, g101, g110, g111 = corner_gradients

    interpolated_gradient = (
            g000 * (1 - x) * (1 - y) * (1 - z) +
            g001 * (1 - x) * (1 - y) * z +
            g010 * (1 - x) * y * (1 - z) +
            g011 * (1 - x) * y * z +
            g100 * x * (1 - y) * (1 - z) +
            g101 * x * (1 - y) * z +
            g110 * x * y * (1 - z) +
            g111 * x * y * z
    )
    return interpolated_gradient


def finite_difference_gradient(gx, gy, gz, voxel_size):
    """
    Estimate the second-order partial derivatives using finite differences for non-isometric voxel sizes.

    Parameters:
    - gx, gy, gz: arrays of shape (number_voxels, 8) containing the gradient components
    - voxel_size: array-like of shape (3,) containing the voxel dimensions in each orthogonal direction

    Returns:
    - hessian_matrices: array of shape (number_voxels, 3, 3) representing the Hessian matrix for each voxel
    """
    hx, hy, hz = voxel_size / 2  # Voxel half-sizes in x, y, z directions
    number_voxels = gx.shape[0]

    d2f_dx2 = (gx[:, 4] - gx[:, 0] + gx[:, 5] - gx[:, 1] + gx[:, 6] - gx[:, 2] + gx[:, 7] - gx[:, 3]) / (4 * hx**2)
    d2f_dy2 = (gy[:, 2] - gy[:, 0] + gy[:, 3] - gy[:, 1] + gy[:, 6] - gy[:, 4] + gy[:, 7] - gy[:, 5]) / (4 * hy**2)
    d2f_dz2 = (gz[:, 1] - gz[:, 0] + gz[:, 3] - gz[:, 2] + gz[:, 5] - gz[:, 4] + gz[:, 7] - gz[:, 6]) / (4 * hz**2)

    d2f_dxdy = (gx[:, 6] - gx[:, 4] - gx[:, 2] + gx[:, 0] + gx[:, 7] - gx[:, 5] - gx[:, 3] + gx[:, 1]) / (4 * hx * hy)
    d2f_dxdz = (gx[:, 5] - gx[:, 4] - gx[:, 1] + gx[:, 0] + gx[:, 7] - gx[:, 6] - gx[:, 3] + gx[:, 2]) / (4 * hx * hz)
    d2f_dydz = (gy[:, 3] - gy[:, 2] - gy[:, 1] + gy[:, 0] + gy[:, 7] - gy[:, 6] - gy[:, 5] + gy[:, 4]) / (4 * hy * hz)

    hessian_matrices = np.zeros((number_voxels, 3, 3))
    hessian_matrices[:, 0, 0] = d2f_dx2
    hessian_matrices[:, 1, 1] = d2f_dy2
    hessian_matrices[:, 2, 2] = d2f_dz2
    hessian_matrices[:, 0, 1] = hessian_matrices[:, 1, 0] = d2f_dxdy
    hessian_matrices[:, 0, 2] = hessian_matrices[:, 2, 0] = d2f_dxdz
    hessian_matrices[:, 1, 2] = hessian_matrices[:, 2, 1] = d2f_dydz

    return hessian_matrices



def compute_curvature(gx, gy, gz, voxel_size):
    """
    Compute the curvature at the center of each voxel.

    Parameters:
    - gx, gy, gz: arrays of shape (number_voxels, 8) containing the gradient components
    - voxel_size: size of the voxel

    Returns:
    - principal_curvatures: array of shape (number_voxels, 3) representing the principal curvatures for each voxel
    """
    hessian_matrices = finite_difference_gradient(gx, gy, gz, voxel_size)
    number_voxels = gx.shape[0]
    principal_curvatures = np.zeros((number_voxels, 3))

    for i in range(number_voxels):
        eigenvalues = np.linalg.eigvals(hessian_matrices[i])
        # principal_curvatures[i] = np.sort(eigenvalues)  # Sort eigenvalues for consistency
        principal_curvatures[i] = np.abs(eigenvalues)  # Sort eigenvalues for consistency

    normalized_curvatures = principal_curvatures / np.prod(voxel_size)

    return normalized_curvatures        


def mark_highest_curvature_voxels(gx, gy, gz, voxel_size, curvature_threshold=0.1):
    principal_curvatures = compute_curvature(gx, gy, gz, voxel_size)

    # Measure curvature using the sum of absolute principal curvatures
    curvature_measure = np.sum(principal_curvatures, axis=1)

    # Get indices of top_n voxels with the highest curvature
    # top_indices = np.argsort(curvature_measure)[-top_n:]

    marked_voxels = curvature_measure > curvature_threshold
    
    return marked_voxels    


def detect_outliers(data, method='zscore', threshold=3.0):
    """
    Detect outliers in a dataset using specified method.

    Parameters:
    - data: list or numpy array of numerical values
    - method: 'zscore', 'modified_zscore', or 'iqr'
    - threshold: threshold value for determining outliers

    Returns:
    - is_outlier: numpy array of booleans where True indicates an outlier
    """
    data = np.array(data)
    is_outlier = np.zeros(data.shape, dtype=bool)

    if method == 'zscore':
        from scipy.stats import zscore
        z_scores = zscore(data)
        is_outlier = np.abs(z_scores) > threshold

    elif method == 'modified_zscore':
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        is_outlier = np.abs(modified_z_scores) > threshold

    elif method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - (threshold * iqr)
        upper_bound = q3 + (threshold * iqr)
        is_outlier = (data < lower_bound) | (data > upper_bound)

    else:
        raise ValueError("Method must be 'zscore', 'modified_zscore', or 'iqr'")

    return is_outlier



def _are_surface_points_within_voxel(points, centers, size, scale_voxel_to_avoid_false_negatives: float = 2.0):
    """
    Determine if a point is within the boundaries of a voxel.

    Parameters:
    point (numpy array of shape (3,)): Coordinates of the point [x, y, z]
    center (numpy array of shape (3,)): Center of the voxel [cx, cy, cz]
    size (numpy array of shape (3,)): Dimensions of the voxel [dx, dy, dz]

    Returns:
    bool: True if the point is within the voxel, False otherwise
    """
    # Calculate the half sizes
    size = np.array(size)
    half_size = size / 2 * scale_voxel_to_avoid_false_negatives

    # Calculate the min and max boundaries of the voxel
    min_bounds = centers - half_size
    max_bounds = centers + half_size

    # Extend point dimensions to (n, 1, 3) and compare with voxel bounds
    points_expanded = points[:, np.newaxis, :]
    within_min = points_expanded >= min_bounds
    within_max = points_expanded <= max_bounds

    # Check if points are within both min and max bounds for all dimensions
    within_bounds_matrix = np.all(within_min & within_max, axis=2)
    
    # Check across all voxels for each point
    within_bounds_voxels = np.any(within_bounds_matrix, axis=0)
    return within_bounds_voxels

def get_regular_grid_value_for_level(octree_list: List[OctreeLevel], level: Optional[int] = None,
                                     value_type: ValueType = ValueType.ids, scalar_n=-1) -> np.ndarray:
    # region Internal Functions ==================================================
    def calculate_oct(shape, n_rep: int) -> np.ndarray:

        f1 = shape[2] * shape[1] * 2 ** (n_rep - 1) * 2 ** (n_rep - 1)
        f2 = shape[2] * 2 ** (n_rep - 1)

        e = 2 ** n_rep

        n_voxel_per_dim = np.arange(e)

        d1 = np.repeat(n_voxel_per_dim, e ** 2) * f1
        d2 = np.tile(np.repeat(n_voxel_per_dim, e), e) * f2
        d3 = np.tile(n_voxel_per_dim, e ** 2)

        oct = d1 + d2 + d3
        return oct

    def get_global_anchor(activ, branch_res, n_rep):
        f1 = branch_res[2] * branch_res[1]
        f2 = branch_res[2]

        f1b = f1 * 2 ** (3 * n_rep)
        f2b = f2 * 2 ** (2 * n_rep)
        f3b = 2 ** (1 * n_rep)

        d1 = (activ // f1) * f1b
        d2 = (activ % f1 // f2) * f2b
        d3 = activ % f1 % f2 * f3b

        anchor = d1 + d2 + d3
        return anchor

    def _expand_regular_grid(active_cells_erg: np.ndarray, n_rep: int) -> np.ndarray:
        active_cells_erg = np.repeat(active_cells_erg, 2 ** n_rep, axis=0)
        active_cells_erg = np.repeat(active_cells_erg, 2 ** n_rep, axis=1)
        active_cells_erg = np.repeat(active_cells_erg, 2 ** n_rep, axis=2)
        return active_cells_erg.ravel()

    def _expand_octree(active_cells_eo: np.ndarray, n_rep: int) -> np.ndarray:
        active_cells_eo = np.repeat(active_cells_eo, 2 ** n_rep, axis=1)
        active_cells_eo = np.repeat(active_cells_eo, 2 ** n_rep, axis=2)
        active_cells_eo = np.repeat(active_cells_eo, 2 ** n_rep, axis=3)

        return active_cells_eo.ravel()

    # endregion =================================================================================
    
    if level is None:
        level = len(octree_list) - 1

    if level > (len(octree_list) - 1):
        raise ValueError("Level cannot be larger than the number of octrees.")

    # Octree - Level 0
    root: OctreeLevel = octree_list[0]

    regular_grid_shape = root.grid_centers.octree_grid_shape

    block = _get_block_from_value_type(root, scalar_n, value_type)
    
    regular_grid: np.ndarray = _expand_regular_grid(
        active_cells_erg=block.reshape(regular_grid_shape.tolist()),
        n_rep=level
    )
    
    shape = regular_grid_shape

    active_cells_index: List["np.ndarray[np.int]"] = []
    global_active_cells_index: np.ndarray = np.array([])

    # Octree - Level n
    for e, octree in enumerate(octree_list[1:level + 1]):
        n_rep = (level - e)
        active_cells: np.ndarray = octree.grid_centers.octree_grid.active_cells

        local_active_cells: np.ndarray = np.where(active_cells)[0]
        shape: np.ndarray = octree.grid_centers.octree_grid_shape
        oct: np.ndarray = calculate_oct(shape, n_rep)

        block = _get_block_from_value_type(octree, scalar_n, value_type)

        ids: np.ndarray = _expand_octree(block.reshape((-1, 2, 2, 2)), n_rep - 1)

        is_branch = e > 0

        if is_branch:
            local_shape: "np.ndarray[np.int]" = np.array([2, 2, 2])
            local_anchors = get_global_anchor(local_active_cells, local_shape, n_rep)
            global_anchors = global_active_cells_index[local_anchors]

            global_active_cells_index = (global_anchors.reshape(-1, 1) + oct).ravel()
            regular_grid[global_active_cells_index] = ids  # + (e * 2)
        else:
            local_shape = shape // 2
            local_anchors = get_global_anchor(local_active_cells, local_shape, n_rep)
            global_active_cells_index = (local_anchors.reshape(-1, 1) + oct).ravel()
            regular_grid[global_active_cells_index] = ids  # + (e * 2)

        active_cells_index.append(global_active_cells_index)

    return regular_grid.reshape(shape.tolist())


def _get_block_from_value_type(root: OctreeLevel, scalar_n: int, value_type: ValueType):
    element_output: InterpOutput = root.outputs_centers[scalar_n]
    block = element_output.get_block_from_value_type(value_type, element_output.grid.octree_grid_slice)
    return block
