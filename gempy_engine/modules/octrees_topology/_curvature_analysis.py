import numpy as np


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

    # Correct slicing based on provided example
    d2f_dx2 = (gx[:, :4] - gx[:, 4:]).sum(axis=1) / (4 * hx**2)
    d2f_dy2 = (gy[:, [0, 1, 4, 5]] - gy[:, [2, 3, 6, 7]]).sum(axis=1) / (4 * hy**2)
    d2f_dz2 = (gz[:, ::2] - gz[:, 1::2]).sum(axis=1) / (4 * hz**2)

    d2f_dxdy = (gx[:, [6, 4, 7, 5]] - gx[:, [2, 0, 3, 1]] - gx[:, [6, 2, 7, 3]] + gx[:, [4, 0, 5, 1]]).sum(axis=1) / (4 * hx * hy)
    d2f_dxdz = (gx[:, [5, 4, 7, 6]] - gx[:, [1, 0, 3, 2]] - gx[:, [5, 1, 7, 3]] + gx[:, [4, 0, 6, 2]]).sum(axis=1) / (4 * hx * hz)
    d2f_dydz = (gy[:, [3, 2, 7, 6]] - gy[:, [1, 0, 5, 4]] - gy[:, [3, 1, 7, 5]] + gy[:, [2, 0, 6, 4]]).sum(axis=1) / (4 * hy * hz)

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
    principal_curvatures = np.abs(np.linalg.eigvals(hessian_matrices))

    return principal_curvatures


def mark_highest_curvature_voxels(gx, gy, gz, voxel_size, curvature_threshold=0.1, verbose: bool = False):
    if gx.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    
    principal_curvatures = compute_curvature(gx, gy, gz, voxel_size)

    curvature_measure = np.sum(principal_curvatures, axis=1)
    
    measure_min = np.min(curvature_measure, axis=0)
    measure_max = np.max(curvature_measure, axis=0)
    if measure_max == measure_min:
        return np.zeros_like(curvature_measure, dtype=bool)
   
    curvature_measure = (curvature_measure - measure_min) / (measure_max - measure_min)

    marked_voxels = curvature_measure > curvature_threshold

    if verbose:
        num_voxels_marked_as_outliers = marked_voxels.sum()
        total_voxels = marked_voxels.size
        print(f"Number of voxels marked as high curvature: {num_voxels_marked_as_outliers} of {total_voxels}")

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
