import numpy as np

from gempy_engine.core.data.centered_grid import CenteredGrid


def calculate_gravity_gradient(centered_grid: CenteredGrid, ugal=True) -> np.ndarray:
    # Extract the voxel center coordinates
    voxel_centers = centered_grid.kernel_grid_centers
    center_x, center_y, center_z = voxel_centers[:, 0], voxel_centers[:, 1], voxel_centers[:, 2]

    # Calculate the coordinates of the voxel corners
    left_edges = centered_grid.left_voxel_edges
    right_edges = centered_grid.right_voxel_edges

    x_corners = np.stack((center_x - left_edges[:, 0], center_x + right_edges[:, 0]), axis=1)
    y_corners = np.stack((center_y - left_edges[:, 1], center_y + right_edges[:, 1]), axis=1)
    z_corners = np.stack((center_z - left_edges[:, 2], center_z + right_edges[:, 2]), axis=1)

    # Prepare coordinates for vector operations
    x_matrix = np.repeat(x_corners, 4, axis=1)
    y_matrix = np.tile(np.repeat(y_corners, 2, axis=1), (1, 2))
    z_matrix = np.tile(z_corners, (1, 4))

    # Calculate the distances from each corner to the observation point
    corner_distances = np.sqrt(x_matrix ** 2 + y_matrix ** 2 + z_matrix ** 2)

    # Factor to determine the sign of each voxel corner
    sign_factor = np.array([1, -1, -1, 1, -1, 1, 1, -1])

    # Choose appropriate gravitational constant
    # ugal     cm3⋅g−1⋅s−26.67408e-2 -- 1 m/s^2 to milligal = 100000 milligal
    # m^3 kg^-1 s^-2
    G = 6.674e-3 if ugal else 6.674_28e-11

    # Calculate the individual components for the sum
    log_term_x = x_matrix * np.log(y_matrix + corner_distances)
    log_term_y = y_matrix * np.log(x_matrix + corner_distances)
    arctan_term = z_matrix * np.arctan(x_matrix * y_matrix / (z_matrix * corner_distances))

    # Combine components and multiply by -1 and the sign factor
    combined_terms = -1 * sign_factor * (log_term_x + log_term_y - arctan_term)

    # Compute the vertical gravity gradient (tz) by summing along axis 1
    tz = G * np.sum(combined_terms, axis=1)

    return tz
