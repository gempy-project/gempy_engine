from dataclasses import dataclass, field

import numpy as np

from .encoders.converters import short_array_type


@dataclass
class CenteredGrid:
    centers: short_array_type  #: This is just used to calculate xyz to interpolate. Tz is independent
    resolution: short_array_type  
    radius: float | short_array_type

    kernel_grid_centers: np.ndarray  = field(init=False)
    left_voxel_edges: np.ndarray  = field(init=False)
    right_voxel_edges: np.ndarray  = field(init=False)

    def __len__(self):
        return self.centers.shape[0] * self.kernel_grid_centers.shape[0]

    def __post_init__(self):
        self.centers = np.atleast_2d(self.centers)

        assert self.centers.shape[1] == 3, 'Centers must be a numpy array of shape (n, 3) that contains the coordinates XYZ'

        self.update_kernels(self.resolution, self.radius)

    def update_kernels(self, grid_resolution, scaling_factor, base_spacing=0.01, z_axis_shift=0.05, z_axis_scale=1.2) -> None:
        """
        Create an isometric grid kernel (centered at 0)
        
        Args:
            grid_resolution: grid resolution in each axis
            scaling_factor: scaling factor in each axis
            base_spacing: base spacing for the grid
            z_axis_shift: shift for the z axis
            z_axis_scale: scale for the z axis
            
        Returns:
            None
        """
        self.kernel_grid_centers, self.left_voxel_edges, self.right_voxel_edges = (
                self.create_irregular_grid_kernel(
                    grid_resolution=grid_resolution,
                    scaling_factor=scaling_factor,
                    base_spacing=base_spacing,
                    z_axis_shift=z_axis_shift,
                    z_axis_scale=z_axis_scale
                )
        )

    @property
    def values(self):
        centers = np.atleast_2d(self.centers)
        values_ = np.empty((0, 3))
        for xyz_device in centers:
            values_ = np.vstack((values_, xyz_device + self.kernel_grid_centers))
        return values_

    @staticmethod
    def create_irregular_grid_kernel(grid_resolution, scaling_factor, base_spacing=0.01, z_axis_shift=0.05, z_axis_scale=1.2):
        if not isinstance(scaling_factor, (list, np.ndarray)):
            scaling_factor = np.repeat(scaling_factor, 3)

        coordinates, voxel_sizes = [], []
        for axis_index in range(3):
            points_in_axis = int(grid_resolution[axis_index] // (1 if axis_index == 2 else 2))
            base_unit_radius = np.geomspace(base_spacing, 1, points_in_axis)

            if axis_index == 2:  # Z-axis
                unit_radius_with_zero = np.concatenate((np.zeros(1), base_unit_radius))
                axis_coordinates = (unit_radius_with_zero + z_axis_shift) * -scaling_factor[axis_index] * z_axis_scale
            else:  # X and Y axes
                unit_radius_with_zero = np.concatenate((-base_unit_radius[::-1], np.zeros(1), base_unit_radius))
                axis_coordinates = unit_radius_with_zero * scaling_factor[axis_index]

            coordinates.append(axis_coordinates)
            padded_coordinates = np.pad(axis_coordinates, 1, 'reflect', reflect_type='odd')
            axis_voxel_sizes = np.diff(padded_coordinates)
            voxel_sizes.append(axis_voxel_sizes)

        left_voxel_edges, right_voxel_edges = [], []
        for sizes in voxel_sizes:
            left_voxel_edges.append(sizes[:-1] / 2)
            right_voxel_edges.append(sizes[1:] / 2)

        grid_centers, left_voxel_edges, right_voxel_edges = np.meshgrid(*coordinates), np.meshgrid(*left_voxel_edges), np.meshgrid(*right_voxel_edges)

        flattened_grid_centers = np.vstack(tuple(map(np.ravel, grid_centers))).T.astype("float64")
        flattened_left_voxel_edges = np.vstack(tuple(map(np.ravel, left_voxel_edges))).T.astype("float64")
        flattened_right_voxel_edges = np.vstack(tuple(map(np.ravel, right_voxel_edges))).T.astype("float64")

        return flattened_grid_centers, flattened_left_voxel_edges, flattened_right_voxel_edges

    def get_number_of_voxels_per_device(self) -> int:
        """
        Calculate the number of voxels in the kernel grid for a single device.

        Returns:
            int: Number of voxels per observation device

        Notes:
            - X and Y axes use symmetric grids: (resolution + 1) points each
            - Z axis uses asymmetric grid (downward only): (resolution + 1) points
            - Total voxels = (rx + 1) × (ry + 1) × (rz + 1)

        Example:
            >>> grid = CenteredGrid(centers=[[500, 500, 600]], 
            ...                      resolution=[10, 10, 10], 
            ...                      radius=[100, 100, 100])
            >>> grid.get_number_of_voxels_per_device()
            1331  # = 11 × 11 × 11
        """
        resolution = np.atleast_1d(self.resolution)

        # Calculate points per axis following the create_irregular_grid_kernel logic
        n_x = int(resolution[0] // 2) * 2 + 1  # Symmetric: 2 * (res//2) + 1
        n_y = int(resolution[1] // 2) * 2 + 1  # Symmetric: 2 * (res//2) + 1
        n_z = int(resolution[2] // 1) + 1  # Asymmetric: res + 1

        return n_x * n_y * n_z

    def get_total_number_of_voxels(self) -> int:
        """
        Calculate the total number of voxels across all observation devices.

        Returns:
            int: Total number of voxels (n_devices × voxels_per_device)

        Example:
            >>> grid = CenteredGrid(centers=[[500, 500, 600], [600, 500, 600]], 
            ...                      resolution=[10, 10, 10], 
            ...                      radius=[100, 100, 100])
            >>> grid.get_total_number_of_voxels()
            2662  # = 2 devices × 1331 voxels
        """
        n_devices = self.centers.shape[0]
        voxels_per_device = self.get_number_of_voxels_per_device()
        return n_devices * voxels_per_device
