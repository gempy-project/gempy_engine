from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np


@dataclass
class CenteredGrid:
    centers: np.ndarray #: This is just used to calculate xyz to interpolate. Tz is independent
    resolution: Sequence[float]
    radius: Union[float, Sequence[float]]

    cached_kernel_grid_centers: np.ndarray = None
    cached_kernel_left_voxel_edges: np.ndarray = None
    cached_kernel_right_voxel_edges: np.ndarray = None

    def __post_init__(self):
        assert self.centers.shape[1] == 3, 'Centers must be a numpy array that contains the coordinates XYZ'
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
        self.cached_kernel_grid_centers, self.cached_kernel_left_voxel_edges, self.cached_kernel_right_voxel_edges = (
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
            values_ = np.vstack((values_, xyz_device + self.cached_kernel_grid_centers))
        return values_ 

    @staticmethod
    def create_irregular_grid_kernel_(resolution, radius_scale):
        """
        Create an isometric grid kernel (centered at 0)

        Args:
            resolution: [s0]
            radius_scale (float): Maximum distance of the kernel

        Returns:
            tuple: center of the voxel, left edge of each voxel (for xyz), right edge of each voxel (for xyz).
        """

        if radius_scale is not list or radius_scale is not np.ndarray:
            radius_scale = np.repeat(radius_scale, 3)

        coord_list = []
        d_list = []
        for xyz in [0, 1, 2]:
            if xyz == 2:  # * Make the grid only negative for the z axis
                unit_radius = np.geomspace(0.01, 1, int(resolution[xyz]))
                unit_radius_with_0 = np.concatenate((np.zeros(1), unit_radius))
                unit_radius_with_0_shifted = unit_radius_with_0 + 0.05
                z_coord = unit_radius_with_0_shifted * - radius_scale[xyz] * 1.2  # * plus 20%
                coord_list.append(z_coord)
            else:
                unit_radius = np.geomspace(0.01, 1, int(resolution[xyz] / 2))
                unit_radius_with_0 = np.concatenate((-unit_radius[::-1], np.zeros(1), unit_radius))
                x_or_y_coords = unit_radius_with_0 * radius_scale[xyz]
                coord_list.append(x_or_y_coords)

            pad = np.pad(coord_list[xyz], 1, 'reflect', reflect_type='odd')
            diff = np.diff(pad)
            d_list.append(diff)

        g = np.meshgrid(*coord_list)
        d_left = np.meshgrid(d_list[0][:-1] / 2, d_list[1][:-1] / 2, d_list[2][:-1] / 2)
        d_right = np.meshgrid(d_list[0][1:] / 2, d_list[1][1:] / 2, d_list[2][1:] / 2)
        kernel_g = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")
        kernel_d_left = np.vstack(tuple(map(np.ravel, d_left))).T.astype("float64")
        kernel_d_right = np.vstack(tuple(map(np.ravel, d_right))).T.astype("float64")

        return kernel_g, kernel_d_left, kernel_d_right

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

        flattened_grid_centers = np.vstack(map(np.ravel, grid_centers)).T.astype("float64")
        flattened_left_voxel_edges = np.vstack(map(np.ravel, left_voxel_edges)).T.astype("float64")
        flattened_right_voxel_edges = np.vstack(map(np.ravel, right_voxel_edges)).T.astype("float64")

        return flattened_grid_centers, flattened_left_voxel_edges, flattened_right_voxel_edges
