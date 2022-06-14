import numpy as np
from gempy_engine.core.data.interpolation_input import InterpolationInput
import matplotlib.pyplot as plt


def plot_2d_scalar_y_direction(interpolation_input: InterpolationInput, Z_x):
    resolution = interpolation_input.grid.regular_grid.resolution
    extent = interpolation_input.grid.regular_grid.extent

    plt.contourf(Z_x.reshape(resolution)[:, resolution[1] // 2, :].T, N=40, cmap="autumn",
                 extent=extent[[0, 1, 4, 5]]
                 )

    xyz = interpolation_input.surface_points.sp_coords
    plt.plot(xyz[:, 0], xyz[:, 2], "o")
    plt.colorbar()

    plt.quiver(interpolation_input.orientations.dip_positions[:, 0],
               interpolation_input.orientations.dip_positions[:, 2],
               interpolation_input.orientations.dip_gradients[:, 0],
               interpolation_input.orientations.dip_gradients[:, 2],
               scale=10
               )

    plt.show()


def calculate_gradient(dip, az, pol):
    """Calculates the gradient from dip, azimuth and polarity values."""
    g_x = np.sin(np.deg2rad(dip)) * np.sin(np.deg2rad(az)) * pol
    g_y = np.sin(np.deg2rad(dip)) * np.cos(np.deg2rad(az)) * pol
    g_z = np.cos(np.deg2rad(dip)) * pol
    return g_x, g_y, g_z


def plot_block(block, grid):
    resolution = grid.resolution
    extent = grid.extent
    plt.imshow(block.reshape(resolution)[:, resolution[1] // 2, :].T, extent=extent[[0, 1, 4, 5]], origin="lower")
    plt.show()