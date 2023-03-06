import numpy as np

from gempy_engine.core.data.grid import RegularGrid
from gempy_engine.core.data.interpolation_input import InterpolationInput
import matplotlib.pyplot as plt


def plot_2d_scalar_y_direction(interpolation_input: InterpolationInput, Z_x, grid: RegularGrid = None):
    if grid is None:
        resolution = interpolation_input.grid.regular_grid.resolution
        extent = interpolation_input.grid.regular_grid.extent
    else:
        resolution = grid.resolution
        extent = grid.extent

    plt.contourf(
        Z_x.reshape(resolution)[:, resolution[1] // 2, :].T,
        N=20,
        cmap="autumn",
        extent=extent[[0, 1, 4, 5]]
    )

    plot_data(interpolation_input)

    plt.show()


def plot_data(interpolation_input):
    xyz = interpolation_input.surface_points.sp_coords
    plt.plot(xyz[:, 0], xyz[:, 2], "o")
    plt.colorbar()
    plt.quiver(interpolation_input.orientations.dip_positions[:, 0],
               interpolation_input.orientations.dip_positions[:, 2],
               interpolation_input.orientations.dip_gradients[:, 0],
               interpolation_input.orientations.dip_gradients[:, 2],
               scale=10
               )


def calculate_gradient(dip, az, pol):
    """Calculates the gradient from dip, azimuth and polarity values."""
    g_x = np.sin(np.deg2rad(dip)) * np.sin(np.deg2rad(az)) * pol
    g_y = np.sin(np.deg2rad(dip)) * np.cos(np.deg2rad(az)) * pol
    g_z = np.cos(np.deg2rad(dip)) * pol
    return g_x, g_y, g_z


def plot_block(block, grid: RegularGrid, interpolation_input=None, direction="y"):
    resolution = grid.resolution
    extent = grid.extent
    if direction == "y":
        plt.imshow(block.reshape(resolution)[:, resolution[1] // 2, :].T, extent=extent[[0, 1, 4, 5]], origin="lower")
    if direction == "x":
        plt.imshow(block.reshape(resolution)[resolution[0] // 2, :, :].T, extent=extent[[2, 3, 4, 5]], origin="lower")

    if interpolation_input is not None:
        plot_data(interpolation_input)

    plt.show()
