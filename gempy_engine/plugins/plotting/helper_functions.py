import numpy as np

from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.core.data.interpolation_input import InterpolationInput
import matplotlib.pyplot as plt

from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level
from gempy_engine.core.data.output.blocks_value_type import ValueType


def plot_2d_scalar_y_direction(interpolation_input: InterpolationInput, Z_x, grid: RegularGrid = None):
    if grid is None:
        resolution = interpolation_input.grid.octree_grid.resolution
        extent = interpolation_input.grid.octree_grid.orthogonal_extent
    else:
        resolution = grid.resolution
        extent = grid.orthogonal_extent

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
    from gempy_engine.core.backend_tensor import BackendTensor
    xyz = BackendTensor.t.to_numpy(xyz)
    plt.plot(xyz[:, 0], xyz[:, 2], "o")
    plt.colorbar()
    dip_positions = BackendTensor.t.to_numpy(interpolation_input.orientations.dip_positions)
    dip_gradients = BackendTensor.t.to_numpy(interpolation_input.orientations.dip_gradients)
    plt.quiver(
        dip_positions[:, 0],
        dip_positions[:, 2],
        dip_gradients[:, 0],
        dip_gradients[:, 2],
        scale=10
    )


def calculate_gradient(dip, az, pol):
    """Calculates the gradient from dip, azimuth and polarity values."""
    g_x = np.sin(np.deg2rad(dip)) * np.sin(np.deg2rad(az)) * pol
    g_y = np.sin(np.deg2rad(dip)) * np.cos(np.deg2rad(az)) * pol
    g_z = np.cos(np.deg2rad(dip)) * pol
    return g_x, g_y, g_z


def plot_block(block, grid: RegularGrid, interpolation_input=None, direction="y"):
    resolution = tuple(grid.resolution)
    extent = grid.orthogonal_extent
    if direction == "y":
        plt.imshow(block.reshape(resolution)[:, resolution[1] // 2, :].T, extent=extent[[0, 1, 4, 5]], origin="lower")
    if direction == "x":
        plt.imshow(block.reshape(resolution)[resolution[0] // 2, :, :].T, extent=extent[[2, 3, 4, 5]], origin="lower")

    if interpolation_input is not None:
        plot_data(interpolation_input)

    plt.show()


def plot_scalar_and_input_2d(foo, interpolation_input, outputs: list[OctreeLevel], structure: StacksStructure):
    structure.stack_number = foo

    regular_grid_scalar = get_regular_grid_value_for_level(outputs, value_type=ValueType.scalar, scalar_n=foo)
    grid: EngineGrid = outputs[-1].grid_centers

    interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(interpolation_input, structure)
    plot_2d_scalar_y_direction(interpolation_input_i, regular_grid_scalar, grid.octree_grid)


def plot_block_and_input_2d(stack_number, interpolation_input, outputs: list[OctreeLevel],
                            structure: StacksStructure, value_type=ValueType.ids):
    from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level

    regular_grid_scalar = get_regular_grid_value_for_level(
        octree_list=outputs,
        value_type=value_type,
        scalar_n=stack_number
    )
    grid: EngineGrid = outputs[-1].grid_centers

    structure.stack_number = stack_number
    interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(
        all_interpolation_input=interpolation_input,
        stack_structure=structure
    )

    plot_block(regular_grid_scalar, grid.octree_grid, interpolation_input_i)
