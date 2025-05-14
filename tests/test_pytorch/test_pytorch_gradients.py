import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import SurfacePoints, Orientations, TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.kernel_classes.solvers import Solvers
from gempy_engine.core.data.options import KernelOptions, InterpolationOptions
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure
from ..conftest import plot_pyvista, TEST_SPEED, REQUIREMENT_LEVEL, Requirements

pytestmark = pytest.mark.skipif(REQUIREMENT_LEVEL.value < Requirements.OPTIONAL.value, reason="This test needs higher requirements.")

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
    from gempy_engine.plugins.plotting.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_points, plot_vector
    import torch
except ImportError:
    plot_pyvista = False

import numpy as np


# This is a mock function to represent the unknown behavior of the classes and functions.
def simple_model_interpolation_input_TORCH():
    resolution = [2, 2, 3]
    extent = [0.25, .75, 0.25, .75, 0.25, .75]

    regular_grid = RegularGrid(extent, resolution)
    grid_0_centers = EngineGrid.from_regular_grid(regular_grid)

    dip_positions = torch.tensor([
            [0.25010, 0.50010, 0.54177],
            [0.66677, 0.50010, 0.62510],
    ], dtype=torch.float32, requires_grad=True)

    sp = torch.tensor([
            [0.25010, 0.50010, 0.37510],
            [0.50010, 0.50010, 0.37510],
            [0.66677, 0.50010, 0.41677],
            [0.70843, 0.50010, 0.47510],
            [0.75010, 0.50010, 0.54177],
            [0.58343, 0.50010, 0.39177],
            [0.73343, 0.50010, 0.50010],
    ], dtype=torch.float32, requires_grad=True)

    nugget_effect_scalar = 0

    spi = SurfacePoints(sp, nugget_effect_scalar)
    dip_gradients = torch.tensor(
        [[0, 0, 1],
         [-.6, 0, .8]],
        dtype=torch.float32, requires_grad=True
    )
    nugget_effect_grad = 0
    range_ = 4.166666666667
    co = 0.1428571429

    ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)
    interpolation_options = InterpolationOptions.from_args(range_, co, 0, number_dimensions=3,
                                                 kernel_function=AvailableKernelFunctions.cubic)
    ids = np.array([1, 2])
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)
    tensor_struct = TensorsStructure(np.array([7]))
    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([7]),
        number_of_orientations_per_stack=np.array([2]),
        number_of_surfaces_per_stack=np.array([1]),
        masking_descriptor=[StackRelationType.ERODE]
    )
    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)
    return interpolation_input, interpolation_options, input_data_descriptor


class TestPytorchGradients:
    def __init__(self):
        BackendTensor.change_backend_gempy(
            engine_backend=AvailableBackends.PYTORCH,
            use_gpu=False,
            dtype='float64'
        )

    def test_pytorch_gradients_I(n_oct_levels=3):
        """Kernel function Cubic"""
        interpolation_input, options, structure = simple_model_interpolation_input_TORCH()
        print(interpolation_input)

        options: InterpolationOptions
        options.number_octree_levels = n_oct_levels
        options.mesh_extraction = False
        options.sigmoid_slope = 50
        options.kernel_options.kernel_solver = Solvers.DEFAULT

        coords: torch.tensor = interpolation_input.surface_points.sp_coords
        coords.register_hook(lambda x: print("I am here!", x))

        solutions = compute_model(interpolation_input, options, structure)
        print("last")
        solutions.octrees_output[0].last_output_center.final_block.sum().backward()

        if plot_pyvista or False:
            pv.global_theme.show_edges = True
            p = pv.Plotter()
            plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
            # TODO: Adapt gradients for mesh extraction?
            # plot_dc_meshes(p, solutions.dc_meshes[0])
            p.show()
