import os

import numpy as np
import pandas as pd
import pytest

from gempy_engine.core.data import SurfacePoints, Orientations, InterpolationOptions
from gempy_engine.core.data.grid import RegularGrid, Grid
from gempy_engine.core.data.input_data_descriptor import StacksStructure, StackRelationType, TensorsStructure, InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from test.helper_functions import calculate_gradient

np.set_printoptions(precision=3, linewidth=200)
dir_name = os.path.dirname(__file__)
data_path = dir_name + "/graben_data/"


@pytest.fixture(scope="session")
def one_fault_model():
    centers = np.array([500, 500, -550])
    rescaling_factor = 1600

    # region InterpolationInput
    orientations = pd.read_csv(data_path + "Tutorial_ch1-9a_Fault_relations_orientations.csv")
    sp = pd.read_csv(data_path + "Tutorial_ch1-9a_Fault_relations_surface_points.csv")

    sp_coords = (sp[["X", "Y", "Z"]].values - centers) / rescaling_factor
    dip_postions = (orientations[["X", "Y", "Z"]].values - centers) / rescaling_factor
    dip_gradients_ = calculate_gradient(orientations["dip"], orientations["azimuth"], orientations["polarity"])
    dip_gradients = np.vstack(dip_gradients_).T

    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_postions, dip_gradients)
    ids = np.array([0, 1, 2, 3, 4, 5])

    resolution = [20, 10, 20]
    extent = np.array([-500, 500., -500, 500, -450, 550]) / rescaling_factor
    regular_grid = RegularGrid(extent, resolution)
    grid = Grid(regular_grid.values, regular_grid=regular_grid)

    interpolation_input = InterpolationInput(spi, ori, grid, ids)
    # endregion

    # region Structure
    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([9, 24, 37]),
        number_of_orientations_per_stack=np.array([1, 4, 6]),
        number_of_surfaces_per_stack=np.array([1, 2, 3]),
        masking_descriptor=[StackRelationType.FAULT, StackRelationType.ERODE, StackRelationType.ERODE, False],
    )

    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([9, 12, 12, 13, 12, 12]))
    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)
    # endregion

    # region InterpolationOptions

    range_ = 1732 / rescaling_factor
    c_o = 71428.57 / rescaling_factor

    options = InterpolationOptions(
        range_, c_o,
        uni_degree=1,
        number_dimensions=3,
        kernel_function=AvailableKernelFunctions.cubic)

    # endregion

    return interpolation_input, input_data_descriptor, options
