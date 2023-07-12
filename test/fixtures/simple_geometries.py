from typing import Tuple

import pandas as pd
import numpy as np
import pytest
import os

from gempy_engine.core.data import TensorsStructure, InterpolationOptions, SurfacePoints, \
    Orientations
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor, StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.core.data.grid import RegularGrid, Grid
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from test.helper_functions import calculate_gradient

dir_name = os.path.dirname(__file__)
data_path = dir_name + "/simple_geometries/"


@pytest.fixture(scope="session")
def horizontal_stratigraphic():
    orientations = pd.read_csv(data_path + "model1_orientations.csv")
    sp = pd.read_csv(data_path + "model1_surface_points.csv")

    sp_coords = sp[["X", "Y", "Z"]].values
    dip_postions = orientations[["X", "Y", "Z"]].values
    dip_gradients_ = calculate_gradient(orientations["dip"],
                                        orientations["azimuth"],
                                        orientations["polarity"])
    dip_gradients = np.vstack(dip_gradients_).T
    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([6, 6]))

    range_ = 1732
    options = InterpolationOptions(range_, range_ ** 2 / 14 / 3, 0, i_res=4, gi_res=2,
                                   number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)

    resolution = [50, 50, 50]
    extent = [0, 1000, 0, 1000, 0, 1000]

    regular_grid = RegularGrid(extent, resolution)
    grid = Grid.from_regular_grid(regular_grid)

    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_postions, dip_gradients)
    ids = np.array([0, 1, 2])

    interpolation_input = InterpolationInput(spi, ori, grid, ids)
    return interpolation_input, options, tensor_struct


@pytest.fixture(scope="session")
def horizontal_stratigraphic_scaled():
    orientations = pd.read_csv(data_path + "model1_orientations_scaled.csv")
    sp = pd.read_csv(data_path + "model1_surface_points_scaled.csv")

    sp_coords = sp[["X", "Y", "Z"]].values
    dip_postions = orientations[["X", "Y", "Z"]].values
    dip_gradients_ = calculate_gradient(orientations["dip"],
                                        orientations["azimuth"],
                                        orientations["polarity"])
    dip_gradients = np.vstack(dip_gradients_).T
    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([6, 6]))

    range_ = 1.083
    c_o = 44.643
    i_r = 4
    gi_r = 2

    # options = InterpolationOptions(range_, range_ ** 2 / 14 / 3, 0, i_res=1, gi_res=1,
    #                            number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)

    options = InterpolationOptions(range_, c_o, 0, i_res=i_r, gi_res=gi_r,
                                   number_dimensions=3,
                                   kernel_function=AvailableKernelFunctions.cubic)

    resolution = [50, 50, 50]
    extent = [0, 1, 0, 1, 0, 1]

    regular_grid = RegularGrid(extent, resolution)

    g = np.load(data_path + "model1_scaled.npy")
    grid = Grid(g, regular_grid=regular_grid)

    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_postions, dip_gradients)
    ids = np.array([0, 1, 2])

    interpolation_input = InterpolationInput(spi, ori, grid, ids)
    return interpolation_input, options, tensor_struct


@pytest.fixture(scope="session")
def recumbent_fold_scaled() -> Tuple[InterpolationInput, InterpolationOptions, TensorsStructure]:
    orientations = pd.read_csv(data_path + "model3_orientations_scaled.csv")
    sp = pd.read_csv(data_path + "model3_surface_points_scaled-2.csv")

    sp_coords = sp[["X", "Y", "Z"]].values
    dip_postions = orientations[["X", "Y", "Z"]].values
    dip_gradients_ = calculate_gradient(orientations["dip"],
                                        orientations["azimuth"],
                                        orientations["polarity"])
    dip_gradients = np.vstack(dip_gradients_).T
    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([55, 50]))

    range_ = 0.8660254
    c_o = 35.71428571
    i_r = 4
    gi_r = 2

    # options = InterpolationOptions(range_, range_ ** 2 / 14 / 3, 0, i_res=1, gi_res=1,
    #                            number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)

    options = InterpolationOptions(range_, c_o, 2, i_res=i_r, gi_res=gi_r,
                                   number_dimensions=3,
                                   kernel_function=AvailableKernelFunctions.cubic)

    resolution = [50, 50, 50]
    extent = [0.3301 - 0.005, .8201 + 0.005,
              0.2551 - 0.005, 0.7451 + 0.005,
              0.2551 - 0.005, 0.7451 + 0.005]

    regular_grid = RegularGrid(extent, resolution)

    grid = Grid(regular_grid.values, regular_grid=regular_grid)

    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_postions, dip_gradients)
    ids = np.array([0, 1, 2])

    interpolation_input = InterpolationInput(spi, ori, grid, ids)
    return interpolation_input, options, tensor_struct


@pytest.fixture(scope="session")
def unconformity() -> Tuple[InterpolationInput, InterpolationOptions, InputDataDescriptor]:
    orientations = pd.read_csv(data_path + "model6_orientations.csv")
    sp = pd.read_csv(data_path + "model6_surface_points.csv")

    sp_coords = sp[["X", "Y", "Z"]].values
    dip_postions = orientations[["X", "Y", "Z"]].values
    dip_gradients_ = calculate_gradient(orientations["dip"],
                                        orientations["azimuth"],
                                        orientations["polarity"])
    dip_gradients = np.vstack(dip_gradients_).T

    stack_structure = StacksStructure(number_of_points_per_stack=np.array([30, 39]),
                                      number_of_orientations_per_stack=np.array([4, 1]),
                                      number_of_surfaces_per_stack=np.array([2, 1]),
                                      masking_descriptor=[StackRelationType.ERODE, StackRelationType.ERODE]
                                      )

    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([18, 12, 9]))
    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)
    
    range_ = 0.8660254 * 1000
    c_o = 35.71428571 * 1000
    i_r = 4
    gi_r = 2

    options = InterpolationOptions(range_, c_o, uni_degree=1, i_res=i_r, gi_res=gi_r,
                                   number_dimensions=3,
                                   kernel_function=AvailableKernelFunctions.cubic)

    resolution = [2, 2, 2]
    extent = [0, 1000, 0, 1000, 0, 1000]

    regular_grid = RegularGrid(extent, resolution)

    grid = Grid(regular_grid.values, regular_grid=regular_grid)

    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_postions, dip_gradients)
    ids = np.array([0, 1, 2, 3])

    interpolation_input = InterpolationInput(spi, ori, grid, ids)
    return interpolation_input, options, input_data_descriptor
