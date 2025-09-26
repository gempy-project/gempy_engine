import os

import numpy as np
import pandas as pd
import pytest

from gempy_engine.core.data import SurfacePoints, Orientations, InterpolationOptions, TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.plugins.plotting.helper_functions import calculate_gradient

np.set_printoptions(precision=3, linewidth=200)
dir_name = os.path.dirname(__file__)
data_path = dir_name + "/graben_data/"


@pytest.fixture(scope="session")
def one_fault_model():
    centers = np.array([500, 500, -550])
    rescaling_factor = 240

    # region InterpolationInput
    orientations = pd.read_csv(data_path + "Tutorial_ch1-9a_Fault_relations_orientations.csv")
    sp = pd.read_csv(data_path + "Tutorial_ch1-9a_Fault_relations_surface_points.csv")

    sp_coords = (sp[["X", "Y", "Z"]].values - centers) / rescaling_factor
    dip_postions = (orientations[["X", "Y", "Z"]].values - centers) / rescaling_factor
    dip_gradients_ = calculate_gradient(orientations["dip"], orientations["azimuth"], orientations["polarity"])
    dip_gradients = np.vstack(dip_gradients_).T

    spi: SurfacePoints = SurfacePoints(sp_coords)
    ori: Orientations = Orientations(dip_postions, dip_gradients)
    ids = np.array([1, 2, 3, 4, 5, 6, 7])

    resolution = [2, 2, 2]
    extent = np.array([-500, 500., -500, 500, -450, 550]) / rescaling_factor
    regular_grid: RegularGrid = RegularGrid(extent, resolution)
    grid: EngineGrid = EngineGrid(octree_grid=regular_grid)

    interpolation_input: InterpolationInput = InterpolationInput(
        surface_points=spi,
        orientations=ori,
        grid=grid,
        unit_values=ids
    )
    

    # endregion

    # region Structure

    faults_relations = np.array(
        [[False, False, True],
         [False, False, False],
         [False, False, False]
         ]
    )

    stack_structure: StacksStructure = StacksStructure(
        number_of_points_per_stack=np.array([9, 24, 37]),
        number_of_orientations_per_stack=np.array([1, 4, 6]),
        number_of_surfaces_per_stack=np.array([1, 2, 3]),
        masking_descriptor=[StackRelationType.FAULT, StackRelationType.ERODE, StackRelationType.ERODE],
        faults_relations=faults_relations
    )

    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([9, 12, 12, 13, 12, 12]))
    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)
    # endregion

    # region InterpolationOptions

    range_ = 7 ** 2  # ? Since we are not getting the square root should we also square this? 
    c_o = 1

    options = InterpolationOptions.from_args(
        range_, c_o,
        uni_degree=1,
        number_dimensions=3,
        kernel_function=AvailableKernelFunctions.exponential)

    options.cache_mode = InterpolationOptions.CacheMode.NO_CACHE
    # endregion

    return interpolation_input, input_data_descriptor, options


@pytest.fixture(scope="session")
def one_finite_fault_model():
    centers = np.array([500, 500, -550])
    rescaling_factor = 240

    # region InterpolationInput
    orientations = pd.read_csv(data_path + "Tutorial_ch1-9a_Fault_relations_orientations.csv")
    sp = pd.read_csv(data_path + "Tutorial_ch1-9a_Finite_Fault_relations_surface_points.csv")

    sp_coords = (sp[["X", "Y", "Z"]].values - centers) / rescaling_factor
    dip_postions = (orientations[["X", "Y", "Z"]].values - centers) / rescaling_factor
    dip_gradients_ = calculate_gradient(orientations["dip"], orientations["azimuth"], orientations["polarity"])
    dip_gradients = np.vstack(dip_gradients_).T

    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_postions, dip_gradients)
    ids = np.array([1, 2, 3, 4, 5, 6])

    # resolution = [40, 2, 40]
    resolution = [4, 4, 4]
    extent = np.array([-500, 500., -500, 500, -450, 550]) / rescaling_factor
    regular_grid = RegularGrid(extent, resolution)
    grid = EngineGrid(regular_grid.values, regular_grid=regular_grid)

    interpolation_input = InterpolationInput(spi, ori, grid, ids)
    # endregion

    # region Structure

    faults_relations = np.array(
        [[False, False, True],
         [False, False, False],
         [False, False, False]
         ]
    )

    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([9, 24, 37]),
        number_of_orientations_per_stack=np.array([1, 4, 6]),
        number_of_surfaces_per_stack=np.array([1, 2, 3]),
        masking_descriptor=[StackRelationType.FAULT, StackRelationType.ERODE, StackRelationType.ERODE, StackRelationType.BASEMENT],
        faults_relations=faults_relations
    )

    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([9, 12, 12, 13, 12, 12]))
    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)
    # endregion

    # region InterpolationOptions

    # range_ = 1732 / rescaling_factor
    # c_o = 71428.57 / rescaling_factor

    range_ = 7 ** 2  # ? Since we are not getting the square root should we also square this? 
    c_o = 1

    options = InterpolationOptions.from_args(
        range_, c_o,
        uni_degree=1,
        number_dimensions=3,
        kernel_function=AvailableKernelFunctions.exponential)

    # endregion

    return interpolation_input, input_data_descriptor, options


@pytest.fixture(scope="session")
def graben_fault_model():
    centers = np.array([500, 500, -550])
    rescaling_factor = 240

    # region InterpolationInput
    orientations = pd.read_csv(data_path + "Tutorial_ch1-9b_Fault_relations_orientations.csv")

    sp = pd.read_csv(data_path + "Tutorial_ch1-9b_Fault_relations_surface_points.csv")

    sp_coords = (sp[["X", "Y", "Z"]].values - centers) / rescaling_factor
    dip_postions = (orientations[["X", "Y", "Z"]].values - centers) / rescaling_factor
    dip_gradients_ = calculate_gradient(orientations["dip"], orientations["azimuth"], orientations["polarity"])
    dip_gradients = np.vstack(dip_gradients_).T

    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_postions, dip_gradients)
    ids = np.array([1, 2, 3, 4, 5, 6, 7])

    resolution = [2, 2, 2]
    extent = np.array([-500, 500., -500, 500, -450, 550]) / rescaling_factor
    regular_grid = RegularGrid(extent, resolution)
    grid = EngineGrid(octree_grid=regular_grid)

    interpolation_input = InterpolationInput(spi, ori, grid, ids)
    # endregion

    # region Structure

    faults_relations = np.array(
        [[False, True, True],
         [False, False, True],
         [False, False, False]
         ]
    )

    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([9, 9, 54]),
        number_of_orientations_per_stack=np.array([1, 1, 10]),
        number_of_surfaces_per_stack=np.array([1, 1, 4]),
        masking_descriptor=[StackRelationType.FAULT, StackRelationType.FAULT, StackRelationType.BASEMENT],
        faults_relations=faults_relations
    )

    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([9, 9, 15, 15, 12, 12]))
    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)
    # endregion

    # region InterpolationOptions

    # range_ = 1732 / rescaling_factor
    # c_o = 71428.57 / rescaling_factor

    range_ = 7 ** 2  # ? Since we are not getting the square root should we also square this? 
    c_o = 1

    options = InterpolationOptions.from_args(
        range_, c_o,
        uni_degree=1,
        number_dimensions=3,
        kernel_function=AvailableKernelFunctions.exponential)

    # endregion

    return interpolation_input, input_data_descriptor, options
