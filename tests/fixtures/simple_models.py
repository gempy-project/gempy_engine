import copy
import dataclasses
import os
from typing import Tuple

import numpy as np
import pandas as pd

from gempy_engine.API.interp_single._interp_scalar_field import _solve_interpolation, _evaluate_sys_eq
from gempy_engine.API.interp_single._interp_single_feature import input_preprocess
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.interpolation_functions import InterpolationFunctions, CustomInterpolationFunctions
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.plugins.plotting.helper_functions import calculate_gradient

import pytest

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.core.data.scalar_field_output import ScalarFieldOutput

from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.kernel_classes.orientations import Orientations, OrientationsInternals
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints, SurfacePointsInternals
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.modules.activator.activator_interface import activate_formation_block
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, \
    orientations_preprocess


np.set_printoptions(precision=3, linewidth=200)

dir_name = os.path.dirname(__file__)
data_path = dir_name + "/simple_geometries/"


def simple_model_2_factory() -> Tuple[SurfacePoints, Orientations, InterpolationOptions, InputDataDescriptor]:
    print(BackendTensor.describe_conf())
    sp_coords = np.array([[4, 0],
                          [0, 0],
                          [2, 0],
                          [3, 0],
                          [3, 3],
                          [0, 2],
                          [2, 2]])
    nugget_effect_scalar = 0
    spi = SurfacePoints(sp_coords, nugget_effect_scalar)
    dip_positions = np.array([[0, 6],
                              [2, 13]])
    dip_gradients = np.array([[0, 1],
                              [0, .8]])
    nugget_effect_grad = 0.0000001
    ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)
    kri = InterpolationOptions.from_args(5, 5 ** 2 / 14 / 3, 0,
                               number_dimensions=2, kernel_function=AvailableKernelFunctions.cubic)
    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([4, 3]))
    stack_structure = StacksStructure(number_of_points_per_stack=np.array([7]),
                                      number_of_orientations_per_stack=np.array([2]),
                                      number_of_surfaces_per_stack=np.array([2]),
                                      masking_descriptor=[StackRelationType.ERODE])
    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)
    return spi, ori_i, kri, input_data_descriptor


@pytest.fixture(scope='session')
def simple_model_2() -> Tuple[SurfacePoints, Orientations, InterpolationOptions, InputDataDescriptor]:
    return simple_model_2_factory()


@pytest.fixture(scope="session")
def simple_model_2_b() -> Tuple[SurfacePoints, Orientations, InterpolationOptions, InputDataDescriptor]:
    return simple_model_2_factory()


@pytest.fixture(scope="session")
def simple_model_2_internals(simple_model_2) -> Tuple[SurfacePointsInternals, OrientationsInternals, InterpolationOptions]:
    surface_points = simple_model_2[0]
    orientations = simple_model_2[1]
    options = simple_model_2[2]
    input_data_descriptor: InputDataDescriptor = simple_model_2[3]

    sp_internals = surface_points_preprocess(surface_points, input_data_descriptor.tensors_structure)
    ori_internals = orientations_preprocess(orientations)
    return sp_internals, ori_internals, options


@pytest.fixture(scope="session")
def simple_model() -> Tuple[SurfacePoints, Orientations, InterpolationOptions, InputDataDescriptor]:
    def _gen():
        dip_positions = np.array([
                [0.25010, 0.50010, 0.54177],
                [0.66677, 0.50010, 0.62510],
        ])
        sp = np.array([
                [0.25010, 0.50010, 0.37510],
                [0.50010, 0.50010, 0.37510],
                [0.66677, 0.50010, 0.41677],
                [0.70843, 0.50010, 0.47510],
                [0.75010, 0.50010, 0.54177],
                [0.58343, 0.50010, 0.39177],
                [0.73343, 0.50010, 0.50010],
        ])
        nugget_effect_scalar = 0
        spi = SurfacePoints(sp, nugget_effect_scalar)
        dip_gradients = np.array([[0, 0, 1],
                                  [-.6, 0, .8]])
        nugget_effect_grad = 0
        range_ = 4.166666666667
        co = 0.1428571429
        ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)
        kri = InterpolationOptions.from_args(
            range_,
            co,
            0,
            number_dimensions=3,
            kernel_function=AvailableKernelFunctions.cubic
        )
        _ = np.ones(3)
        tensor_struct = TensorsStructure(np.array([7]))
        stack_structure = StacksStructure(
            number_of_points_per_stack=np.array([7]),
            number_of_orientations_per_stack=np.array([2]),
            number_of_surfaces_per_stack=np.array([2]),
            masking_descriptor=[StackRelationType.ERODE])
        input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)
        return input_data_descriptor, kri, ori_i, spi

    input_data_descriptor, kri, ori_i, spi = _gen()

    return spi, ori_i, kri, input_data_descriptor


def simple_model_interpolation_input_factory():
    from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache
    WeightCache.initialize_cache_dir()
    
    resolution = [2, 2, 3]
    extent = [0.25, .75, 0.25, .75, 0.25, .75]

    regular_grid = RegularGrid(extent, resolution)
    grid_0_centers = EngineGrid.from_regular_grid(regular_grid)

    dip_positions = np.array([
            [0.25010, 0.50010, 0.54177],
            [0.66677, 0.50010, 0.62510],
    ])
    sp = np.array([
            [0.25010, 0.50010, 0.37510],
            [0.50010, 0.50010, 0.37510],
            [0.66677, 0.50010, 0.41677],
            [0.70843, 0.50010, 0.47510],
            [0.75010, 0.50010, 0.54177],
            [0.58343, 0.50010, 0.39177],
            [0.73343, 0.50010, 0.50010],
    ])
    nugget_effect_scalar = 0
    spi = SurfacePoints(sp, nugget_effect_scalar)
    dip_gradients = np.array([[0, 0, 1],
                              [-.6, 0, .8]])
    nugget_effect_grad = 0
    range_ = 4.166666666667
    co = 0.1428571429
    ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)
    interpolation_options = InterpolationOptions.from_args(range_, co, 0, number_dimensions=3,
                                                 kernel_function=AvailableKernelFunctions.cubic)
    interpolation_options.cache_mode = InterpolationOptions.CacheMode.NO_CACHE
    ids = np.array([1, 2])
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)
    tensor_struct = TensorsStructure(np.array([7]))
    stack_structure = StacksStructure(number_of_points_per_stack=np.array([7]),
                                      number_of_orientations_per_stack=np.array([2]),
                                      number_of_surfaces_per_stack=np.array([1]),
                                      masking_descriptor=[StackRelationType.ERODE])
    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)
    return interpolation_input, interpolation_options, input_data_descriptor


@pytest.fixture(scope="session")
def simple_model_interpolation_input() -> Tuple[InterpolationInput, InterpolationOptions, InputDataDescriptor]:
    return simple_model_interpolation_input_factory()


@pytest.fixture(scope="session")
def simple_model_3_layers(simple_grid_3d_octree) -> Tuple[InterpolationInput, InterpolationOptions, InputDataDescriptor]:
    input_data_descriptor, interpolation_input, interpolation_options = _gen_simple_model_3_layers(simple_grid_3d_octree)

    return interpolation_input, interpolation_options, input_data_descriptor


def _gen_simple_model_3_layers(simple_grid_3d_octree):

    from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache
    WeightCache.initialize_cache_dir() 
    
    grid_0_centers = dataclasses.replace(simple_grid_3d_octree)
    np.set_printoptions(precision=3, linewidth=200)
    dip_positions = np.array([
            [0.28010, 0.50010, 0.54177],
            [0.66677, 0.50010, 0.62510],
    ])
    sp = np.array([
            [0.25010, 0.50010, 0.37510],
            [0.50010, 0.50010, 0.37510],
            [0.66677, 0.50010, 0.41677],
            [0.70843, 0.50010, 0.47510],
            [0.75010, 0.50010, 0.54177],
            [0.58343, 0.50010, 0.39177],
            [0.73343, 0.50010, 0.50010],
            [0.25010, 0.50010, 0.47510],
            [0.5010, 0.50010, 0.47510],
            [0.25010, 0.50010, 0.6510],
            [0.50010, 0.50010, 0.6510],

    ])
    nugget_effect_scalar = 0
    spi = SurfacePoints(sp, nugget_effect_scalar)
    dip_gradients = np.array([[0, 0, 1],
                              [-.6, 0, .8]])
    nugget_effect_grad = 0
    range_ = 4.166666666667
    co = 0.1428571429
    ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)
    interpolation_options = InterpolationOptions.from_args(range_, co, 0, i_res=4, gi_res=2,
                                                 number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)
    tensor_structure = TensorsStructure(number_of_points_per_surface=np.array([7, 2, 2]))
    stack_structure = StacksStructure(number_of_points_per_stack=np.array([11]),
                                      number_of_orientations_per_stack=np.array([2]),
                                      number_of_surfaces_per_stack=np.array([3]),
                                      masking_descriptor=[StackRelationType.ERODE])
    input_data_descriptor = InputDataDescriptor(tensor_structure, stack_structure)
    ids = np.array([1, 2, 3, 4])
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)
    return input_data_descriptor, interpolation_input, interpolation_options


@pytest.fixture(scope="session")
def simple_model_3_layers_high_res(simple_grid_3d_more_points_grid) -> Tuple[InterpolationInput, InterpolationOptions, InputDataDescriptor]:
    from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache 
    WeightCache.initialize_cache_dir() 
    grid_0_centers = dataclasses.replace(simple_grid_3d_more_points_grid)

    np.set_printoptions(precision=3, linewidth=200)

    dip_positions = np.array([
            [0.28010, 0.50010, 0.54177],
            [0.66677, 0.50010, 0.62510],
    ])
    sp = np.array([
            [0.25010, 0.50010, 0.37510],
            [0.50010, 0.50010, 0.37510],
            [0.66677, 0.50010, 0.41677],
            [0.70843, 0.50010, 0.47510],
            [0.75010, 0.50010, 0.54177],
            [0.58343, 0.50010, 0.39177],
            [0.73343, 0.50010, 0.50010],
            [0.25010, 0.50010, 0.47510],
            [0.5010, 0.50010, 0.47510],
            [0.25010, 0.50010, 0.6510],
            [0.50010, 0.50010, 0.6510],

    ])

    nugget_effect_scalar = 0
    spi = SurfacePoints(sp, nugget_effect_scalar)

    dip_gradients = np.array([[0, 0, 1],
                              [-.6, 0, .8]])
    nugget_effect_grad = 0

    range_ = 4.166666666667
    co = 0.1428571429

    ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)

    interpolation_options = InterpolationOptions.from_args(range_, co, 0,
                                                 number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)
    interpolation_options.cache_mode = InterpolationOptions.CacheMode.NO_CACHE

    ids = np.array([1, 2, 3, 4])

    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    tensor_struct = TensorsStructure(np.array(np.array([7, 2, 2])))
    stack_structure = StacksStructure(number_of_points_per_stack=np.array([11]),
                                      number_of_orientations_per_stack=np.array([2]),
                                      number_of_surfaces_per_stack=np.array([3]),
                                      masking_descriptor=[StackRelationType.ERODE])

    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)

    return interpolation_input, interpolation_options, input_data_descriptor


@pytest.fixture(scope="session")
def simple_model_values_block_output(simple_model, simple_grid_3d_more_points_grid):
    surface_points = simple_model[0]
    orientations = simple_model[1]
    options = simple_model[2]
    data_shape = simple_model[3].tensors_structure
    grid = dataclasses.replace(simple_grid_3d_more_points_grid)
    options.compute_scalar_gradient = True

    ids = np.array([1, 2])
    ii = InterpolationInput(surface_points, orientations, grid, ids)
    interp_input: SolverInput = input_preprocess(data_shape, ii)

    weights = _solve_interpolation(interp_input, options.kernel_options)

    exported_fields = _evaluate_sys_eq(interp_input, weights, options)

    exported_fields.set_structure_values(
        reference_sp_position=data_shape.reference_sp_position,
        slice_feature=ii.slice_feature,
        grid_size=ii.grid.len_all_grids)

    # -----------------
    # Export and Masking operations can happen even in parallel
    # TODO: [~X] Export block
    values_block: np.ndarray = activate_formation_block(exported_fields, ids, sigmoid_slope=50000)

    output = InterpOutput(
        ScalarFieldOutput(
            weights=weights,
            grid=grid,
            exported_fields=exported_fields,
            values_block=values_block,
            stack_relation=ii.stack_relation,
        ),
    )

    return output


@pytest.fixture(scope="session")
def simple_model_3_layers_output(simple_model_3_layers):
    interporlation_input = simple_model_3_layers[0]
    options = simple_model_3_layers[1]
    data_shape = simple_model_3_layers[2].tensors_structure
    ids = np.array([1, 2, 3, 4])

    interp_input: SolverInput = input_preprocess(data_shape, interporlation_input)
    weights = _solve_interpolation(interp_input, options.kernel_options)

    exported_fields = _evaluate_sys_eq(interp_input, weights, options)

    exported_fields.set_structure_values(
        reference_sp_position=data_shape.reference_sp_position,
        slice_feature=interporlation_input.slice_feature,
        grid_size=interporlation_input.grid.len_all_grids)

    # -----------------
    # Export and Masking operations can happen even in parallel
    # TODO: [~X] Export block
    values_block: np.ndarray = activate_formation_block(exported_fields, ids, sigmoid_slope=50000)

    output = InterpOutput(
        ScalarFieldOutput(
            weights=weights,
            grid=interporlation_input.grid,
            exported_fields=exported_fields,
            values_block=values_block,
            stack_relation=interporlation_input.stack_relation,
        ),
    )

    return output


@pytest.fixture(scope="session")
def unconformity_complex():
    return unconformity_complex_factory()


def unconformity_complex_factory():
    orientations = pd.read_csv(data_path + "05_toy_fold_unconformity_orientations.csv")
    sp = pd.read_csv(data_path + "05_toy_fold_unconformity_interfaces.csv")
    sp_coords = sp[["X", "Y", "Z"]].values
    dip_postions = orientations[["X", "Y", "Z"]].values
    dip_gradients_ = calculate_gradient(orientations["dip"],
                                        orientations["azimuth"],
                                        orientations["polarity"])
    dip_gradients = np.vstack(dip_gradients_).T
    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([3, 2, 6]),
        number_of_orientations_per_stack=np.array([2, 1, 6]),
        number_of_surfaces_per_stack=np.array([1, 1, 2]),
        masking_descriptor=[StackRelationType.ERODE, StackRelationType.ERODE, StackRelationType.BASEMENT],
    )
    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([3, 2, 3, 3]))
    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)
    range_ = 0.8660254 * 100
    c_o = 35.71428571 * 100
    i_r = 4
    gi_r = 2
    options = InterpolationOptions.from_args(
        range_, c_o, uni_degree=0, i_res=i_r, gi_res=gi_r,
        number_dimensions=3,
        kernel_function=AvailableKernelFunctions.cubic)
    resolution = [15, 2, 15]
    extent = [0, 10., 0, 2., 0, 5.]
    regular_grid = RegularGrid(extent, resolution)
    grid = EngineGrid(octree_grid=regular_grid)
    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_postions, dip_gradients)
    ids = np.array([0, 1, 2, 3, 4, 5])
    interpolation_input = InterpolationInput(spi, ori, grid, ids)
    return interpolation_input, options, input_data_descriptor


@pytest.fixture(scope="session")
def unconformity_complex_implicit():
    resolution = [15, 2, 15]
    extent = [0, 10., 0, 2., 0, 5.]

    regular_grid = RegularGrid(extent, resolution)
    grid = EngineGrid(octree_grid=regular_grid)

    orientations = pd.read_csv(data_path + "05_toy_fold_unconformity_orientations.csv")
    sp = pd.read_csv(data_path + "05_toy_fold_unconformity_interfaces.csv")

    sp_coords = sp[["X", "Y", "Z"]].values
    dip_postions = orientations[["X", "Y", "Z"]].values
    dip_gradients_ = calculate_gradient(orientations["dip"],
                                        orientations["azimuth"],
                                        orientations["polarity"])
    dip_gradients = np.vstack(dip_gradients_).T

    custom_function = CustomInterpolationFunctions.from_builtin(
        interpolation_function=InterpolationFunctions.SPHERE,
        scalar_field_at_surface_points=np.array([-5]),
        extent=extent)

    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([3, 2, 6]),
        number_of_orientations_per_stack=np.array([2, 1, 6]),
        number_of_surfaces_per_stack=np.array([1, 1, 2]),
        masking_descriptor=[StackRelationType.ERODE, StackRelationType.ERODE, StackRelationType.BASEMENT],
        interp_functions_per_stack=[custom_function, None, None, None])  # * For custom functions we need to add the feature on the DataDescriptor (first element in this model)

    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([3, 2, 3, 3]))
    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)

    range_ = 0.8660254 * 100
    c_o = 35.71428571 * 100
    i_r = 4
    gi_r = 2

    options = InterpolationOptions.from_args(range_, c_o, uni_degree=0, i_res=i_r, gi_res=gi_r,
                                   number_dimensions=3,
                                   kernel_function=AvailableKernelFunctions.cubic)

    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_postions, dip_gradients)
    ids = np.array([0, 1, 2, 3, 4, 5, 6])

    interpolation_input = InterpolationInput(spi, ori, grid, ids)

    return interpolation_input, options, input_data_descriptor


@pytest.fixture(scope="session")
def unconformity_complex_one_layer():
    orientations = pd.read_csv(data_path + "05_toy_fold_unconformity_orientations.csv")
    sp = pd.read_csv(data_path + "05_toy_fold_unconformity_interfaces.csv")

    sp_coords = sp[["X", "Y", "Z"]].values
    dip_postions = orientations[["X", "Y", "Z"]].values
    dip_gradients_ = calculate_gradient(orientations["dip"],
                                        orientations["azimuth"],
                                        orientations["polarity"])
    dip_gradients = np.vstack(dip_gradients_).T

    stack_structure = StacksStructure(number_of_points_per_stack=np.array([3]),
                                      number_of_orientations_per_stack=np.array([2]),
                                      number_of_surfaces_per_stack=np.array([1]),
                                      masking_descriptor=[StackRelationType.BASEMENT])

    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([3]))
    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)

    range_ = 0.8660254 * 100
    c_o = 35.71428571 * 100
    i_r = 4
    gi_r = 2

    options = InterpolationOptions.from_args(range_, c_o, uni_degree=0, i_res=i_r, gi_res=gi_r,
                                   number_dimensions=3,
                                   kernel_function=AvailableKernelFunctions.cubic)

    resolution = [15, 2, 15]
    extent = [0.0000001, 10., 0.0000001, 2., 0.0000001, 5.]

    regular_grid = RegularGrid(extent, resolution)

    grid = EngineGrid(octree_grid=regular_grid)

    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_postions, dip_gradients)
    ids = np.array([0, 1, 2, 3, 4, 5])

    interpolation_input = InterpolationInput(spi, ori, grid, ids)
    return interpolation_input, options, input_data_descriptor
