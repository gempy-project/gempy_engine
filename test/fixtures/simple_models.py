import copy
import dataclasses

import numpy as np

from gempy_engine.config import DEFAULT_DTYPE

np.set_printoptions(precision=3, linewidth=200)

import pytest

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.data_shape import TensorsStructure
from gempy_engine.core.data.exported_structs import InterpOutput

from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.integrations.interp_single._interp_single_internals import _solve_interpolation, \
    _input_preprocess, _evaluate_sys_eq
from gempy_engine.modules.activator.activator_interface import activate_formation_block
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, \
    orientations_preprocess






# simple_grid_3d = np.array([
#     [0.25010, 0.50010, 0.12510],
#     [0.25010, 0.50010, 0.29177],
#     [0.25010, 0.50010, 0.45843],
#     [0.25010, 0.50010, 0.62510],
#     [0.41677, 0.50010, 0.12510],
#     [0.41677, 0.50010, 0.29177],
#     [0.41677, 0.50010, 0.45843],
#     [0.41677, 0.50010, 0.62510],
#     [0.58343, 0.50010, 0.12510],
#     [0.58343, 0.50010, 0.29177],
#     [0.58343, 0.50010, 0.45843],
#     [0.58343, 0.50010, 0.62510],
#     [0.75010, 0.50010, 0.12510],
#     [0.75010, 0.50010, 0.29177],
#     [0.75010, 0.50010, 0.45843],
#     [0.75010, 0.50010, 0.62510]
# ])



# def create_regular_grid(extent, resolution, faces = False):
#     dx = (extent[1] - extent[0]) / resolution[0]
#     dy = (extent[3] - extent[2]) / resolution[1]
#     dz = (extent[5] - extent[4]) / resolution[2]
#
#     x = np.linspace(extent[0] + dx / 2, extent[1] - dx / 2, resolution[0],
#                     dtype="float64")
#     y = np.linspace(extent[2] + dy / 2, extent[3] - dy / 2, resolution[1],
#                     dtype="float64")
#     z = np.linspace(extent[4] + dz / 2, extent[5] - dz / 2, resolution[2],
#                     dtype="float64")
#     xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
#     g = np.vstack((xv.ravel(), yv.ravel(), zv.ravel())).T
#     if faces == False:
#         return g, dx, dy, dz
#     else:
#         x = np.linspace(extent[0], extent[1] , resolution[0] + 1, dtype="float64")
#         y = np.linspace(extent[2], extent[3] , resolution[1]  + 1,  dtype="float64")
#         z = np.linspace(extent[4], extent[5] , resolution[2] + 1,  dtype="float64")
#         xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
#         g_faces = np.vstack((xv.ravel(), yv.ravel(), zv.ravel())).T
#         return g, g_faces, dx, dy, dz










# def tensor_structure_simple_model_2(simple_grid_2d):
#     _ = np.ones(3)
#     return TensorsStructure(number_of_points_per_surface=np.array([4, 3]))


@pytest.fixture(scope='session')
def simple_model_2():

    print(BackendTensor.describe_conf())

    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([4, 3]))

    sp_coords = np.array([[4, 0],
                          [0, 0],
                          [2, 0],
                          [3, 0],
                          [3, 3],
                          [0, 2],
                          [2, 2]], dtype=DEFAULT_DTYPE)

    nugget_effect_scalar = 0
    spi = SurfacePoints(sp_coords, nugget_effect_scalar)

    dip_positions = np.array([[0, 6],
                              [2, 13]], dtype=DEFAULT_DTYPE)

    dip_gradients = np.array([[0, 1], [0, .8]], dtype=DEFAULT_DTYPE)

    nugget_effect_grad = 0.0000001
    ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)

    kri = InterpolationOptions(5, 5 ** 2 / 14 / 3, 0,
                               number_dimensions=2, kernel_function=AvailableKernelFunctions.cubic)

    return spi, ori_i, kri, tensor_struct



@pytest.fixture(scope="session")
def simple_model_2_internals(simple_model_2):
    surface_points = simple_model_2[0]
    orientations = simple_model_2[1]
    options = simple_model_2[2]
    tensors_structure = simple_model_2[3]

    sp_internals = surface_points_preprocess(surface_points, tensors_structure.number_of_points_per_surface)
    ori_internals = orientations_preprocess(orientations)
    return sp_internals, ori_internals, options



@pytest.fixture(scope="session")
def simple_model():

    dip_positions = np.array([
        [0.25010, 0.50010, 0.54177],
        [0.66677, 0.50010, 0.62510],
    ], dtype=DEFAULT_DTYPE)
    sp = np.array([
        [0.25010, 0.50010, 0.37510],
        [0.50010, 0.50010, 0.37510],
        [0.66677, 0.50010, 0.41677],
        [0.70843, 0.50010, 0.47510],
        [0.75010, 0.50010, 0.54177],
        [0.58343, 0.50010, 0.39177],
        [0.73343, 0.50010, 0.50010],
    ], dtype=DEFAULT_DTYPE)

    nugget_effect_scalar = 0
    spi = SurfacePoints(sp, nugget_effect_scalar)

    dip_gradients = np.array([[0, 0, 1],
                              [-.6, 0, .8]], dtype=DEFAULT_DTYPE)
    nugget_effect_grad = 0

    range_ = 4.166666666667
    co = 0.1428571429

    ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)

    kri = InterpolationOptions(range_, co, 0,
                               number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)

    tensor_structure = TensorsStructure(np.array([7]))
    return spi, ori_i, kri, tensor_structure


@pytest.fixture(scope="session")
def simple_model_interpolation_input(simple_grid_3d_octree):
    grid_0_centers = copy.deepcopy(simple_grid_3d_octree)

    dip_positions = np.array([
        [0.25010, 0.50010, 0.54177],
        [0.66677, 0.50010, 0.62510],
    ], dtype=DEFAULT_DTYPE)
    sp = np.array([
        [0.25010, 0.50010, 0.37510],
        [0.50010, 0.50010, 0.37510],
        [0.66677, 0.50010, 0.41677],
        [0.70843, 0.50010, 0.47510],
        [0.75010, 0.50010, 0.54177],
        [0.58343, 0.50010, 0.39177],
        [0.73343, 0.50010, 0.50010],
    ], dtype=DEFAULT_DTYPE)

    nugget_effect_scalar = 0
    spi = SurfacePoints(sp, nugget_effect_scalar)

    dip_gradients = np.array([[0, 0, 1],
                              [-.6, 0, .8]], dtype=DEFAULT_DTYPE)
    nugget_effect_grad = 0

    range_ = 4.166666666667
    co = 0.1428571429

    ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)

    interpolation_options = InterpolationOptions(range_, co, 0,
                                                 number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)

    tensor_structure = TensorsStructure(np.array([7]))

    ids = np.array([1, 2])


    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    yield interpolation_input, interpolation_options, tensor_structure


@pytest.fixture(scope="session")
def simple_model_interpolation_input_optimized(simple_grid_3d_octree):
    grid_0_centers = copy.deepcopy(simple_grid_3d_octree)

    dip_positions = np.array([
        [0.25010, 0.50010, 0.54177],
        [0.66677, 0.50010, 0.62510],
    ], dtype=DEFAULT_DTYPE)

    sp = np.array([
        [0.25010, 0.50010, 0.37510],
        [0.50010, 0.50010, 0.37510],
        [0.66677, 0.50010, 0.41677],
        [0.70843, 0.50010, 0.47510],
        [0.75010, 0.50010, 0.54177],
        [0.58343, 0.50010, 0.39177],
        [0.73343, 0.50010, 0.50010],
    ], dtype=DEFAULT_DTYPE)

    nugget_effect_scalar = 0
    spi = SurfacePoints(sp, nugget_effect_scalar)

    dip_gradients = np.array([[0, 0, 1],
                              [-.6, 0, .8]], dtype=DEFAULT_DTYPE)
    nugget_effect_grad = 0

    range_ = 4.4
    co = 0.1428571429

    ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)

    interpolation_options = InterpolationOptions(range_, co, 0,
                                                 number_dimensions=3, kernel_function=AvailableKernelFunctions.exponential)

    tensor_structure = TensorsStructure(np.array([7]))

    ids = np.array([1, 2])


    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    yield interpolation_input, interpolation_options, tensor_structure



@pytest.fixture(scope="session")
def simple_model_3_layers(simple_grid_3d_octree):
    grid_0_centers = dataclasses.replace(simple_grid_3d_octree)

    np.set_printoptions(precision=3, linewidth=200)

    dip_positions = np.array([
        [0.28010, 0.50010, 0.54177],
        [0.66677, 0.50010, 0.62510],
    ], dtype=DEFAULT_DTYPE)
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

    ], dtype=DEFAULT_DTYPE)

    nugget_effect_scalar = 0
    spi = SurfacePoints(sp, nugget_effect_scalar)

    dip_gradients = np.array([[0, 0, 1],
                              [-.6, 0, .8]], dtype=DEFAULT_DTYPE)
    nugget_effect_grad = 0

    range_ = 4.166666666667
    co = 0.1428571429

    ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)

    interpolation_options = InterpolationOptions(range_, co, 0, i_res=4, gi_res=2,
                               number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)

    tensor_structure = TensorsStructure(np.array([7, 2, 2]))

    ids = np.array([1, 2, 3, 4])

    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    return interpolation_input, interpolation_options, tensor_structure


@pytest.fixture(scope="session")
def simple_model_3_layers_high_res(simple_grid_3d_more_points_grid):
    
    grid_0_centers = dataclasses.replace(simple_grid_3d_more_points_grid)

    np.set_printoptions(precision=3, linewidth=200)

    dip_positions = np.array([
        [0.28010, 0.50010, 0.54177],
        [0.66677, 0.50010, 0.62510],
    ], dtype=DEFAULT_DTYPE)
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
    ],
    dtype=DEFAULT_DTYPE)

    nugget_effect_scalar = 0
    spi = SurfacePoints(sp, nugget_effect_scalar)

    dip_gradients = np.array([[0, 0, 1],
                              [-.6, 0, .8]], dtype=DEFAULT_DTYPE)
    nugget_effect_grad = 0

    range_ = 4.166666666667
    co = 0.1428571429

    ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)

    interpolation_options = InterpolationOptions(range_, co, 0,
                               number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)

    tensor_structure = TensorsStructure(np.array([7, 2, 2]))

    ids = np.array([1, 2, 3, 4])

    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    return interpolation_input, interpolation_options, tensor_structure



@pytest.fixture(scope="session")
def simple_model_values_block_output(simple_model, simple_grid_3d_more_points_grid):
    surface_points = simple_model[0]
    orientations = simple_model[1]
    options = simple_model[2]
    data_shape = simple_model[3]
    grid = dataclasses.replace(simple_grid_3d_more_points_grid)

    ids = np.array([1, 2])

    grid_internal, ori_internal, sp_internal = _input_preprocess(data_shape, grid, orientations,
                                                                 surface_points)
    interp_input = SolverInput(sp_internal, ori_internal, options)

    weights = _solve_interpolation(interp_input)

    exported_fields = _evaluate_sys_eq(grid_internal, interp_input, weights)

    exported_fields.n_points_per_surface = data_shape.nspv
    exported_fields.n_surface_points = surface_points.n_points

    # -----------------
    # Export and Masking operations can happen even in parallel
    # TODO: [~X] Export block
    values_block: np.ndarray = activate_formation_block(exported_fields, ids, sigmoid_slope=50000)

    output = InterpOutput()
    output.grid = grid
    output.exported_fields = exported_fields
    output.weights = weights
    output.values_block = values_block

    return output



# @pytest.fixture(scope="session")
# def simple_model_output(simple_model, simple_grid_3d_more_points_grid):
#     surface_points = simple_model[0]
#     orientations = simple_model[1]
#     options = simple_model[2]
#     data_shape = simple_model[3]
#
#     ids = np.array([1, 2])
#     return interpolate_and_segment(
#         InterpolationInput(surface_points, orientations, simple_grid_3d_more_points_grid, ids),
#         options, data_shape)
