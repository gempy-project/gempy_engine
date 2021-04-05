import numpy as np
import pytest

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.data_shape import TensorsStructure
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.integrations.interp_single.interp_single_interface import interpolate_single_scalar
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, orientations_preprocess


@pytest.fixture(scope='session')
def simple_grid_2d():
    nx, ny = (5, 5)
    x = np.linspace(0, 5, nx)
    y = np.linspace(0, 5, ny)
    xv, yv = np.meshgrid(x, y)
    g = np.vstack((xv.ravel(), yv.ravel())).T
    return g


simple_grid_3d = np.array([
    [0.25010, 0.50010, 0.12510],
    [0.25010, 0.50010, 0.29177],
    [0.25010, 0.50010, 0.45843],
    [0.25010, 0.50010, 0.62510],
    [0.41677, 0.50010, 0.12510],
    [0.41677, 0.50010, 0.29177],
    [0.41677, 0.50010, 0.45843],
    [0.41677, 0.50010, 0.62510],
    [0.58343, 0.50010, 0.12510],
    [0.58343, 0.50010, 0.29177],
    [0.58343, 0.50010, 0.45843],
    [0.58343, 0.50010, 0.62510],
    [0.75010, 0.50010, 0.12510],
    [0.75010, 0.50010, 0.29177],
    [0.75010, 0.50010, 0.45843],
    [0.75010, 0.50010, 0.62510]
])



@pytest.fixture(scope='session')
def simple_grid_3d_more_points():
    nx, ny, nz = (50, 5, 50)

    x = np.linspace(0.25, 0.75, nx)
    y = np.linspace(0.25, 0.75, ny)
    z = np.linspace(0.25, .75, nz)
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    g = np.vstack((xv.ravel(), yv.ravel(), zv.ravel())).T
    return g



@pytest.fixture(scope='session')
def tensor_structure_simple_model_2(simple_grid_2d):
    _ = np.ones(3)
    return TensorsStructure(number_of_points_per_surface=np.array([4, 3]),
                            len_grids=np.atleast_1d(simple_grid_2d.shape[0]))


@pytest.fixture(scope='session')
def simple_model_2(tensor_structure_simple_model_2):
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

    dip_gradients = np.array([[0, 1], [0, .8]])

    nugget_effect_grad = 0.0000001
    ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)

    kri = InterpolationOptions(5, 5 ** 2 / 14 / 3, 0, i_res=1, gi_res=1,
                               number_dimensions=2, kernel_function=AvailableKernelFunctions.cubic)

    return spi, ori_i, kri, tensor_structure_simple_model_2



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
    import numpy

    numpy.set_printoptions(precision=3, linewidth=200)

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

    kri = InterpolationOptions(range_, co, 0, i_res=1, gi_res=1,
                               number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)
    _ = np.ones(3)
    tensor_structure = TensorsStructure(np.array([7]), _)
    return spi, ori_i, kri, tensor_structure


@pytest.fixture(scope="session")
def simple_model_3_layers():
    import numpy

    numpy.set_printoptions(precision=3, linewidth=200)

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
        [0.25010, 0.50010, 0.27510],
        [0.50010, 0.50010, 0.27510],
        [0.25010, 0.50010, 0.17510],
        [0.50010, 0.50010, 0.17510],

    ])

    nugget_effect_scalar = 0
    spi = SurfacePoints(sp, nugget_effect_scalar)

    dip_gradients = np.array([[0, 0, 1],
                              [-.6, 0, .8]])
    nugget_effect_grad = 0

    range_ = 4.166666666667
    co = 0.1428571429

    ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)

    kri = InterpolationOptions(range_, co, 0, i_res=1, gi_res=1,
                               number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)
    _ = np.ones(3)
    tensor_structure = TensorsStructure(np.array([7, 2, 2]), _)
    return spi, ori_i, kri, tensor_structure



@pytest.fixture(scope="session")
def simple_model_output(simple_model, simple_grid_3d_more_points):
    surface_points = simple_model[0]
    orientations = simple_model[1]
    options = simple_model[2]
    data_shape = simple_model[3]

    return interpolate_single_scalar(surface_points, orientations, simple_grid_3d_more_points, options, data_shape)

#
# def test_simple_model_gempy_engine():
#     g = gempy.create_data("test_engine", extent=[-2, 2, -2, 2, -2, 2], resolution=[2, 2, 2])
#     sp = np.array([[4, 0, 0],
#                    [0, 0, 0],
#                    [2, 0, 0],
#                    [3, 0, 0],
#                    [3, 0, 3],
#                    [0, 0, 2],
#                    [2, 0, 2]])
#
#     g.set_default_surfaces()
#
#     for i in sp:
#         g.add_surface_points(*i, surface="surface1")
#
#     g.add_orientations(0, 0, 6, pole_vector=(0, 0, 1), surface="surface1")
#     g.add_orientations(2, 0, 13, pole_vector=(0, 0, .8), surface="surface1")
#
#     g.modify_kriging_parameters("range", 5)
#     g.modify_kriging_parameters("$C_o$", 5 ** 2 / 14 / 3)
#
#     gempy.set_interpolator(g, verbose=["covariance_matrix"])
#
#     print(g.solutions.scalar_field_matrix)
