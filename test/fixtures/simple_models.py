import numpy as np
import pytest

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.data_shape import TensorsStructure
from gempy_engine.core.data.exported_structs import InterpOutput
from gempy_engine.core.data.grid import Grid
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.integrations.interp_single.interp_single_interface import interpolate_single_scalar, input_preprocess, \
    solve_interpolation, _evaluate_sys_eq, _get_scalar_field_at_surface_points
from gempy_engine.modules.activator.activator_interface import activate_formation_block
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, orientations_preprocess


@pytest.fixture(scope='session')
def simple_grid_2d():
    return simple_grid_2d_f()

def simple_grid_2d_f():
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



def create_regular_grid(extent, resolution):
    dx = (extent[1] - extent[0]) / resolution[0]
    dy = (extent[3] - extent[2]) / resolution[1]
    dz = (extent[5] - extent[4]) / resolution[2]

    x = np.linspace(extent[0] + dx / 2, extent[1] - dx / 2, resolution[0],
                    dtype="float64")
    y = np.linspace(extent[2] + dy / 2, extent[3] - dy / 2, resolution[1],
                    dtype="float64")
    z = np.linspace(extent[4] + dz / 2, extent[5] - dz / 2, resolution[2],
                    dtype="float64")
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    g = np.vstack((xv.ravel(), yv.ravel(), zv.ravel())).T

    return g, dx, dy, dz



@pytest.fixture(scope='session')
def simple_grid_3d_more_points():
    nx, ny, nz = (50, 5, 50)

    x = np.linspace(0.25, 0.75, nx)
    y = np.linspace(0.25, 0.75, ny)
    z = np.linspace(0.25, .75, nz)
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    g = np.vstack((xv.ravel(), yv.ravel(), zv.ravel())).T
    return g


@pytest.fixture(scope="session")
def simple_grid_3d_more_points_grid():
    resolution = [50, 5, 50]
    g, dx, dy, dz = create_regular_grid([0.25, .75, 0.25, .75, 0.25, .75], resolution)

    grid = Grid(g, [g.shape[0]], [50, 5, 50], [dx, dy, dz])
    return grid


def tensor_structure_simple_model_2(simple_grid_2d):
    _ = np.ones(3)
    return TensorsStructure(number_of_points_per_surface=np.array([4, 3]))


@pytest.fixture(scope='session')
def simple_model_2():

    print(BackendTensor.describe_conf())

    tensor_struct = tensor_structure_simple_model_2(simple_grid_2d_f())

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
    tensor_structure = TensorsStructure(np.array([7]))
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
    tensor_structure = TensorsStructure(np.array([7, 2, 2]))
    return spi, ori_i, kri, tensor_structure


@pytest.fixture(scope="session")
def simple_model_values_block_output(simple_model, simple_grid_3d_more_points_grid):
    surface_points = simple_model[0]
    orientations = simple_model[1]
    options = simple_model[2]
    data_shape = simple_model[3]
    grid = simple_grid_3d_more_points_grid

    ids = np.array([1, 2])



    grid_internal, ori_internal, sp_internal = input_preprocess(data_shape, grid, orientations,
                                                                surface_points)
    interp_input = SolverInput(sp_internal, ori_internal, options)

    weights = solve_interpolation(interp_input)

    exported_fields = _evaluate_sys_eq(grid_internal, interp_input, weights)

    scalar_at_surface_points = _get_scalar_field_at_surface_points(
        exported_fields.scalar_field, data_shape.nspv, surface_points.n_points)

    # -----------------
    # Export and Masking operations can happen even in parallel
    # TODO: [~X] Export block
    values_block: np.ndarray = activate_formation_block(exported_fields.scalar_field, scalar_at_surface_points,
                                                         ids, sigmoid_slope=50000)

    output = InterpOutput()
    output.grid = grid
    output.exported_fields = exported_fields
    output.weights = weights
    output.scalar_field_at_sp = scalar_at_surface_points
    output.values_block = values_block

    return output



@pytest.fixture(scope="session")
def simple_model_output(simple_model, simple_grid_3d_more_points_grid):
    surface_points = simple_model[0]
    orientations = simple_model[1]
    options = simple_model[2]
    data_shape = simple_model[3]

    ids = np.array([1, 2])
    return interpolate_single_scalar(surface_points, orientations, simple_grid_3d_more_points_grid,
                                     ids, options, data_shape)
