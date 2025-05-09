import dataclasses
import os

import matplotlib.pyplot as plt
import numpy as np

from gempy_engine.API.interp_single._interp_scalar_field import _solve_interpolation, _evaluate_sys_eq
from gempy_engine.API.interp_single._interp_single_feature import input_preprocess
from gempy_engine.config import AvailableBackends
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.modules.activator.activator_interface import activate_formation_block
from gempy_engine.core.backend_tensor import BackendTensor

dir_name = os.path.dirname(__file__)

plot = True


def test_activator_3_layers_segmentation_function(simple_model_3_layers, simple_grid_3d_more_points_grid):
    Z_x, grid, ids_block, interpolation_input = _run_test(
        backend=AvailableBackends.numpy,
        ids=np.array([1, 20, 3, 4]),
        simple_grid_3d_more_points_grid=simple_grid_3d_more_points_grid,
        simple_model_3_layers=simple_model_3_layers
    )

    if plot:
        _plot_continious(grid, ids_block, interpolation_input)


def test_activator_3_layers_segmentation_function_II(simple_model_3_layers, simple_grid_3d_more_points_grid):
    Z_x, grid, ids_block, interpolation_input = _run_test(
        backend=AvailableBackends.numpy,
        ids=np.array([1, 2, 3, 4]),
        simple_grid_3d_more_points_grid=simple_grid_3d_more_points_grid,
        simple_model_3_layers=simple_model_3_layers
    )

    BackendTensor.change_backend_gempy(AvailableBackends.numpy)

    if plot:
        _plot_continious(grid, ids_block, interpolation_input)


def test_activator_3_layers_segmentation_function_torch(simple_model_3_layers, simple_grid_3d_more_points_grid):
    Z_x, grid, ids_block, interpolation_input = _run_test(
        backend=AvailableBackends.PYTORCH,
        ids=np.array([1, 2, 3, 4]),
        simple_grid_3d_more_points_grid=simple_grid_3d_more_points_grid,
        simple_model_3_layers=simple_model_3_layers
    )

    BackendTensor.change_backend_gempy(AvailableBackends.numpy)
    if plot:
        _plot_continious(grid, ids_block, interpolation_input)


def _run_test(backend, ids, simple_grid_3d_more_points_grid, simple_model_3_layers):
    interpolation_input = simple_model_3_layers[0]
    options = simple_model_3_layers[1]
    data_shape = simple_model_3_layers[2].tensors_structure
    grid = dataclasses.replace(simple_grid_3d_more_points_grid)
    interpolation_input.set_temp_grid(grid)
    interp_input: SolverInput = input_preprocess(data_shape, interpolation_input)
    weights = _solve_interpolation(interp_input, options.kernel_options)
    exported_fields = _evaluate_sys_eq(interp_input, weights, options)
    exported_fields.set_structure_values(
        reference_sp_position=data_shape.reference_sp_position,
        slice_feature=interpolation_input.slice_feature,
        grid_size=interpolation_input.grid.len_all_grids)
    Z_x: np.ndarray = exported_fields.scalar_field
    sasp = exported_fields.scalar_field_at_surface_points
    print(Z_x, Z_x.shape[0])
    print(sasp)
    BackendTensor.change_backend_gempy(backend)
    ids_block = activate_formation_block(
        exported_fields=exported_fields,
        ids=ids,
        sigmoid_slope=500 * 4
    )[0, :-7]
    return Z_x, grid, ids_block, interpolation_input


def _plot_continious(grid, ids_block, interpolation_input):
    block__ = ids_block[grid.dense_grid_slice]
    unique = np.unique(block__)
    t = block__.reshape(50, 5, 50)[:, 2, :].T
    unique = np.unique(t)

    levels = np.linspace(t.min(), t.max(), 40)
    plt.contourf(
        t,
        levels=levels,
        cmap="jet",
        extent=(.25, .75, .25, .75)
    )
    xyz = interpolation_input.surface_points.sp_coords
    plt.plot(xyz[:, 0], xyz[:, 2], "o")
    plt.colorbar()
    plt.show()
