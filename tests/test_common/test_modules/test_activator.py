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


def test_activator(simple_model_values_block_output):
    Z_x: np.ndarray = simple_model_values_block_output.exported_fields_dense_grid.scalar_field
    sasp = simple_model_values_block_output.scalar_field_at_sp
    ids = np.array([1, 2])

    print(Z_x, Z_x.shape[0])
    print(sasp)

    ids_block = activate_formation_block(
        exported_fields=simple_model_values_block_output.exported_fields,
        ids=ids, 
        sigmoid_slope=50000)[:, simple_model_values_block_output.grid.dense_grid_slice]
    
    print(ids_block)

    if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
        ids_block = ids_block.detach().numpy()
        Z_x = Z_x.detach().numpy()
        
    if plot:
        plt.contourf(Z_x.reshape(50, 5, 50)[:, 2, :].T, N=40, cmap="autumn")
        plt.colorbar()

        plt.show()

        plt.contourf(
            ids_block[0].reshape(50, 5, 50)[:, 2, :].T,
            N=40,
            cmap="viridis",
            # levels=[-1, 0.5, 1, 1.5, 2.5]
        )
        plt.colorbar()

        plt.show()


# @pytest.mark.skip(reason="This is unfinished I have to extract the 3 layers values")
def test_activator_3_layers(simple_model_3_layers, simple_grid_3d_more_points_grid):
    interpolation_input = simple_model_3_layers[0]
    options = simple_model_3_layers[1]
    data_shape = simple_model_3_layers[2].tensors_structure
    grid = dataclasses.replace(simple_grid_3d_more_points_grid)
    interpolation_input.set_temp_grid(grid)
    
    ids = np.array([1, 2, 3, 4])

    interp_input: SolverInput = input_preprocess(data_shape, interpolation_input)
    weights = _solve_interpolation(interp_input, options.kernel_options)

    exported_fields = _evaluate_sys_eq(interp_input, weights, options)

    exported_fields.set_structure_values(
        reference_sp_position=data_shape.reference_sp_position,
        slice_feature=interpolation_input.slice_feature,
        grid_size=interpolation_input.grid.len_all_grids)

    Z_x: np.ndarray = exported_fields.scalar_field
    sasp = exported_fields.scalar_field_at_surface_points
    ids = np.array([1, 2, 3, 4])

    print(Z_x, Z_x.shape[0])
    print(sasp)

    ids_block = activate_formation_block(
        exported_fields=exported_fields,
        ids= ids,
        sigmoid_slope=50000
    )[0, :-7]

    if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
        ids_block = ids_block.detach().numpy()
        Z_x = Z_x.detach().numpy()
        interpolation_input.surface_points.sp_coords = interpolation_input.surface_points.sp_coords.detach().numpy()
    
    if plot:
        # plt.contourf(Z_x.reshape(50, 5, 50)[:, 0, :].T, N=40, cmap="autumn",
        #              extent=(.25, .75, .25, .75))
        # 
        # xyz = interpolation_input.surface_points.sp_coords
        # plt.plot(xyz[:, 0], xyz[:, 2], "o")
        # plt.colorbar()
        # 
        # plt.show()

        block_ = ids_block[grid.dense_grid_slice]
        plt.contourf(
            block_.reshape(50, 5, 50)[:, 2, :].T, 
            N=250,
            cmap="viridis"
        )
        plt.colorbar()

        plt.show()
