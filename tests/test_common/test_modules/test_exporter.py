import copy
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import os

import pytest

from gempy_engine.API.interp_single._multi_scalar_field_manager import interpolate_all_fields
from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.API.interp_single.interp_features import interpolate_single_field
from tests.conftest import TEST_SPEED
from gempy_engine.plugins.plotting.helper_functions import plot_block
from tests.fixtures.grids import simple_grid_3d_octree
from tests.fixtures.simple_models import simple_model_interpolation_input_factory

dir_name = os.path.dirname(__file__)


def test_export_scalars(simple_model_values_block_output, plot=True, save_sol=False):
    output = simple_model_values_block_output
    Z_x = output.exported_fields_dense_grid.scalar_field
    gx = output.exported_fields_dense_grid.gx_field
    gy = output.exported_fields_dense_grid.gy_field
    gz = output.exported_fields_dense_grid.gz_field
    print(output.weights)

    ids_block = output.ids_block_dense_grid

    if save_sol:
        np.save(dir_name + "/solutions/zx", Z_x)
        np.save(dir_name + "/solutions/gx", gx)
        np.save(dir_name + "/solutions/gy", gy)
        np.save(dir_name + "/solutions/gz", gz)

    gx_sol = np.load(dir_name + "/solutions/gx.npy")
    gy_sol = np.load(dir_name + "/solutions/gy.npy")
    gz_sol = np.load(dir_name + "/solutions/gz.npy")

    # np.testing.assert_almost_equal(gx, gx_sol[:-7], decimal=3)
    # np.testing.assert_almost_equal(gy, gy_sol[:-7], decimal=3)
    # np.testing.assert_almost_equal(gz, gz_sol[:-7], decimal=3)

    if plot:
        plt.contourf(Z_x.reshape(50, 5, 50)[:, 2, :].T, N=40, cmap="autumn", extent=(.25, .75, .25, .75))

        xyz = output.grid.dense_grid_values.reshape(-1,3)
        every = 10
        plt.quiver(xyz[::every, 0], xyz[::every, 2], gx[::every], gz[::every], scale=50)

        plt.show()

        plt.contourf(Z_x.reshape(50, 5, 50)[:, 2, :].T, N=40, cmap="autumn", extent=(.25, .75, .25, .75))
        every = 1
        plt.quiver(xyz[::every, 0], xyz[::every, 2], gx[::every], gz[::every], scale=80)

        plt.show()

        plt.contourf(Z_x.reshape(50, 5, 50)[:, 2, :].T, N=40, cmap="autumn")

        gx_np, gz_np = np.gradient(Z_x.reshape(50, 5, 50)[:, 2, :].T)
        plt.quiver(
            gx_np.reshape(50, 50),
            gz_np.reshape(50, 50),
            scale=.5
        )
        plt.show()


def test_export_simple_model_low_res(plot=True):
    interpolation_input, options, structure = simple_model_interpolation_input_factory()
    options.compute_scalar_gradient = True
    
    output: InterpOutput = interpolate_single_field(interpolation_input, options, structure.tensors_structure)
    Z_x = output.exported_fields_dense_grid.scalar_field
    # ids_block = output.ids_block
    gx = output.exported_fields.gx_field
    gy = output.exported_fields.gy_field
    gz = output.exported_fields.gz_field

    if plot:
        plt.contourf(Z_x.reshape(2, 2, 3)[:, 0, :].T, N=40, cmap="autumn",
                     extent=(.25, .75, .25, .75)
                     )

        xyz = interpolation_input.surface_points.sp_coords
        plt.plot(xyz[:, 0], xyz[:, 2], "o")
        plt.colorbar()

        plt.quiver(interpolation_input.orientations.dip_positions[:, 0],
                   interpolation_input.orientations.dip_positions[:, 2],
                   interpolation_input.orientations.dip_gradients[:, 0],
                   interpolation_input.orientations.dip_gradients[:, 2],
                   scale=10
                   )

        xyz = interpolation_input.grid.values
        plt.quiver(xyz[:, 0], xyz[:, 2], gx, gz)

        plt.show()


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_export_3_layers(simple_model_3_layers_high_res, plot=True):
    interpolation_input, options, structure = simple_model_3_layers_high_res
    options.compute_scalar_gradient = True
    output: InterpOutput = interpolate_single_field(interpolation_input, options, structure.tensors_structure)

    Z_x = output.exported_fields_dense_grid.scalar_field
    
    gx = output.exported_fields_dense_grid.gx_field
    gy = output.exported_fields_dense_grid.gy_field
    gz = output.exported_fields_dense_grid.gz_field

    if plot:
        plt.contourf(Z_x.reshape(50, 5, 50)[:, 2, :].T, N=40, cmap="autumn",
                     extent=(.25, .75, .25, .75)
                     )

        xyz = interpolation_input.surface_points.sp_coords
        plt.plot(xyz[:, 0], xyz[:, 2], "o")
        plt.colorbar()

        plt.quiver(interpolation_input.orientations.dip_positions[:, 0],
                   interpolation_input.orientations.dip_positions[:, 2],
                   interpolation_input.orientations.dip_gradients[:, 0],
                   interpolation_input.orientations.dip_gradients[:, 2],
                   scale=10
                   )

        # plt.quiver(
        #      gx.reshape(50, 5, 50)[:, 2, :].T,
        #      gz.reshape(50, 5, 50)[:, 2, :].T,
        #      scale=1
        #  )

        plt.show()


def test_final_exported_fields_one_layer(unconformity_complex_one_layer):
    interpolation_input, options, structure = copy.deepcopy(unconformity_complex_one_layer)
    options.compute_scalar_gradient = True
    
    outputs: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, structure)

    xyz_lvl0 = interpolation_input.grid.values
    resolution = interpolation_input.grid.octree_grid.resolution

    if True:
        plt.quiver(xyz_lvl0[:, 0], xyz_lvl0[:, 2],
                   outputs[0].exported_fields.gx_field,
                   outputs[0].exported_fields.gz_field,
                   pivot="tail",
                   color='green', alpha=.6, )

        grid = interpolation_input.grid.octree_grid
        from gempy_engine.core.backend_tensor import BackendTensor
        plot_block(
            BackendTensor.t.to_numpy(outputs[0].final_exported_fields._scalar_field),
            grid
        )
