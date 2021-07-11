import matplotlib.pyplot as plt
import numpy as np
import os

from gempy_engine.core.data.exported_structs import InterpOutput
from gempy_engine.integrations.interp_single.interp_single_interface import interpolate_single_field

dir_name = os.path.dirname(__file__)


def test_export_scalars(simple_model_values_block_output, plot=True, save_sol=False):
    output = simple_model_values_block_output
    Z_x = output.exported_fields.scalar_field
    gx = output.exported_fields.gx_field
    gy = output.exported_fields.gy_field
    gz = output.exported_fields.gz_field
    print(output.weights)

    ids_block = output.ids_block_regular_grid

    if save_sol:
        np.save(dir_name+"/solutions/zx", Z_x)
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
        plt.contourf(Z_x.reshape(50, 5, 50)[:, 2, :].T, N=40, cmap="autumn")
        plt.colorbar()
        plt.contourf(Z_x.reshape(50, 5, 50)[:, 2, :].T, N=40, cmap="autumn",
                     extent=(.25, .75, .25, .75)
                     )

        # plt.quiver(
        #     gx.reshape(50, 5, 50)[:, 2, :].T,
        #     gz.reshape(50, 5, 50)[:, 2, :].T,
        #     scale=10
        # )

        plt.savefig("bar")
        plt.show()


def test_export_simple_model_low_res(simple_model_interpolation_input, plot = True):
    interpolation_input, options, structure = simple_model_interpolation_input

    output: InterpOutput = interpolate_single_field(interpolation_input, options, structure)
    Z_x = output.exported_fields.scalar_field
   # ids_block = output.ids_block
    gx = output.exported_fields.gx_field
    gy = output.exported_fields.gy_field
    gz = output.exported_fields.gz_field

    # one layer weights:
    # [  0.263 -10.335   0.      0.      5.425   4.585
    #   -9.435 -19.55   -2.565  16.013 -12.997   1.39 ]


  #  print(ids_block)
    # np.save(dir_name+"/solutions/test_activator", np.round(ids_block))

    # ids_sol = np.load(dir_name+"/solutions/test_activator.npy")
    # np.testing.assert_almost_equal(
    #     np.round(ids_block),
    #     ids_sol[:, :-7], # NOTE(miguel) Now we only segment on the grid
    #     decimal=3)
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


        # plt.quiver(
        #      gx.reshape(50, 5, 50)[:, 2, :].T,
        #      gz.reshape(50, 5, 50)[:, 2, :].T,
        #      scale=1
        #  )

        plt.savefig("foo2")
        plt.show()



def test_export_3_layers(simple_model_3_layers, plot = True):
    interpolation_input, options, structure = simple_model_3_layers

    output: InterpOutput = interpolate_single_field(interpolation_input, options, structure)
    Z_x = output.exported_fields.scalar_field
   # ids_block = output.ids_block
    gx = output.exported_fields.gx_field
    gy = output.exported_fields.gy_field
    gz = output.exported_fields.gz_field

    # one layer weights:
    # [  0.263 -10.335   0.      0.      5.425   4.585
    #   -9.435 -19.55   -2.565  16.013 -12.997   1.39 ]


  #  print(ids_block)
    # np.save(dir_name+"/solutions/test_activator", np.round(ids_block))

    # ids_sol = np.load(dir_name+"/solutions/test_activator.npy")
    # np.testing.assert_almost_equal(
    #     np.round(ids_block),
    #     ids_sol[:, :-7], # NOTE(miguel) Now we only segment on the grid
    #     decimal=3)
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

        plt.savefig("foo")
        plt.show()



