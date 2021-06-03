import matplotlib.pyplot as plt
import numpy as np
import os

dir_name = os.path.dirname(__file__)

def test_export_scalars(simple_model_values_block_output, plot=True, save_sol=False):
    output = simple_model_values_block_output
    Z_x = output.exported_fields.scalar_field
    gx = output.exported_fields.gx_field
    gy = output.exported_fields.gy_field
    gz = output.exported_fields.gz_field

    ids_block = output.ids_block_regular_grid

    if save_sol:
        np.save(dir_name+"/solutions/zx", Z_x)
        np.save(dir_name + "/solutions/gx", gx)
        np.save(dir_name + "/solutions/gy", gy)
        np.save(dir_name + "/solutions/gz", gz)

    gx_sol = np.load(dir_name + "/solutions/gx.npy")
    gy_sol = np.load(dir_name + "/solutions/gy.npy")
    gz_sol = np.load(dir_name + "/solutions/gz.npy")

    np.testing.assert_almost_equal(gx, gx_sol, decimal=3)
    np.testing.assert_almost_equal(gy, gy_sol, decimal=3)
    np.testing.assert_almost_equal(gz, gz_sol, decimal=3)

    if plot:
        plt.contourf(Z_x[:-7].reshape(50, 5, 50)[:, 2, :].T, N=40, cmap="autumn")
        plt.quiver(
            gx[:-7].reshape(50, 5, 50)[:, 2, :].T,
            gz[:-7].reshape(50, 5, 50)[:, 2, :].T,
            scale=10
        )

        plt.colorbar()


        plt.show()

        # plt.contourf(ids_block[:, 2, :].T, N=40, cmap="viridis")
        # plt.colorbar()
        # plt.show()
        #

