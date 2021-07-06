import os

from gempy_engine.core.data.exported_structs import InterpOutput
import matplotlib.pyplot as plt

from gempy_engine.modules.activator.activator_interface import activate_formation_block
import numpy as np

dir_name = os.path.dirname(__file__)

plot = True
def test_activator(simple_model_values_block_output):
    Z_x = simple_model_values_block_output.exported_fields.scalar_field
    sasp = simple_model_values_block_output.scalar_field_at_sp
    ids = np.array([1, 2])


    print(Z_x, Z_x.shape[0])
    print(sasp)

    ids_block = activate_formation_block(simple_model_values_block_output.exported_fields,
                                         ids, 50000)
    print(ids_block)
    # np.save(dir_name+"/solutions/test_activator", np.round(ids_block))

    ids_sol = np.load(dir_name+"/solutions/test_activator.npy")
    np.testing.assert_almost_equal(
        np.round(ids_block),
        ids_sol[:, :-7], # NOTE(miguel) Now we only segment on the grid
        decimal=3)
    if plot:
        plt.contourf(Z_x.reshape(50, 5, 50)[:, 2, :].T, N=40, cmap="autumn")
        plt.colorbar()
        plt.show()

        plt.contourf(ids_block[0].reshape(50, 5, 50)[:, 2, :].T, N=40, cmap="viridis")
        plt.colorbar()
        plt.show()
