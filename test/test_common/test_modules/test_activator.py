from gempy_engine.core.data.exported_structs import Output
import matplotlib.pyplot as plt

from gempy_engine.modules.activator.activator_interface import activate_formation_block
import numpy as np

plot = True
def test_activator(simple_model_output: Output, tensor_structure):
    Z_x = simple_model_output.exported_fields.scalar_field
    sasp = simple_model_output.scalar_field_at_sp
    ids = np.array([1, 2, 3])

    print(Z_x, Z_x.shape[0])
    print(sasp)

    ids_block = activate_formation_block(
        Z_x,
        sasp,
        ids,
        5000
    )
    print(ids_block)

    if plot:
        plt.contourf(Z_x[:16].reshape(4,1, 4)[:,0,:].T, N=40, cmap="autumn")
        plt.colorbar()

        plt.show()

        foo = ids_block[0][0]
        plt.contourf(foo[:16].reshape(4, 1, 4)[:, 0, :].T, N=40, cmap="viridis")
        plt.colorbar()
        plt.show()

        foo = ids_block[1][0]
        plt.contourf(foo[:16].reshape(4, 1, 4)[:, 0, :].T, N=40, cmap="viridis")
        plt.colorbar()
        plt.show()

        foo = ids_block[2][0]
        plt.contourf(foo[:16].reshape(4, 1, 4)[:, 0, :].T, N=40, cmap="viridis")
        plt.colorbar()
        plt.show()
