from gempy_engine.core.data.exported_structs import Output
import matplotlib.pyplot as plt

plot = False
def test_activator(simple_model_output: Output, tensor_structure):
    Z_x = simple_model_output.exported_fields.scalar_field
    sasp = simple_model_output.scalar_field_at_sp

    print(Z_x, Z_x.shape[0])
    print(sasp)

    if plot:
        plt.contourf(Z_x[:16].reshape(4,1, 4)[:,0,:].T, N=40, cmap="autumn")
        plt.show()