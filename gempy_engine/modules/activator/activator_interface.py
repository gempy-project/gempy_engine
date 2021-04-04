from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.backend_tensor import BackendTensor as bt
import numpy as np

def activate_formation_block(Z_x: np.ndarray,
                             scalar_value_at_sp:np.ndarray,
                             ids: np.ndarray,
                             sigmoid_slope:float):

    for i in range(scalar_value_at_sp.size):
        sigm = _compute_sigmoid(Z_x, i, ids, scalar_value_at_sp, sigmoid_slope)

    return sigm


def _compute_sigmoid(Z_x, i, ids, scalar_value_at_sp, sigmoid_slope):
    n_surface_0 = ids[i]
    n_surface_1 = ids[i + 1]
    a = scalar_value_at_sp[i]
    b = scalar_value_at_sp[i + 1]
    sigm = (-n_surface_0.reshape((-1, 1)) / (1 + bt.tfnp.exp(-sigmoid_slope * (Z_x - a)))) - \
           (n_surface_1.reshape((-1, 1)) / (1 + bt.tfnp.exp(sigmoid_slope * (Z_x - b)))) + \
           n_surface_0.reshape((-1, 1))
    return sigm