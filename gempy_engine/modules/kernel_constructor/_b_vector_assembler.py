from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as bt, BackendTensor
from gempy_engine.core.data.kernel_classes.orientations import OrientationsInternals
import numpy as np


def b_vector_assembly(ori: OrientationsInternals, cov_size: int) -> bt.tensor_types:
    g_s = ori.n_orientations_tiled
    n_dim = ori.orientations.n_dimensions

    if n_dim == 3:
        g_vector = ori.orientations.gx, ori.orientations.gy, ori.orientations.gz
    elif n_dim == 2:
        g_vector = ori.orientations.gx, ori.orientations.gy
    else:
        raise ValueError("Wrong number of dimensions in the gradients.")

    zeros = bt.t.zeros(cov_size - g_s, dtype=BackendTensor.dtype_obj)
    b_vector = bt.tfnp.concatenate([*g_vector, zeros], -1)
    b_vector = bt.tfnp.expand_dims(b_vector, axis=1)

    return b_vector
