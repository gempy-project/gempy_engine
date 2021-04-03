from gempy_engine.core.backend_tensor import BackendTensor as bt
from gempy_engine.core.data.kernel_classes.orientations import OrientationsInternals


def b_vector_assembly(ori: OrientationsInternals, cov_size: int) -> bt.tensor_types:
    g_s = ori.n_orientations_tiled
    n_dim = ori.orientations.n_dimensions

    if n_dim == 3:
        g_vector = ori.orientations.gx, ori.orientations.gy, ori.orientations.gz
    elif n_dim == 2:
        g_vector = ori.orientations.gx, ori.orientations.gy
    else:
         raise ValueError("Wrong number of dimensions in the gradients.")

    g = bt.tfnp.concat([*g_vector, bt.tfnp.zeros(cov_size - g_s, dtype='float64')], -1)
    g = bt.tfnp.expand_dims(g, axis=1)

    b_vector = g
    return b_vector
