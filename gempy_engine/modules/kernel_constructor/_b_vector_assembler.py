from gempy_engine.config import TENSOR_DTYPE
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

    b_vector = bt.tfnp.concatenate([*g_vector, bt.tfnp.zeros(cov_size - g_s, dtype=TENSOR_DTYPE)], -1)
    b_vector = bt.tfnp.expand_dims(b_vector, axis=1)

    return b_vector
