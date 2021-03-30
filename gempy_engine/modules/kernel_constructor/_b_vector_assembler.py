from gempy_engine.config import BackendTensor as bt
from gempy_engine.modules.kernel_constructor._structs import OrientationsInternals


def b_vector_assembly(ori: OrientationsInternals, cov_size: int) -> bt.tensor_types:
    g_s = ori.n_orientations_tiled
    g = bt.tfnp.concat([ori.gx_tiled, ori.gy_tiled, ori.gz_tiled,
                        bt.tfnp.zeros(cov_size - g_s, dtype='float64')],
                       -1)
    g = bt.tfnp.expand_dims(g, axis=1)
    b_vector = g
    return b_vector
