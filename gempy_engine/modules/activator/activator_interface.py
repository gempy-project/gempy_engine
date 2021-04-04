from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.backend_tensor import BackendTensor as bt
import numpy as np

def activate_formation_block(Z_x: np.ndarray,
                             scalar_value_at_sp:np.ndarray,
                             ids: np.ndarray,
                             sigmoid_slope:float):
    def _compute_sigmoid(Z_x, scale_0, scale_1, drift_0, drift_1, sigmoid_slope):

        active_sig   = -scale_0.reshape((-1, 1)) / (1 + bt.tfnp.exp(-sigmoid_slope * (Z_x - drift_0)))
        deactive_sig = -scale_1.reshape((-1, 1)) / (1 + bt.tfnp.exp( sigmoid_slope * (Z_x - drift_1)))
        activation_sig = active_sig + deactive_sig

        sigm = activation_sig + scale_0.reshape((-1, 1))
        return sigm

    drift_0_v = bt.tfnp.concatenate([np.array([0]), scalar_value_at_sp, scalar_value_at_sp[-1:]])
    drift_1_v = bt.tfnp.concatenate([scalar_value_at_sp[:0], scalar_value_at_sp, np.array([0])])

    scalar_0_v = bt.tfnp.concatenate([np.array([0]), ids, ids[-1:]])
    scalar_1_v = bt.tfnp.concatenate([ids[:0], ids, np.array([0])])

    ids_iter = bt.tfnp.repeat(ids, 2, axis=0)
    ids_iter[0] = 0
    ids_iter[-1] = 0
    # drift = bt.tfnp.repeat(ids, 2, axis=0)
    # drift[-1] = 0

    sigm = np.zeros((1, Z_x.shape[0]))
    sigm = []
    for i in range(ids.size):
        sigm.append(_compute_sigmoid(Z_x, scalar_0_v[i], scalar_1_v[i], drift_0_v[i], drift_1_v[i], sigmoid_slope)
                    )
    if False: _add_relu() # TODO: Add this

    ids_bloc = sigm#.sum(axis=0)
    return sigm




def _add_relu():
    # ReLU_up = T.switch(Z_x < scalar_field_iter[1], 0,
    #                    - 0.01 * (Z_x - scalar_field_iter[1]))
    # ReLU_down = T.switch(Z_x > scalar_field_iter[-2], 0,
    #                      0.01 * T.abs_(Z_x - scalar_field_iter[-2]))
    # formations_block += ReLU_down + ReLU_up
    pass