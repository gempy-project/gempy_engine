from gempy_engine.core.backend_tensor import BackendTensor as bt
import numpy as np

from gempy_engine.core.data.exported_fields import ExportedFields


def activate_formation_block(exported_fields: ExportedFields, ids: np.ndarray, sigmoid_slope: float) -> np.ndarray:
    
    Z_x: np.ndarray = exported_fields.scalar_field_everywhere
    scalar_value_at_sp: np.ndarray = exported_fields.scalar_field_at_surface_points

    sigm = activate_formation_block_from_args(Z_x, ids, scalar_value_at_sp, sigmoid_slope)

    return sigm


def activate_formation_block_from_args(Z_x, ids, scalar_value_at_sp, sigmoid_slope):
    drift_0_v = bt.tfnp.concatenate([np.array([0]), scalar_value_at_sp])
    drift_1_v = bt.tfnp.concatenate([scalar_value_at_sp, np.array([0])])
    scalar_0_v = ids.copy()
    scalar_1_v = ids.copy()
    scalar_0_v[0] = 0
    scalar_1_v[-1] = 0
    ids_iter = bt.tfnp.repeat(ids, 2, axis=0)
    ids_iter[0] = 0
    ids_iter[-1] = 0
    sigm = np.zeros((1, Z_x.shape[0]))
    for i in range(ids.size):
        sigm += _compute_sigmoid(Z_x, scalar_0_v[i], scalar_1_v[i], drift_0_v[i], drift_1_v[i], ids[i], sigmoid_slope)
    if False: _add_relu()  # TODO: Add this
    return sigm


def _compute_sigmoid(Z_x, scale_0, scale_1, drift_0, drift_1, drift_id, sigmoid_slope):

    # TODO: Test to remove reshape once multiple values are implemented
    active_sig = -scale_0.reshape((-1, 1)) / (1 + bt.tfnp.exp(-sigmoid_slope * (Z_x - drift_0)))
    deactive_sig = -scale_1.reshape((-1, 1)) / (1 + bt.tfnp.exp(sigmoid_slope * (Z_x - drift_1)))
    activation_sig = active_sig + deactive_sig

    sigm = activation_sig + drift_id.reshape((-1, 1))
    return sigm


def _add_relu():
    # ReLU_up = T.switch(Z_x < scalar_field_iter[1], 0,
    #                    - 0.01 * (Z_x - scalar_field_iter[1]))
    # ReLU_down = T.switch(Z_x > scalar_field_iter[-2], 0,
    #                      0.01 * T.abs_(Z_x - scalar_field_iter[-2]))
    # formations_block += ReLU_down + ReLU_up
    pass
