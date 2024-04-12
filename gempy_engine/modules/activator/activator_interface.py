import warnings

from gempy_engine.config import DEBUG_MODE, AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as bt, BackendTensor
import numpy as np

from gempy_engine.core.data.exported_fields import ExportedFields


def activate_formation_block(exported_fields: ExportedFields, ids: np.ndarray,
                             sigmoid_slope: float) -> np.ndarray:
    Z_x: np.ndarray = exported_fields.scalar_field_everywhere
    scalar_value_at_sp: np.ndarray = exported_fields.scalar_field_at_surface_points

    if LEGACY := True: # * Here we branch to the experimental activation function with hard sigmoid
        sigm = activate_formation_block_from_args(Z_x, ids, scalar_value_at_sp, sigmoid_slope)
    else:
        from .torch_activation import activate_formation_block_from_args_hard_sigmoid
        sigm = activate_formation_block_from_args_hard_sigmoid(Z_x, ids, scalar_value_at_sp, sigmoid_slope)

    return sigm


def activate_formation_block_from_args(Z_x, ids, scalar_value_at_sp, sigmoid_slope):
    def _compute_sigmoid(Z_x, scale_0, scale_1, drift_0, drift_1, drift_id, sigmoid_slope):
        # TODO: Test to remove reshape once multiple values are implemented

        with warnings.catch_warnings():
            if DEBUG_MODE:
                warnings.simplefilter("ignore", category=RuntimeWarning)

            sigmoid_slope_tensor = BackendTensor.t.array(sigmoid_slope, dtype=BackendTensor.dtype_obj)

            active_denominator = (1 + bt.tfnp.exp(-sigmoid_slope_tensor * (Z_x - drift_0)))
            deactive_denominator = (1 + bt.tfnp.exp(sigmoid_slope_tensor * (Z_x - drift_1)))

            active_sig = -scale_0.reshape((-1, 1)) / active_denominator
            deactive_sig = -scale_1.reshape((-1, 1)) / deactive_denominator
            activation_sig = active_sig + deactive_sig

        sigm = activation_sig + drift_id.reshape((-1, 1))
        return sigm

    element_0 = bt.t.array([0], dtype=BackendTensor.dtype_obj)

    drift_0_v = bt.tfnp.concatenate([element_0, scalar_value_at_sp])
    drift_1_v = bt.tfnp.concatenate([scalar_value_at_sp, element_0])

    ids = bt.t.array(ids, dtype="int32")
    scalar_0_v = bt.t.copy(ids)
    scalar_0_v[0] = 0

    scalar_1_v = bt.t.copy(ids)
    scalar_1_v[-1] = 0

    # * Iterate over surface
    sigm = bt.t.zeros((1, Z_x.shape[0]), dtype=BackendTensor.dtype_obj)

    for i in range(len(ids)):
        sigm += _compute_sigmoid(Z_x, scalar_0_v[i], scalar_1_v[i], drift_0_v[i], drift_1_v[i], ids[i], sigmoid_slope)
    return sigm
