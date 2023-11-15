import warnings

from gempy_engine.config import DEBUG_MODE, AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as bt, BackendTensor
import numpy as np

from gempy_engine.core.data.exported_fields import ExportedFields


def activate_formation_block(exported_fields: ExportedFields, ids: np.ndarray, sigmoid_slope: float) -> np.ndarray:
    Z_x: np.ndarray = exported_fields.scalar_field_everywhere
    scalar_value_at_sp: np.ndarray = exported_fields.scalar_field_at_surface_points

    if LEGACY := True:
        sigm = activate_formation_block_from_args(Z_x, ids, scalar_value_at_sp, sigmoid_slope)
    else:
        sigm = activate_formation_block_from_args_hard_sigmoid(Z_x, ids, scalar_value_at_sp, sigmoid_slope)

    return sigm


def activate_formation_block_from_args(Z_x, ids, scalar_value_at_sp, sigmoid_slope):
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


def activate_formation_block_from_args_hard_sigmoid(Z_x, ids, scalar_value_at_sp, sigmoid_slope):
    element_0 = bt.t.array([0], dtype=BackendTensor.dtype_obj)

    min_Z_x = BackendTensor.t.min(Z_x, axis=0).reshape(-1)  # ? Is this as good as it gets?
    max_Z_x = BackendTensor.t.max(Z_x, axis=0).reshape(-1)  # ? Is this as good as it gets?

    # Add 5%
    min_Z_x = min_Z_x - 0.5 * (max_Z_x - min_Z_x)
    max_Z_x = max_Z_x + 0.5 * (max_Z_x - min_Z_x)

    drift_0_v = bt.tfnp.concatenate([min_Z_x, scalar_value_at_sp])
    drift_1_v = bt.tfnp.concatenate([scalar_value_at_sp, max_Z_x])

    ids = bt.t.array(ids, dtype="int32")
    scalar_0_v = bt.t.copy(ids)
    scalar_0_v[0] = 0
    
    ids = bt.t.flip(ids, (0,))
    # * Iterate over surface
    sigm = bt.t.zeros((1, Z_x.shape[0]), dtype=BackendTensor.dtype_obj)

    for i in range(len(ids) - 1):
        if False:
            sigm += HardSigmoidModified2.apply(
                Z_x,
                (drift_0_v[i] + drift_1_v[i]) / 2,
                (drift_0_v[i + 1] + drift_1_v[i + 1]) / 2,
                ids[i]
            )
        else:
            output = bt.t.zeros_like(Z_x)
            a = (drift_0_v[i] + drift_1_v[i]) / 2
            b = (drift_0_v[i + 1] + drift_1_v[i + 1]) / 2

            slope_up = -1 / (b - a)

            # For x in the range [a, b]
            b_ = (Z_x > a) & (Z_x <= b)
            pos = slope_up * (Z_x[b_] - a)

            output[b_] = ids[i] + 0.5 + pos
            sigm += output
    return sigm.reshape(1, -1)






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
