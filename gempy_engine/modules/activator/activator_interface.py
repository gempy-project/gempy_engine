import warnings

from gempy_engine.config import DEBUG_MODE, AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as bt, BackendTensor
import numpy as np

from gempy_engine.core.data.exported_fields import ExportedFields


def activate_formation_block(exported_fields: ExportedFields, ids: np.ndarray, sigmoid_slope: float) -> np.ndarray:
    Z_x: np.ndarray = exported_fields.scalar_field_everywhere
    scalar_value_at_sp: np.ndarray = exported_fields.scalar_field_at_surface_points

    sigm = activate_formation_block_from_args(Z_x, ids, scalar_value_at_sp, sigmoid_slope)

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
        # sigm  += CustomSigmoidFunction.apply(Z_x, scalar_0_v[i], scalar_1_v[i], drift_0_v[i], drift_1_v[i], ids[i], sigmoid_slope)

    if False: _add_relu()  # TODO: Add this
    return sigm


def _compute_sigmoid(Z_x, scale_0, scale_1, drift_0, drift_1, drift_id, sigmoid_slope):
    # TODO: Test to remove reshape once multiple values are implemented

    with warnings.catch_warnings():
        if DEBUG_MODE:
            warnings.simplefilter("ignore", category=RuntimeWarning)

        sigmoid_slope_tensor = BackendTensor.t.array(sigmoid_slope, dtype=BackendTensor.dtype_obj)

        active_sig = -scale_0.reshape((-1, 1)) / (1 + bt.tfnp.exp(-sigmoid_slope_tensor * (Z_x - drift_0)))
        deactive_sig = -scale_1.reshape((-1, 1)) / (1 + bt.tfnp.exp(sigmoid_slope_tensor * (Z_x - drift_1)))
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

# * This gets the scalar gradient
import torch
class CustomSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Z_x, scale_0, scale_1, drift_0, drift_1, drift_id, sigmoid_slope, epsilon=1e-7):
        sigmoid_slope_tensor = sigmoid_slope

        active_sig = -scale_0 / (1 + torch.exp(-sigmoid_slope_tensor * (Z_x - drift_0)).clamp(min=epsilon))
        deactive_sig = -scale_1 / (1 + torch.exp(sigmoid_slope_tensor * (Z_x - drift_1)).clamp(min=epsilon))
        activation_sig = active_sig + deactive_sig

        sigm = activation_sig + drift_id

        ctx.save_for_backward(sigm)
        return sigm

    @staticmethod
    def backward(ctx, grad_output):
        sigm, = ctx.saved_tensors
        # Here you need to compute the actual gradient of your function with respect to the inputs.
        # The following is just a placeholder to illustrate replacing NaNs with zeros.
        # grad_input = torch.nan_to_num(grad_output)  # Replace NaNs with zeros
        # Do the actual gradient computation here
        return grad_output, None, None, None, None, None, None
