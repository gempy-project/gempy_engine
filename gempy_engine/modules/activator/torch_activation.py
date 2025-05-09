import warnings

import torch
from ...core.backend_tensor import BackendTensor as bt, BackendTensor


'''
The idea of this activation is to apply a Hard Sigmoid activation function but ideally only 
to the gradient part of the function, while we keep a harder gradient to the actual values.

Definitely needs more thought and testing.

Here is the torch implementation https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#Hardsigmoid

'''


def activate_formation_block_from_args_hard_sigmoid(Z_x, ids, scalar_value_at_sp):
    
    warnings.warn(DeprecationWarning("This function is deprecated. Use activate_formation_block instead."))
    
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
        a = (drift_0_v[i] + drift_1_v[i]) / 2
        b = (drift_0_v[i + 1] + drift_1_v[i + 1]) / 2
        if False:
            sigm += HardSigmoid.apply(
                Z_x,
                a,
                b,
                ids[i]
            )
        else:
            sigm += _baseHardSigmoid(
                Z_x,
                a,
                b,
                ids[i]
            )
    return sigm.reshape(1, -1)


class HardSigmoidModified2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, a, b, id):
        ctx.save_for_backward(input)
        ctx.bounds = (a, b)
        ctx.id = id
        output = bt.t.zeros_like(input)
        slope_up = -1 / (b - a)
        # 
        # # For x in the range [a, b]
        b_ = (input > a) & (input <= b)
        pos = slope_up * (input[b_] - a)

        output[b_] = bt.t.round(id + pos)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        a, b = ctx.bounds
        slope_up = - 1 / (b - a)

        b_ = (input > a) & (input <= b)

        grad_input = grad_output.clone()
        # Apply gradient only within the range [a, b]
        grad_input[b_] = grad_input[b_] + slope_up

        return grad_input, None, None, None


class HardSigmoidModified(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, a, b, id):
        ctx.save_for_backward(input)
        ctx.bounds = (a, b)
        output = torch.zeros_like(input)
        slope_up = 100 / (b - a)
        midpoint = (a + b) / 2

        # For x in the range [a, b]
        b_ = (input > a) & (input <= b)

        pos = slope_up * (input[b_] - a)

        neg = -slope_up * (input[b_] - b)

        print("Max min:", pos.max(), pos.min())
        foo = id * pos - (id - 1) * neg

        # output[b_] = id * pos
        output[b_] = id + pos

        # output[(input >= a) & (input <= b)] = torch.clamp(neg, min=0, max=1)
        # output[(input >= a) & (input <= b)] = torch.clamp(pos + neg, min=0, max=1)
        # output[(input >= a) & (input <= b)] = torch.clamp(pos + neg, min=0, max=1)

        # Clamping the values outside the range [a, c] to zero
        # output[input < a] = 0
        # output[input >= b] = 0

        # output[b_] *= id

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        a, b = ctx.bounds
        midpoint = (a + b) / 2
        grad_input = grad_output.clone()

        # Gradient is 1/(b-a) for x in [a, midpoint), -1/(b-a) for x in (midpoint, b], and 0 elsewhere
        grad_input[input < a] = 0
        grad_input[input > b] = 0
        grad_input[(input >= a) & (input < midpoint)] = 1 / (b - a)
        grad_input[(input > midpoint) & (input <= b)] = -1 / (b - a)

        return grad_input, None, None, None


class HardSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, a, b, c):
        ctx.save_for_backward(input)
        ctx.bounds = (a, b)
        slope = 1 / (b - a)
        return torch.clamp(slope * (input - a) + 0.5, min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        a, b = ctx.bounds
        grad_input = grad_output.clone()
        grad_input[input < a] = 0
        grad_input[input > b] = 0
        grad_input[(input >= a) & (input <= b)] = 1 / (b - a)
        return grad_input, None, None, None


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



def _baseHardSigmoid(Z_x, a, b, id):
    output = bt.t.zeros_like(Z_x)
    slope_up = -1 / (b - a)
    # For x in the range [a, b]
    b_ = (Z_x > a) & (Z_x <= b)
    pos = slope_up * (Z_x[b_] - a)
    output[b_] = id + 0.5 + pos
    return output
