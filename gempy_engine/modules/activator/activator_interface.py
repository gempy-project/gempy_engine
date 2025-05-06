import warnings

from gempy_engine.config import DEBUG_MODE, AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as bt, BackendTensor
import numpy as np

from gempy_engine.core.data.exported_fields import ExportedFields


def activate_formation_block(exported_fields: ExportedFields, ids: np.ndarray,
                             sigmoid_slope: float) -> np.ndarray:
    Z_x: np.ndarray = exported_fields.scalar_field_everywhere
    scalar_value_at_sp: np.ndarray = exported_fields.scalar_field_at_surface_points

    sigmoid_slope_negative = isinstance(sigmoid_slope, float) and sigmoid_slope < 0 # * sigmoid_slope can be array for finite faultskA
    
    if LEGACY := False and not sigmoid_slope_negative: # * Here we branch to the experimental activation function with hard sigmoid
        sigm = activate_formation_block_from_args(Z_x, ids, scalar_value_at_sp, sigmoid_slope)
    else:
        # from .torch_activation import activate_formation_block_from_args_hard_sigmoid
        # sigm = activate_formation_block_from_args_hard_sigmoid(Z_x, ids, scalar_value_at_sp)

        # assume scalar_value_at_sp is shape (K-1,)
        # bt.t.array
        # edges = bt.t.concatenate([
        #          bt.t.array([0.], dtype=BackendTensor.dtype_obj),
        #         scalar_value_at_sp,
        #         bt.t.array([float('inf')], dtype=BackendTensor.dtype_obj)
        # ])  # now length K+1
        # ids = torch.arange(K, dtype=scalar_value_at_sp.dtype, device=scalar_value_at_sp.device)
        
        sigm = soft_segment_unbounded(
            Z=Z_x,
            edges=scalar_value_at_sp,
            ids=ids,
            tau=sigmoid_slope, 
        )

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


import torch

def soft_segment_unbounded(Z, edges, ids, tau):
    """
    Z:       (...,) tensor of scalar values
    edges:   (K-1,) tensor of finite split points [e1, e2, ..., e_{K-1}]
    ids:     (K,)   tensor of the id for each of the K bins
    tau:     temperature > 0
    returns: (...,) tensor of the soft‐assigned id
    """
    # first bin: (-∞, e1)
    
    # convert numpy to torch tensor
    edges = torch.tensor(edges, dtype=BackendTensor.dtype_obj)
    ids = torch.tensor(ids, dtype=BackendTensor.dtype_obj)
    Z = torch.tensor(Z, dtype=BackendTensor.dtype_obj)
    tau = torch.tensor(tau, dtype=BackendTensor.dtype_obj)
    
    first = torch.sigmoid((edges[0] - Z) / tau)[..., None]       # (...,1)

    # last  bin: [e_{K-1}, ∞)
    last  = torch.sigmoid((Z - edges[-1]) / tau)[..., None]      # (...,1)

    # middle bins: [e_i, e_{i+1})
    # edges[:-1] are e1..e_{K-2}, edges[1:] are e2..e_{K-1}
    left  =  torch.sigmoid((Z[...,None] - edges[:-1]) / tau)     # (...,K-2)
    right =  torch.sigmoid((Z[...,None] - edges[1: ]) / tau)     # (...,K-2)
    middle = left - right                                        # (...,K-2)

    # concatenate into (...,K) membership probabilities
    membership = torch.cat([first, middle, last], dim=-1)        # (...,K)

    # weighted sum by the ids
    ids__sum = (membership * ids).sum(dim=-1)
    return np.atleast_2d(ids__sum.numpy())