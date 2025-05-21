import numbers

import numpy as np

from ...core.backend_tensor import BackendTensor as bt, BackendTensor
from ...core.data.kernel_classes.kernel_functions import dtype

try:
    import torch
except ModuleNotFoundError:
    pass


def soft_segment_unbounded(Z, edges, ids, sigmoid_slope):
    """
    Z:            array of shape (...,) of scalar values
    edges:        array of shape (K-1,) of finite split points [e1, e2, ..., e_{K-1}]
    ids:          array of shape (K,)   of the id for each of the K bins
    sigmoid_slope: scalar target peak slope m > 0
    returns:      array of shape (...,) of the soft-assigned id
    """
    ids = bt.t.array(ids[::-1].copy())

    # Check if sigmoid function is num or array
    match sigmoid_slope:
        case numbers.Number():
            membership = _lith_segmentation(Z, edges, ids, sigmoid_slope)
        case _ if isinstance(sigmoid_slope, (np.ndarray, torch.Tensor)):
            membership = _final_faults_segmentation(Z, edges, sigmoid_slope)
        case _:
            raise ValueError("sigmoid_slope must be a float or an array")

    ids__sum = bt.t.sum(membership * ids, axis=-1)
    return ids__sum[None, :]


def _final_faults_segmentation(Z, edges, sigmoid_slope):
    first = _sigmoid(
        scalar_field=Z,
        edges=edges[0],
        tau_k=1 / sigmoid_slope
    )  # shape (...,)
    last = _sigmoid(
        scalar_field=Z,
        edges=edges[-1],
        tau_k=1 / sigmoid_slope
    )
    membership = bt.t.concatenate(
        [first[..., None], last[..., None]],
        axis=-1
    )  # shape (...,K)
    return membership


def _lith_segmentation(Z, edges, ids, sigmoid_slope):
    # 1) per-edge temperatures τ_k = |Δ_k|/(4·m)
    jumps = bt.t.abs(ids[1:] - ids[:-1], dtype=bt.dtype_obj)  # shape (K-1,)
    tau_k = jumps / float(sigmoid_slope)  # shape (K-1,)
    # 2) first bin (-∞, e1) via σ((e1 - Z)/τ₁)
    first = _sigmoid(
        scalar_field=-Z,
        edges=-edges[0],
        tau_k=tau_k[0]
    )  # shape (...,)
    # 3) last  bin [e_{K-1}, ∞) via σ((Z - e_{K-1})/τ_{K-1})
    # last = 1.0 / (1.0 + np.exp(-(Z - edges[-1]) / tau_k[-1]))  # shape (...,)
    last = _sigmoid(
        scalar_field=Z,
        edges=edges[-1],
        tau_k=tau_k[-1]
    )
    # 4) middle bins [e_i, e_{i+1}): σ((Z - e_i)/τ_i) - σ((Z - e_{i+1})/τ_{i+1})
    # shape (...,1)
    left = _sigmoid(
        scalar_field=(Z[..., None]),
        edges=edges[:-1],
        tau_k=tau_k[:-1]
    )
    right = _sigmoid(
        scalar_field=(Z[..., None]),
        edges=edges[1:],
        tau_k=tau_k[1:]
    )
    middle = left - right  # (...,K-2)
    # 5) assemble memberships and weight by ids
    membership = bt.t.concatenate(
        [first[..., None], middle, last[..., None]],
        axis=-1
    )  # shape (...,K)
    return membership


def _sigmoid(scalar_field, edges, tau_k):
    x = -(scalar_field - edges) / tau_k
    return 1.0 / (1.0 + bt.t.exp(x))


def _sigmoid_stable(scalar_field, edges, tau_k):
    """
    Numerically‐stable sigmoid of (scalar_field - edges)/tau_k,
    only exponentiates on the needed slice.
    """
    x = (scalar_field - edges) / tau_k
    # allocate output
    out = bt.t.empty_like(x)

    # mask which positions are >=0 or <0
    pos = x >= 0
    neg = ~pos

    # for x>=0: safe to compute exp(-x)
    x_pos = x[pos]
    exp_neg = bt.t.exp(-x_pos)            # no overflow since -x_pos <= 0
    out[pos] = 1.0 / (1.0 + exp_neg)

    # for x<0: safe to compute exp(x)
    x_neg = x[neg]
    exp_pos = bt.t.exp(x_neg)             # no overflow since x_neg < 0
    out[neg] = exp_pos / (1.0 + exp_pos)

    return out