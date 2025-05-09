import warnings

from gempy_engine.config import DEBUG_MODE, AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as bt, BackendTensor
import numpy as np
import numbers

from gempy_engine.core.data.exported_fields import ExportedFields


def activate_formation_block(exported_fields: ExportedFields, ids: np.ndarray,
                             sigmoid_slope: float) -> np.ndarray:
    Z_x: np.ndarray = exported_fields.scalar_field_everywhere
    scalar_value_at_sp: np.ndarray = exported_fields.scalar_field_at_surface_points

    sigmoid_slope_negative = isinstance(sigmoid_slope, float) and sigmoid_slope < 0  # * sigmoid_slope can be array for finite faultskA

    if LEGACY := False and not sigmoid_slope_negative:  # * Here we branch to the experimental activation function with hard sigmoid
        sigm = activate_formation_block_from_args(
            Z_x=Z_x,
            ids=ids,
            scalar_value_at_sp=scalar_value_at_sp,
            sigmoid_slope=sigmoid_slope
        )
    else:
        match BackendTensor.engine_backend:
            case AvailableBackends.PYTORCH:
                sigm = soft_segment_unbounded(
                    Z=Z_x,
                    edges=scalar_value_at_sp,
                    ids=ids,
                    sigmoid_slope=sigmoid_slope
                )
            case AvailableBackends.numpy:
                sigm = soft_segment_unbounded_np(
                    Z=Z_x,
                    edges=scalar_value_at_sp,
                    ids=ids,
                    sigmoid_slope=sigmoid_slope
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


def soft_segment_unbounded(Z, edges, ids, sigmoid_slope):
    """
    Z:            (...,) tensor of scalar values
    edges:   (K-1,) tensor of finite split points [e1, e2, ..., e_{K-1}]
    ids:       (K,) tensor of the id for each of the K bins
    tau:         scalar target peak slope m > 0
    returns:      (...,) tensor of the soft‐assigned id
    """
    # --- ensure torch tensors on the same device/dtype ---
    if not isinstance(Z, torch.Tensor):
        Z = torch.tensor(Z, dtype=torch.float32, device=edges.device)
    if not isinstance(edges, torch.Tensor):
        edges = torch.tensor(edges, dtype=Z.dtype, device=Z.device)
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids, dtype=Z.dtype, device=Z.device)

    # --- 1) per-edge temp: tau_k = jump_k / (4 * m) ---
    # jumps = ids[1:] - ids[:-1]  # shape (K-1,)

    jumps = torch.abs(ids[1:] - ids[:-1])  # shape (K-1,)
    tau_k = jumps / (4 * sigmoid_slope)  # shape (K-1,)

    # --- 2) first bin (-∞, e1) ---
    first = torch.sigmoid((edges[0] - Z) / tau_k[0])[..., None]  # (...,1)

    # --- 3) last bin [e_{K-1}, ∞) ---
    last = torch.sigmoid((Z - edges[-1]) / tau_k[-1])[..., None]  # (...,1)

    # --- 4) middle bins [e_i, e_{i+1}) ---
    left = torch.sigmoid((Z[..., None] - edges[:-1]) / tau_k[:-1])  # (...,K-2)
    right = torch.sigmoid((Z[..., None] - edges[1:]) / tau_k[1:])  # (...,K-2)
    middle = left - right  # (...,K-2)

    # --- 5) assemble memberships and weight by ids ---
    membership = torch.cat([first, middle, last], dim=-1)  # (...,K)
    # return (membership * ids).sum(dim=-1)                  # (...,)

    # weighted sum by the ids
    ids__sum = (membership * ids).sum(dim=-1)
    return np.atleast_2d(ids__sum.numpy())


import numpy as np


def soft_segment_unbounded_np(Z, edges, ids, sigmoid_slope):
    """
    Z:            array of shape (...,) of scalar values
    edges:        array of shape (K-1,) of finite split points [e1, e2, ..., e_{K-1}]
    ids:          array of shape (K,)   of the id for each of the K bins
    sigmoid_slope: scalar target peak slope m > 0
    returns:      array of shape (...,) of the soft-assigned id
    """
    Z = np.asarray(Z)
    edges = np.asarray(edges)
    ids = np.asarray(ids)

    # Check if sigmoid function is num or array
    match sigmoid_slope:
        case np.ndarray():
            membership = _final_faults_segmentation(Z, edges, sigmoid_slope)
        case numbers.Number():
             membership = _lith_segmentation(Z, edges, ids, sigmoid_slope)
        case _:
             raise ValueError("sigmoid_slope must be a float or an array")


    ids__sum = np.sum(membership * ids, axis=-1)
    return np.atleast_2d(ids__sum)


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
    membership = np.concatenate(
        [first[..., None], last[..., None]],
        axis=-1
    )  # shape (...,K)
    return membership


def _lith_segmentation(Z, edges, ids, sigmoid_slope):
    # 1) per-edge temperatures τ_k = |Δ_k|/(4·m)
    jumps = np.abs(ids[1:] - ids[:-1])  # shape (K-1,)
    tau_k = jumps / (4 * sigmoid_slope)  # shape (K-1,)
    # 2) first bin (-∞, e1) via σ((e1 - Z)/τ₁)
    first = _sigmoid(
        scalar_field=Z,
        edges=edges[0],
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
    membership = np.concatenate(
        [first[..., None], middle, last[..., None]],
        axis=-1
    )  # shape (...,K)
    return membership


def _sigmoid(scalar_field, edges, tau_k):
    return 1.0 / (1.0 + np.exp((scalar_field - edges) / tau_k))
