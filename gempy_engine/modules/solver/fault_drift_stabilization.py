from dataclasses import dataclass

import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor


@dataclass
class FaultDriftScaling:
    factors: np.ndarray

    def restore_weights(self, weights: np.ndarray) -> np.ndarray:
        if weights.ndim == 1:
            return weights * self.factors
        return weights * self.factors[:, None]


def stabilize_fault_drift_system(
        covariance: np.ndarray,
        b_vector: np.ndarray,
        n_faults: int,
        relative_regularization: float,
        equilibrate: bool = True,
) -> FaultDriftScaling | None:
    """Equilibrate fault coefficient rows and add relative diagonal loading in place."""
    if n_faults == 0:
        return None
    if relative_regularization < 0:
        raise ValueError("relative_regularization must be non-negative")
    if BackendTensor.pykeops_enabled:
        raise ValueError("Fault drift stabilization requires a dense covariance matrix")

    matrix_size = covariance.shape[0]
    fault_start = matrix_size - n_faults
    factors = BackendTensor.t.ones(matrix_size, dtype=BackendTensor.dtype_obj)

    non_fault_block = covariance[:fault_start, :fault_start]
    column_norms = BackendTensor.t.linalg.norm(non_fault_block, axis=0)
    positive_column_norms = column_norms[column_norms > 0]
    reference_norm = (
        BackendTensor.t.median(positive_column_norms)
        if positive_column_norms.shape[0] > 0
        else BackendTensor.t.array(1.0, dtype=BackendTensor.dtype_obj)
    )

    if equilibrate:
        for fault_index in range(n_faults):
            matrix_index = fault_start + fault_index
            fault_column_norm = BackendTensor.t.linalg.norm(covariance[:fault_start, matrix_index])
            if fault_column_norm > 0:
                factors[matrix_index] = reference_norm / fault_column_norm

        covariance *= factors[:, None] * factors[None, :]
        if b_vector.ndim == 1:
            b_vector *= factors
        else:
            b_vector *= factors[:, None]

    diagonal = BackendTensor.t.abs(BackendTensor.t.diag(non_fault_block))
    positive_diagonal = diagonal[diagonal > 0]
    matrix_scale = (
        BackendTensor.t.median(positive_diagonal)
        if positive_diagonal.shape[0] > 0
        else reference_norm
    )
    regularization = relative_regularization * matrix_scale
    fault_indices = BackendTensor.t.arange(fault_start, matrix_size)
    covariance[fault_indices, fault_indices] -= regularization

    return FaultDriftScaling(factors=factors)
