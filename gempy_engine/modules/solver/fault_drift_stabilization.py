import math

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.modules.solver.linear_system_transform import (
    LinearSystemTransform,
    add_fault_regularization,
    identity_system_transform,
)


def stabilize_fault_drift_system(
        covariance,
        b_vector,
        n_faults: int,
        relative_regularization: float,
        equilibrate: bool = True,
) -> LinearSystemTransform:
    """Equilibrate fault coefficients and add relative diagonal loading functionally."""
    transform = identity_system_transform(covariance, b_vector)
    if n_faults == 0:
        return transform
    if n_faults < 0 or n_faults > covariance.shape[0]:
        raise ValueError("n_faults must be between zero and the matrix size")
    if not math.isfinite(relative_regularization) or relative_regularization < 0:
        raise ValueError("relative_regularization must be finite and non-negative")
    if BackendTensor.pykeops_enabled:
        raise ValueError("Fault drift stabilization requires a dense covariance matrix")

    matrix_size = covariance.shape[0]
    fault_start = matrix_size - n_faults
    factors = BackendTensor.t.ones(matrix_size, dtype=covariance.dtype)

    detached_covariance = covariance.detach() if hasattr(covariance, "detach") else covariance
    non_fault_block = detached_covariance[:fault_start, :fault_start]
    column_norms = BackendTensor.t.linalg.norm(non_fault_block, axis=0)
    positive_column_norms = column_norms[column_norms > 0]
    reference_norm = (
        BackendTensor.t.median(positive_column_norms)
        if positive_column_norms.shape[0] > 0
        else BackendTensor.t.array(1.0, dtype=covariance.dtype)
    )

    if equilibrate:
        fault_column_norms = BackendTensor.t.linalg.norm(detached_covariance[:fault_start, fault_start:], axis=0)
        positive_fault_norms = fault_column_norms > 0
        fault_factors = BackendTensor.t.where(
            positive_fault_norms,
            reference_norm / BackendTensor.t.where(
                positive_fault_norms,
                fault_column_norms,
                BackendTensor.t.ones(fault_column_norms.shape, dtype=fault_column_norms.dtype),
            ),
            BackendTensor.t.ones(fault_column_norms.shape, dtype=fault_column_norms.dtype),
        )
        factors = BackendTensor.tfnp.concatenate((factors[:fault_start], fault_factors))

    scaled_matrix = covariance * factors[:, None] * factors[None, :]
    scaled_rhs = b_vector * factors if b_vector.ndim == 1 else b_vector * factors[:, None]
    transform = LinearSystemTransform(matrix=scaled_matrix, rhs=scaled_rhs, factors=factors)

    diagonal = BackendTensor.t.abs(BackendTensor.t.diag(non_fault_block))
    positive_diagonal = diagonal[diagonal > 0]
    matrix_scale = (
        BackendTensor.t.median(positive_diagonal)
        if positive_diagonal.shape[0] > 0
        else reference_norm
    )
    regularization = relative_regularization * matrix_scale
    return add_fault_regularization(transform, n_faults, regularization)
