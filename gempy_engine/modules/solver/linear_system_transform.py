from dataclasses import dataclass
import math

from gempy_engine.core.backend_tensor import BackendTensor


@dataclass
class LinearSystemTransform:
    """A symmetric diagonal transformation of a dense linear system."""

    matrix: object
    rhs: object
    factors: object
    iterations: int = 0
    converged: bool = True

    def scale_initial_guess(self, initial_guess):
        if initial_guess is None or len(initial_guess) == 0:
            return initial_guess
        self._validate_rows(initial_guess, "initial_guess")
        return _scale_rows(initial_guess, 1.0 / self.factors)

    def restore_weights(self, weights):
        if weights is None:
            return None
        self._validate_rows(weights, "weights")
        return _scale_rows(weights, self.factors)

    def _validate_rows(self, values, name: str):
        if values.ndim not in (1, 2):
            raise ValueError(f"{name} must have shape (n,), (n, 1), or (n, k)")
        if values.shape[0] != self.factors.shape[0]:
            raise ValueError(f"{name} has {values.shape[0]} rows, expected {self.factors.shape[0]}")


def identity_system_transform(matrix, rhs) -> LinearSystemTransform:
    _validate_system(matrix, rhs)
    factors = BackendTensor.t.ones(matrix.shape[0], dtype=matrix.dtype)
    return LinearSystemTransform(matrix=matrix, rhs=rhs, factors=factors)


def equilibrate_symmetric_system(
        matrix,
        rhs,
        max_iterations: int = 10,
        tolerance: float = 1e-2,
) -> LinearSystemTransform:
    """Apply max-norm Ruiz equilibration without mutating the input system."""
    _validate_system(matrix, rhs)
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    factors = BackendTensor.t.ones(matrix.shape[0], dtype=matrix.dtype)
    scaled_matrix = matrix
    scaled_rhs = rhs
    converged = False

    for iteration in range(1, max_iterations + 1):
        detached_matrix = _detached(scaled_matrix)
        row_norms = BackendTensor.tfnp.max(BackendTensor.t.abs(detached_matrix), axis=1)
        positive = row_norms > 0
        step = BackendTensor.t.where(
            positive,
            1.0 / BackendTensor.t.sqrt(BackendTensor.t.where(positive, row_norms, BackendTensor.t.ones(row_norms.shape, dtype=row_norms.dtype))),
            BackendTensor.t.ones(row_norms.shape, dtype=row_norms.dtype),
        )
        factors = factors * step
        scaled_matrix = scaled_matrix * step[:, None] * step[None, :]
        scaled_rhs = _scale_rows(scaled_rhs, step)

        remaining_norms = BackendTensor.tfnp.max(BackendTensor.t.abs(_detached(scaled_matrix)), axis=1)
        positive_remaining = remaining_norms[remaining_norms > 0]
        if positive_remaining.shape[0] == 0:
            converged = True
            break
        error = BackendTensor.tfnp.max(BackendTensor.t.abs(positive_remaining - 1.0), axis=0)
        if float(BackendTensor.t.to_numpy(error)) <= tolerance:
            converged = True
            break

    return LinearSystemTransform(
        matrix=scaled_matrix,
        rhs=scaled_rhs,
        factors=factors,
        iterations=iteration,
        converged=converged,
    )


def add_fault_regularization(transform: LinearSystemTransform, n_faults: int, regularization) -> LinearSystemTransform:
    """Add negative fault diagonal loading in transformed coordinates."""
    if n_faults < 0 or n_faults > transform.matrix.shape[0]:
        raise ValueError("n_faults must be between zero and the matrix size")
    if n_faults == 0 or regularization == 0:
        return transform

    matrix_size = transform.matrix.shape[0]
    fault_start = matrix_size - n_faults
    diagonal = BackendTensor.tfnp.concatenate((
        BackendTensor.t.zeros(fault_start, dtype=transform.matrix.dtype),
        -regularization * BackendTensor.t.ones(n_faults, dtype=transform.matrix.dtype),
    ))
    matrix = transform.matrix + BackendTensor.t.eye(matrix_size, dtype=transform.matrix.dtype) * diagonal
    return LinearSystemTransform(
        matrix=matrix,
        rhs=transform.rhs,
        factors=transform.factors,
        iterations=transform.iterations,
        converged=transform.converged,
    )


def normalized_residual(matrix, rhs, weights) -> float:
    comparable_rhs = rhs
    if weights.ndim == 1 and rhs.ndim == 2 and rhs.shape[1] == 1:
        comparable_rhs = rhs[:, 0]
    elif weights.ndim == 2 and rhs.ndim == 1:
        comparable_rhs = rhs[:, None]
    residual = matrix @ weights - comparable_rhs
    numerator = BackendTensor.t.linalg.norm(residual)
    denominator = (
        BackendTensor.t.linalg.norm(matrix) * BackendTensor.t.linalg.norm(weights)
        + BackendTensor.t.linalg.norm(comparable_rhs)
    )
    denominator_value = float(BackendTensor.t.to_numpy(_detached(denominator)))
    if denominator_value == 0:
        return float(BackendTensor.t.to_numpy(_detached(numerator)))
    return float(BackendTensor.t.to_numpy(_detached(numerator / denominator)))


def _scale_rows(values, factors):
    if values.ndim == 1:
        return values * factors
    return values * factors[:, None]


def _validate_system(matrix, rhs):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if rhs.ndim not in (1, 2):
        raise ValueError("rhs must have shape (n,), (n, 1), or (n, k)")
    if rhs.shape[0] != matrix.shape[0]:
        raise ValueError(f"rhs has {rhs.shape[0]} rows, expected {matrix.shape[0]}")


def _detached(values):
    return values.detach() if hasattr(values, "detach") else values
