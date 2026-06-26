import numpy as np
from typing import Optional


def evaluate_micro_correction(
    xyz_to_interpolate: np.ndarray,      # (M, 3)
    micro_points: np.ndarray,            # (N, 3)
    micro_weights: np.ndarray,           # (N,)
    anisotropy_matrices: np.ndarray,     # (N, 3, 3)
    kernel_range: float = 1.0,
) -> np.ndarray:
    """Evaluate the micro correction field at target points.

    V(x) = sum_i w_i * exp(-||A_i (x - p_i)|| / range)
    """
    M = xyz_to_interpolate.shape[0]
    N = micro_points.shape[0]
    correction = np.zeros(M, dtype=np.float64)

    for j in range(N):
        Aj = anisotropy_matrices[j]
        wj = micro_weights[j]
        pj = micro_points[j]
        diffs = xyz_to_interpolate - pj[np.newaxis, :]
        transformed = np.einsum('ij,mj->mi', Aj, diffs)
        dists = np.linalg.norm(transformed, axis=1)
        correction += wj * np.exp(-dists / kernel_range)

    return correction


def build_micro_covariance(
    micro_points: np.ndarray,            # (N, 3)
    anisotropy_matrices: np.ndarray,     # (N, 3, 3)
    kernel_range: float = 1.0,
    nugget: float = 0.0,
) -> np.ndarray:
    """Build the symmetric NxN covariance matrix for the micro solve.

    K[i,j] = exp(-dist_sym(i,j) / range)

    where dist_sym(i,j)^2 = (x_i - x_j)^T * M_ij * (x_i - x_j)
    with M_ij = (A_i^T A_i + A_j^T A_j) / 2
    """
    N = micro_points.shape[0]
    K = np.zeros((N, N), dtype=np.float64)

    ATA = np.einsum('nki,nkj->nij', anisotropy_matrices, anisotropy_matrices)

    for i in range(N):
        for j in range(i, N):
            M_ij = 0.5 * (ATA[i] + ATA[j])
            diff = micro_points[i] - micro_points[j]
            dist_sq = diff @ M_ij @ diff
            dist = np.sqrt(max(dist_sq, 0.0))
            val = np.exp(-dist / kernel_range)
            K[i, j] = val
            K[j, i] = val

    if nugget > 0:
        np.fill_diagonal(K, K.diagonal() + nugget)

    return K


def solve_micro_weights(
    micro_points: np.ndarray,            # (N, 3)
    residuals: np.ndarray,               # (N,)
    anisotropy_matrices: np.ndarray,     # (N, 3, 3)
    kernel_range: float = 1.0,
    nugget: float = 0.0,
) -> np.ndarray:
    """Solve K @ w = residuals for the micro correction weights.

    Returns weights array of shape (N,).
    """
    K = build_micro_covariance(micro_points, anisotropy_matrices, kernel_range, nugget)
    weights = np.linalg.solve(K, residuals)
    return weights


def compute_macro_values_at_micro_points(
    xyz_to_interpolate: np.ndarray,
    weights: np.ndarray,
    solver_input: 'SolverInput',
    options: 'InterpolationOptions',
) -> np.ndarray:
    """Extract the macro scalar field at micro point locations.

    This evaluates the macro interpolation exactly at the micro contact points
    to compute residuals = target_values - macro_values.
    """
    from gempy_engine.modules.evaluator.symbolic_evaluator import symbolic_evaluator
    from gempy_engine.core.data.internal_structs import SolverInput

    proxy_input = SolverInput(
        sp_internal=solver_input.sp_internal,
        ori_internal=solver_input.ori_internal,
        xyz_to_interpolate=xyz_to_interpolate,
        fault_internal=solver_input._fault_internal,
    )

    exported = symbolic_evaluator(proxy_input, weights, options)
    return exported.scalar_field


def compute_anisotropy_matrices_from_gradients(
    micro_points: np.ndarray,            # (N, 3)
    gradients: np.ndarray,               # (N, 3) normalized gradient vectors
    r_vertical: float = 1.0,
    r_lateral: float = 10.0,
) -> np.ndarray:
    """Build per-point anisotropy matrices from macro gradient directions.

    A_i = S * R_i^T

    R_i^T projects world coordinates into a local frame aligned with the gradient
    (z_axis = gradient direction = stratigraphic up).
    S = diag(1/r_lateral, 1/r_lateral, 1/r_vertical)

    The lateral/stratal directions are derived from gradient and a fixed reference.
    """
    N = micro_points.shape[0]
    S = np.diag(np.array([1.0 / r_lateral, 1.0 / r_lateral, 1.0 / r_vertical]))

    matrices = np.zeros((N, 3, 3), dtype=np.float64)

    for i in range(N):
        grad = gradients[i]
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            grad = np.array([0.0, 0.0, 1.0])

        z_axis = grad / np.linalg.norm(grad)

        ref = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(z_axis, ref)) > 0.99:
            ref = np.array([1.0, 0.0, 0.0])

        x_axis = np.cross(z_axis, ref)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])
        matrices[i] = S @ R.T

    return matrices
