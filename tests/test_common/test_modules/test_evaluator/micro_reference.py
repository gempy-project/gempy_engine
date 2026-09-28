"""Dense reference micro solve used by evaluator tests, not the production fit."""

import numpy as np

from gempy_engine.modules.evaluator.micro_anisotropic_evaluator import _kernel_value, MicroKernelType


def build_micro_covariance(
    micro_points: np.ndarray,
    anisotropy_matrices: np.ndarray,
    kernel_range: float = 1.0,
    kernel_type: MicroKernelType = "exponential",
    nugget: float = 0.0,
) -> np.ndarray:
    """Build the symmetric NxN covariance matrix for the reference micro solve.

    Distances use the symmetric metric M_ij = (A_i^T A_i + A_j^T A_j) / 2.
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
            r = dist / kernel_range
            val = float(_kernel_value(np.array(r), kernel_type))
            K[i, j] = val
            K[j, i] = val

    if nugget > 0:
        np.fill_diagonal(K, K.diagonal() + nugget)
    return K


def solve_micro_weights(
    micro_points: np.ndarray,
    residuals: np.ndarray,
    anisotropy_matrices: np.ndarray,
    kernel_range: float = 1.0,
    kernel_type: MicroKernelType = "exponential",
    nugget: float = 0.0,
) -> np.ndarray:
    """Solve the dense reference covariance system."""
    K = build_micro_covariance(micro_points, anisotropy_matrices, kernel_range, kernel_type, nugget)
    weights = np.linalg.solve(K, residuals)
    return weights


def compute_anisotropy_matrices_from_gradients(
    micro_points: np.ndarray,
    gradients: np.ndarray,
    r_vertical: float = 1.0,
    r_lateral: float = 10.0,
) -> np.ndarray:
    """Build per-point anisotropy matrices from macro gradient directions (2D/3D)."""
    N, D = micro_points.shape
    assert gradients.shape == (N, D), f"gradients shape {gradients.shape} != (N, D) {(N, D)}"

    if D == 2:
        lateral_scales = np.array([1.0 / r_lateral], dtype=np.float64)
        scales = np.concatenate([lateral_scales, [1.0 / r_vertical]])
        S = np.diag(scales)
    else:
        S = np.diag(np.array([1.0 / r_lateral, 1.0 / r_lateral, 1.0 / r_vertical]))

    matrices = np.zeros((N, D, D), dtype=np.float64)
    for i in range(N):
        grad = gradients[i].astype(np.float64)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            grad = np.zeros(D, dtype=np.float64)
            grad[-1] = 1.0

        z_axis = grad / np.linalg.norm(grad)
        if D == 2:
            x_axis = np.array([z_axis[1], -z_axis[0]], dtype=np.float64)
            R = np.column_stack([x_axis, z_axis])
        else:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            if abs(np.dot(z_axis, ref)) > 0.99:
                ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)

            x_axis = np.cross(z_axis, ref)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            R = np.column_stack([x_axis, y_axis, z_axis])

        matrices[i] = S @ R.T
    return matrices
