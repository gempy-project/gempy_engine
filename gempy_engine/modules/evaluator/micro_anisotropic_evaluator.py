import numpy as np
from typing import Literal, Optional

MicroKernelType = Literal["exponential", "matern_3_2", "matern_5_2"]


def _kernel_value(r: np.ndarray, kernel_type: MicroKernelType) -> np.ndarray:
    """Evaluate the micro radial kernel K(r) where r = anisotropic_distance / kernel_range.

    All kernels satisfy K(0) = 1 and are positive and finite for r >= 0.

    exponential   — Matérn 1/2:   K(r) = exp(-r)
    matern_3_2    — Matérn 3/2:   K(r) = (1 + sqrt(3) r) exp(-sqrt(3) r)
    matern_5_2    — Matérn 5/2:   K(r) = (1 + sqrt(5) r + 5r²/3) exp(-sqrt(5) r)
    """
    if kernel_type == "exponential":
        return np.exp(-r)
    elif kernel_type == "matern_3_2":
        a = np.sqrt(3.0) * r
        return (1.0 + a) * np.exp(-a)
    elif kernel_type == "matern_5_2":
        a = np.sqrt(5.0) * r
        return (1.0 + a + (5.0 / 3.0) * r * r) * np.exp(-a)
    else:
        raise ValueError(f"Unknown micro kernel type: {kernel_type}")


def evaluate_micro_correction(
    xyz_to_interpolate: np.ndarray,      # (M, 3)
    micro_points: np.ndarray,            # (N, 3)
    micro_weights: np.ndarray,           # (N,)
    anisotropy_matrices: np.ndarray,     # (N, 3, 3)
    kernel_range: float = 1.0,
    kernel_type: MicroKernelType = "exponential",
) -> np.ndarray:
    """Evaluate the micro correction field at target points.

    V(x) = sum_i w_i * K(||A_i (x - p_i)|| / range)
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
        r = dists / kernel_range
        correction += wj * _kernel_value(r, kernel_type)

    return correction


def build_micro_covariance(
    micro_points: np.ndarray,            # (N, 3)
    anisotropy_matrices: np.ndarray,     # (N, 3, 3)
    kernel_range: float = 1.0,
    kernel_type: MicroKernelType = "exponential",
    nugget: float = 0.0,
) -> np.ndarray:
    """Build the symmetric NxN covariance matrix for the micro solve.

    K[i,j] = K(||A_i (p_i - p_j)|| / range)

    where K is the selected micro kernel and distances use the symmetric
    metric M_ij = (A_i^T A_i + A_j^T A_j) / 2.
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
    micro_points: np.ndarray,            # (N, 3)
    residuals: np.ndarray,               # (N,)
    anisotropy_matrices: np.ndarray,     # (N, 3, 3)
    kernel_range: float = 1.0,
    kernel_type: MicroKernelType = "exponential",
    nugget: float = 0.0,
) -> np.ndarray:
    """Solve K @ w = residuals for the micro correction weights.

    Returns weights array of shape (N,).
    """
    K = build_micro_covariance(micro_points, anisotropy_matrices, kernel_range, kernel_type, nugget)
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
    micro_points: np.ndarray,            # (N, D)
    gradients: np.ndarray,               # (N, D) gradient vectors
    r_vertical: float = 1.0,
    r_lateral: float = 10.0,
) -> np.ndarray:
    """Build per-point anisotropy matrices from macro gradient directions.

    A_i = S * R_i^T

    R_i^T projects world coordinates into a local frame aligned with the gradient
    (last axis = gradient direction = stratigraphic up).
    S = diag(lateral scale repeated, vertical scale)

    Works for 2D and 3D.
    """
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
