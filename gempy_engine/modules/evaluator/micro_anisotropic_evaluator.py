import numpy as np
from ...core.data.options.micro_anisotropic_options import MicroKernelType


def _kernel_value(r: np.ndarray, kernel_type: MicroKernelType) -> np.ndarray:
    """Evaluate the micro radial kernel K(r) where r = anisotropic_distance / kernel_range.

    All kernels satisfy K(0) = 1 and are positive and finite for r >= 0.

    exponential   — Matérn 1/2:   K(r) = exp(-r)
    matern_3_2    — Matérn 3/2:   K(r) = (1 + sqrt(3) r) exp(-sqrt(3) r)
    matern_5_2    — Matérn 5/2:   K(r) = (1 + sqrt(5) r + 5r²/3) exp(-sqrt(5) r)
    """
    if not isinstance(r, (np.ndarray, np.generic, float, int)):
        import torch
        exp = torch.exp
    else:
        exp = np.exp
    if kernel_type == "exponential":
        return exp(-r)
    elif kernel_type == "matern_3_2":
        a = np.sqrt(3.0) * r
        return (1.0 + a) * exp(-a)
    elif kernel_type == "matern_5_2":
        a = np.sqrt(5.0) * r
        return (1.0 + a + (5.0 / 3.0) * r * r) * exp(-a)
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
    return evaluate_micro_values_and_gradient(
        xyz_to_interpolate, micro_points, micro_weights, anisotropy_matrices,
        kernel_range, kernel_type,
    )[0]


def build_micro_design_matrix(xyz: np.ndarray, centers: np.ndarray,
                              anisotropy_matrices: np.ndarray, kernel_range: float,
                              kernel_type: MicroKernelType) -> np.ndarray:
    """Rows are evaluation points; column j uses the evaluator's center j metric."""
    if not isinstance(xyz, np.ndarray):
        import torch
        centers = torch.as_tensor(centers, dtype=xyz.dtype, device=xyz.device)
        matrices = torch.as_tensor(anisotropy_matrices, dtype=xyz.dtype, device=xyz.device)
        return torch.stack([_kernel_value(torch.linalg.vector_norm((xyz - center) @ matrix.T, dim=1) / kernel_range,
                                          kernel_type) for center, matrix in zip(centers, matrices)], dim=1)
    result = np.empty((len(xyz), len(centers)), dtype=np.float64)
    for j, (center, matrix) in enumerate(zip(centers, anisotropy_matrices)):
        result[:, j] = _kernel_value(
            np.linalg.norm((xyz - center) @ matrix.T, axis=1) / kernel_range, kernel_type
        )
    return result


def evaluate_micro_gradient(xyz: np.ndarray, centers: np.ndarray, weights: np.ndarray,
                            matrices: np.ndarray, kernel_range: float,
                            kernel_type: MicroKernelType) -> np.ndarray:
    return evaluate_micro_values_and_gradient(xyz, centers, weights, matrices, kernel_range,
                                              kernel_type, compute_gradient=True)[1]


def evaluate_micro_values_and_gradient(xyz, centers, weights, matrices, kernel_range=1.0,
                                       kernel_type: MicroKernelType = "exponential", compute_gradient=False):
    """Evaluate values and, if requested, gradients in one pass over the centers."""
    if not isinstance(xyz, np.ndarray):
        result = _evaluate_micro_torch(xyz, centers, weights, matrices, kernel_range, kernel_type,
                                       compute_gradient=compute_gradient)
        return result if compute_gradient else (result, None)

    correction = np.zeros(xyz.shape[0], dtype=np.float64)
    gradient = np.zeros_like(xyz, dtype=np.float64) if compute_gradient else None
    for center, matrix, weight in zip(centers, matrices, weights):
        delta = xyz - center
        transformed = delta @ matrix.T
        distance = np.linalg.norm(transformed, axis=1)
        r = distance / kernel_range
        correction += weight * _kernel_value(r, kernel_type)
        if compute_gradient:
            if kernel_type == "exponential":
                derivative = -np.exp(-r)
            elif kernel_type == "matern_3_2":
                derivative = -3 * r * np.exp(-np.sqrt(3) * r)
            elif kernel_type == "matern_5_2":
                derivative = -(5 / 3) * r * (1 + np.sqrt(5) * r) * np.exp(-np.sqrt(5) * r)
            else:
                raise ValueError(f"Unknown micro kernel type: {kernel_type}")
            scale = np.divide(derivative, kernel_range * distance,
                              out=np.zeros_like(distance), where=distance > 0)
            gradient += weight * scale[:, None] * (transformed @ matrix)
    return correction, gradient


def _evaluate_micro_torch(xyz, centers, weights, matrices, kernel_range, kernel_type,
                          compute_gradient=False):
    import torch

    centers = torch.as_tensor(centers, dtype=xyz.dtype, device=xyz.device)
    weights = torch.as_tensor(weights, dtype=xyz.dtype, device=xyz.device)
    matrices = torch.as_tensor(matrices, dtype=xyz.dtype, device=xyz.device)
    correction = xyz.new_zeros(xyz.shape[0])
    gradient = xyz.new_zeros(xyz.shape) if compute_gradient else None
    for center, matrix, weight in zip(centers, matrices, weights):
        transformed = (xyz - center) @ matrix.T
        distance = torch.linalg.vector_norm(transformed, dim=1)
        r = distance / kernel_range
        if kernel_type == "exponential":
            value = torch.exp(-r)
            if compute_gradient:
                derivative = -value
        elif kernel_type == "matern_3_2":
            a = np.sqrt(3.) * r
            value = (1 + a) * torch.exp(-a)
            if compute_gradient:
                derivative = -3 * r * torch.exp(-a)
        elif kernel_type == "matern_5_2":
            a = np.sqrt(5.) * r
            value = (1 + a + 5 / 3 * r * r) * torch.exp(-a)
            if compute_gradient:
                derivative = -(5 / 3) * r * (1 + a) * torch.exp(-a)
        else:
            raise ValueError(f"Unknown micro kernel type: {kernel_type}")
        correction = correction + weight * value
        if compute_gradient:
            scale = derivative / (kernel_range * distance.clamp_min(torch.finfo(xyz.dtype).tiny))
            gradient = gradient + weight * torch.where(distance > 0, scale, 0)[:, None] * (transformed @ matrix)
    return (correction, gradient) if compute_gradient else correction
