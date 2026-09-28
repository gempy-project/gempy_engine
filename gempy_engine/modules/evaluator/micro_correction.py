"""Fit and apply stack-local authored micro contacts without changing macro solves."""
from dataclasses import dataclass
import warnings

import numpy as np

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.micro_points import MicroArray
from .micro_anisotropic_evaluator import build_micro_design_matrix, evaluate_micro_correction, evaluate_micro_gradient


@dataclass(frozen=True)
class MicroCorrection:
    points: MicroArray
    matrices: MicroArray
    weights: MicroArray
    kernel_range: float
    kernel_type: str


def fit_micro_correction(interpolation_input, macro_values, options, surface_sizes):
    micro = interpolation_input.micro_points
    settings = options.micro_options
    if not settings.enabled or micro is None or not len(micro.points):
        return None
    if (not np.isfinite(settings.kernel_range) or settings.kernel_range <= 0
            or not np.isfinite(settings.nugget) or settings.nugget < 0):
        raise ValueError("micro kernel_range must be positive and nuggets nonnegative and finite")

    is_torch = BackendTensor.engine_backend is AvailableBackends.PYTORCH
    if is_torch:
        import torch
    solve_error = torch.linalg.LinAlgError if is_torch else np.linalg.LinAlgError
    def tensor(value):
        return torch.as_tensor(value, dtype=macro_values.dtype, device=macro_values.device) if is_torch else np.asarray(value)

    macro_points = tensor(interpolation_input.surface_points.sp_coords)
    points = tensor(micro.points)
    matrices = tensor(micro.anisotropy_matrices)
    if is_torch and isinstance(surface_sizes, torch.Tensor):
        # Surface counts are integer metadata, not part of the differentiable fit.
        sizes = surface_sizes.cpu().tolist()
    else:
        sizes = np.asarray(surface_sizes, dtype=int).tolist()
    if not sizes or sum(sizes) != len(macro_points):
        raise ValueError("Micro targets require the stack's surface point counts")

    macro_sp = macro_values[interpolation_input.grid.len_all_grids:interpolation_input.macro_reference_size]
    macro_sp = macro_sp[interpolation_input.slice_feature]
    contact_values = macro_values[interpolation_input.micro_slice][interpolation_input.micro_indices]
    chunks = macro_sp.split(sizes) if is_torch else np.split(macro_sp, np.cumsum(sizes)[:-1])
    means = (torch.stack([chunk.mean() for chunk in chunks]) if is_torch
             else np.array([chunk.mean() for chunk in chunks]))
    residuals = means[micro.surface_indices] - contact_values

    centers = points
    if settings.preserve_macro_points:
        # Nearest-center assignment is discrete; gradients flow through selected coordinates and metrics.
        nearest = (torch.cdist(macro_points, points).argmin(dim=1) if is_torch else
                   np.sum((macro_points[:, None] - points[None]) ** 2, axis=2).argmin(axis=1))
        centers = torch.cat((points, macro_points)) if is_torch else np.vstack((points, macro_points))
        matrices = torch.cat((matrices, matrices[nearest])) if is_torch else np.concatenate((matrices, matrices[nearest]))
    design = build_micro_design_matrix(centers, centers, matrices, settings.kernel_range, settings.kernel_type)
    diagonal = tensor(micro.nuggets) + settings.nugget
    if settings.preserve_macro_points:
        diagonal = (torch.cat((diagonal, macro_values.new_zeros(len(macro_points)))) if is_torch else
                    np.concatenate((diagonal, np.zeros(len(macro_points)))))
    design = design + (torch.diag(diagonal) if is_torch else np.diag(diagonal))
    rhs = (torch.cat((residuals, macro_values.new_zeros(len(macro_points)))) if is_torch and settings.preserve_macro_points
           else np.concatenate((residuals, np.zeros(len(macro_points)))) if settings.preserve_macro_points else residuals)
    try:
        fitted = torch.linalg.solve(design, rhs) if is_torch else np.linalg.solve(design, rhs)
    except solve_error:
        fitted = torch.linalg.lstsq(design, rhs).solution if is_torch else np.linalg.lstsq(design, rhs, rcond=None)[0]
        warnings.warn("Micro-point system is singular; using least-squares fit", RuntimeWarning, stacklevel=2)
    if not (torch.isfinite(fitted).all() if is_torch else np.isfinite(fitted).all()):
        raise ValueError("Micro-point fit produced non-finite weights")
    return MicroCorrection(centers, matrices, fitted * settings.strength, settings.kernel_range, settings.kernel_type)


def apply_micro_correction(fields, xyz, correction: MicroCorrection | None):
    if correction is None:
        return fields
    values = evaluate_micro_correction(xyz, correction.points, correction.weights,
                                       correction.matrices, correction.kernel_range, correction.kernel_type)
    if isinstance(fields.scalar_field_everywhere, np.ndarray):
        values = values.astype(fields.scalar_field_everywhere.dtype)
    fields._scalar_field = fields.scalar_field_everywhere + values
    if fields.gx_field_everywhere is not None:
        grad = evaluate_micro_gradient(xyz, correction.points, correction.weights,
                                       correction.matrices, correction.kernel_range, correction.kernel_type)
        if isinstance(fields.gx_field_everywhere, np.ndarray):
            grad = grad.astype(fields.gx_field_everywhere.dtype)
        fields._gx_field = fields.gx_field_everywhere + grad[:, 0]
        fields._gy_field = fields.gy_field_everywhere + grad[:, 1]
        if fields.gz_field_everywhere is not None:
            fields._gz_field = fields.gz_field_everywhere + grad[:, 2]
    return fields
