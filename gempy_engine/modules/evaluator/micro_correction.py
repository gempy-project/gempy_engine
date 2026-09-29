"""Fit and apply stack-local authored micro contacts without changing macro solves."""
from dataclasses import dataclass
import warnings

import numpy as np

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.micro_points import MicroArray, MicroPointResults
from .micro_anisotropic_evaluator import build_micro_design_matrix, evaluate_micro_values_and_gradient


@dataclass(frozen=True)
class MicroCorrection:
    points: MicroArray
    matrices: MicroArray
    weights: MicroArray
    kernel_range: float
    kernel_type: str
    surface_isovalues: MicroArray | None = None


def align_micro_matrices(micro, gradients):
    """Align world support z to the uncorrected macro normal, retaining world radii."""
    is_torch = BackendTensor.engine_backend is AvailableBackends.PYTORCH
    if is_torch:
        import torch
        tensor = lambda value: torch.as_tensor(value, dtype=gradients.dtype, device=gradients.device)
        linalg = torch.linalg
        cross = torch.linalg.cross
        stack = torch.stack
        where = torch.where
    else:
        tensor = np.asarray
        linalg = np.linalg
        cross = np.cross
        stack = np.stack
        where = np.where

    matrices = tensor(micro.anisotropy_matrices)
    a = tensor(micro.support_to_engine) if micro.support_to_engine is not None else tensor(np.eye(3))
    basis = linalg.inv(a) @ linalg.inv(matrices)
    radii = linalg.vector_norm(basis, dim=1) if is_torch else linalg.norm(basis, axis=1)
    world_gradient = gradients @ a
    norm = linalg.vector_norm(world_gradient, dim=1, keepdim=True) if is_torch else linalg.norm(world_gradient, axis=1, keepdims=True)
    valid = norm > 1e-12
    z = world_gradient / where(valid, norm, tensor(1.))
    z = where(valid, z, tensor([0., 0., 1.]))
    reference = where(abs(z[:, :1]) < 0.9, tensor([1., 0., 0.]), tensor([0., 1., 0.]))
    x = cross(reference, z)
    x = x / (linalg.vector_norm(x, dim=1, keepdim=True) if is_torch else linalg.norm(x, axis=1, keepdims=True))
    y = cross(z, x)
    frame = stack((x, y, z), dim=-1) if is_torch else stack((x, y, z), axis=-1)
    aligned = linalg.inv(a @ (frame * radii[:, None, :]))
    return where(valid[:, :, None], aligned, matrices)


def micro_evaluation_options(options, interpolation_input):
    """Request macro gradients locally without modifying caller-visible options."""
    micro = interpolation_input.micro_points
    if (micro is None or not len(micro.points) or not options.micro_options.enabled
            or not options.micro_options.align_to_macro or options.compute_scalar_gradient):
        return options
    updated = options.model_copy(deep=True)
    updated.evaluation_options.compute_scalar_gradient = True
    return updated


def fit_micro_correction(interpolation_input, macro_values, options, surface_sizes, macro_gradients=None):
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
    if settings.align_to_macro:
        if macro_gradients is None:
            raise ValueError("Aligned micro contacts require macro gradients")
        matrices = align_micro_matrices(micro, macro_gradients)
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
    # ExportedFields uses the first macro point of each surface as its reference.
    surface_isovalues = (torch.stack([chunk[0] for chunk in chunks]) if is_torch
                        else np.array([chunk[0] for chunk in chunks]))
    residuals = surface_isovalues[micro.surface_indices] - contact_values

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
    return MicroCorrection(centers, matrices, fitted * settings.strength, settings.kernel_range,
                           settings.kernel_type, surface_isovalues)


def fit_micro_fields(interpolation_input, fields, options, surface_sizes, xyz, publish_gradients):
    """Keep contact gradients uncorrected and publish only authored aligned rows."""
    micro = interpolation_input.micro_points
    aligned = micro is not None and len(micro.points) and options.micro_options.enabled and options.micro_options.align_to_macro
    gradients = None
    if aligned:
        rows = interpolation_input.micro_slice
        indices = interpolation_input.micro_indices
        gradients = BackendTensor.t.stack((fields.gx_field_everywhere[rows][indices],
                                           fields.gy_field_everywhere[rows][indices],
                                           fields.gz_field_everywhere[rows][indices]), axis=1)
    correction = fit_micro_correction(interpolation_input, fields.scalar_field_everywhere,
                                      options, surface_sizes, gradients)
    if aligned:
        fields.micro_point_results = MicroPointResults(interpolation_input.micro_indices.copy(), gradients,
                                                        correction.matrices[:len(micro.points)])
    apply_micro_correction(fields, xyz, correction)
    if not publish_gradients:
        fields._gx_field = fields._gy_field = fields._gz_field = None


def apply_micro_correction(fields, xyz, correction: MicroCorrection | None):
    if correction is None:
        return fields
    values, grad = evaluate_micro_values_and_gradient(
        xyz, correction.points, correction.weights, correction.matrices,
        correction.kernel_range, correction.kernel_type,
        compute_gradient=fields.gx_field_everywhere is not None,
    )
    if isinstance(fields.scalar_field_everywhere, np.ndarray):
        values = values.astype(fields.scalar_field_everywhere.dtype)
    fields._scalar_field = fields.scalar_field_everywhere + values
    if correction.surface_isovalues is not None:
        # Keep segmentation and extraction on the fitted targets even if macro points move.
        fields.scalar_field_at_surface_points = correction.surface_isovalues
    if fields.gx_field_everywhere is not None:
        if isinstance(fields.gx_field_everywhere, np.ndarray):
            grad = grad.astype(fields.gx_field_everywhere.dtype)
        fields._gx_field = fields.gx_field_everywhere + grad[:, 0]
        fields._gy_field = fields.gy_field_everywhere + grad[:, 1]
        if fields.gz_field_everywhere is not None:
            fields._gz_field = fields.gz_field_everywhere + grad[:, 2]
    return fields
