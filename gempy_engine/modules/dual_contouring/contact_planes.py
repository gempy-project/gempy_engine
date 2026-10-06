"""Pure plane fitting for the affine-only initial two-plane contact support."""

import numpy as np


def fit_contact_plane(points, scalar_values, isovalue, tolerance=1e-8):
    """Return ``(normal, offset)`` with ``normal @ x = offset`` at isovalue.

    The unit normal points toward increasing scalar values. CPU array-like
    samples must have shapes (N, 3) and (N,), with full three-dimensional
    affine support. Only fields affine on the supplied samples are supported:
    curved fields are rejected, never approximated with heuristic planes.
    The caller must supply representative samples of the entire field.

    ``tolerance`` bounds the maximum affine residual relative to the scalar
    range, with an additional floating-point scalar/coordinate precision
    allowance. Invalid or unsupported inputs raise ValueError. Inputs are
    not modified and no engine backend state is accessed.
    """
    points = np.asarray(points, dtype=np.float64)
    scalar_values = np.asarray(scalar_values, dtype=np.float64)
    isovalue = np.asarray(isovalue, dtype=np.float64)
    tolerance = np.asarray(tolerance, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        raise ValueError("points must have shape (N, 3) with N >= 4")
    if scalar_values.shape != (points.shape[0],):
        raise ValueError("scalar_values must have shape (N,)")
    if isovalue.ndim != 0 or tolerance.ndim != 0:
        raise ValueError("isovalue and tolerance must be scalars")
    if not (np.all(np.isfinite(points)) and np.all(np.isfinite(scalar_values))
            and np.isfinite(isovalue) and np.isfinite(tolerance)):
        raise ValueError("plane fitting requires all finite inputs")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")

    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            # Anchor before centering so large world coordinates do not enter
            # the least-squares matrix or lose their small local differences.
            local_points = points - points[0]
            center = np.mean(local_points, axis=0)
            centered_points = local_points - center
            local_scalars = scalar_values - scalar_values[0]
            scalar_center = np.mean(local_scalars)
            centered_scalars = local_scalars - scalar_center
            coordinate_scale = np.max(np.abs(centered_points), axis=0)
            if np.any(coordinate_scale == 0):
                raise ValueError("plane support must have full 3D rank")
            scalar_range = np.ptp(local_scalars)
            if scalar_range == 0:
                raise ValueError("scalar field has zero gradient")
            coefficients, _, rank, _ = np.linalg.lstsq(
                centered_points / coordinate_scale,
                centered_scalars / scalar_range,
                rcond=None
            )
            if rank != 3:
                raise ValueError("plane support must have full 3D rank")
            gradient = coefficients / coordinate_scale
            gradient_scale = np.max(np.abs(gradient))
            if gradient_scale == 0:
                raise ValueError("scalar field has zero gradient")
            normal = gradient / gradient_scale
            normal_length = np.linalg.norm(normal)
            normal = normal / normal_length

            residual = centered_points @ gradient - centered_scalars / scalar_range
            eps = np.finfo(np.float64).eps
            precision = (
                8 * eps * np.max(np.abs(scalar_values)) / scalar_range
                + 4 * np.max(np.spacing(np.abs(points)), axis=0) @ np.abs(gradient)
                + 32 * eps
            )
            if np.max(np.abs(residual)) > tolerance + precision:
                raise ValueError("non-affine scalar fields are not supported")

            offset = (
                normal @ points[0] + normal @ center
                + ((float(isovalue) - scalar_values[0]) / scalar_range
                   - scalar_center / scalar_range) / gradient_scale / normal_length
            )
            if not (np.all(np.isfinite(normal)) and np.isfinite(offset)):
                raise ValueError("plane coefficients must be finite")
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("plane fitting exceeds numerical precision") from error
    return normal, float(offset)
