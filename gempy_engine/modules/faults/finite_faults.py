import numpy as np

from scipy.interpolate import make_interp_spline

from gempy_engine.core.data.finite_fault import FiniteFault, TaperType

_DEFAULT_SPLINE_CONTROL_POINTS = np.array([
        [0.0, 1.0],
        [0.2, 0.95],
        [0.5, 0.5],
        [0.8, 0.05],
        [1.0, 0.0]
])


def calculate_slip(finite_fault: FiniteFault, points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Calculate a slip multiplier using the prototype local-plane model."""
    u, v, _ = get_local_frame(normal, angle_deg=finite_fault.rotation_deg)
    distance = get_ellipsoid_distance(
        points=points,
        center=np.asarray(finite_fault.center),
        u=u,
        v=v,
        a=finite_fault.strike_radius,
        b=finite_fault.dip_radius,
    )

    if finite_fault.taper is TaperType.CUBIC:
        return cubic_hermite_taper(distance)
    if finite_fault.taper is TaperType.QUADRATIC:
        return quadratic_taper(distance)
    if finite_fault.taper is TaperType.SPLINE:
        control_points = finite_fault.spline_control_points or _DEFAULT_SPLINE_CONTROL_POINTS
        return spline_taper(distance, np.asarray(control_points))
    raise ValueError(f"Unknown taper type: {finite_fault.taper}")


def get_local_frame(normal: np.ndarray, angle_deg: float = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a local orthonormal basis (u, v, w) given a normal vector w.
    w is the normal to the fault surface.
    u is the strike vector (horizontal, perpendicular to w and global Z).
    v is the dip vector (perpendicular to u and w).
    
    Args:
        normal: Normal vector to the fault surface.
        angle_deg: Rotation angle in degrees around the normal vector.
    """
    w = normal / np.linalg.norm(normal)
    z_axis = np.array([0, 0, 1.0])
    if np.abs(np.dot(w, z_axis)) > 0.999:
        u = np.cross(w, np.array([1, 0, 0.0]))
    else:
        u = np.cross(w, z_axis)
    u /= np.linalg.norm(u)
    
    if angle_deg != 0:
        # Rodrigues' rotation formula: rotate u around w
        theta = np.radians(angle_deg)
        u = u * np.cos(theta) + np.cross(w, u) * np.sin(theta) + w * np.dot(w, u) * (1 - np.cos(theta))
        u /= np.linalg.norm(u)

    v = np.cross(w, u)
    v /= np.linalg.norm(v)
    return u, v, w


def get_ellipsoid_distance(points: np.ndarray, center: np.ndarray, u: np.ndarray, v: np.ndarray,
                           a: float | tuple[float, float] = 1.0,
                           b: float | tuple[float, float] = 1.0,
                           w: np.ndarray = None,
                           c: float | tuple[float, float] = 1.0) -> np.ndarray:
    """
    Calculate normalized distance d from the center in the local (u, v, w) plane.
    d = sqrt((x_local/a)^2 + (y_local/b)^2 + (z_local/c)^2)
    
    Args:
        points: Points to calculate distance for.
        center: Center of the ellipsoid.
        u: Strike vector.
        v: Dip vector.
        a: Radius along u. If tuple (a_pos, a_neg), different radii for positive and negative u.
        b: Radius along v. If tuple (b_pos, b_neg), different radii for positive and negative v.
        w: Normal vector.
        c: Radius along w. If tuple (c_pos, c_neg), different radii for positive and negative w.
    """
    relative_points = points - center
    x_local = np.dot(relative_points, u)
    y_local = np.dot(relative_points, v)

    def get_radius_mask(val, radius):
        if isinstance(radius, (tuple, list, np.ndarray)):
            r_pos, r_neg = radius
            return np.where(val >= 0, r_pos, r_neg)
        return radius

    a_mask = get_radius_mask(x_local, a)
    b_mask = get_radius_mask(y_local, b)

    if w is not None:
        z_local = np.dot(relative_points, w)
        c_mask = get_radius_mask(z_local, c)
        d = np.sqrt((x_local / a_mask) ** 2 + (y_local / b_mask) ** 2 + (z_local / c_mask) ** 2)
    else:
        d = np.sqrt((x_local / a_mask) ** 2 + (y_local / b_mask) ** 2)
    return d


def project_points_onto_surface(
        points: np.ndarray,
        scalar_field_values: np.ndarray,
        gradient_fields: tuple[np.ndarray, np.ndarray, np.ndarray],
        target_scalar_value: float = 0.0,
        *,
        gradient_tolerance: float = 1e-12,
        surface_tolerance: float = 1e-12,
) -> np.ndarray:
    """
    Apply one Newton projection step toward ``F(x, y, z) = target``.

    The step is exact for a linear scalar field. Nonlinear fields require the
    scalar field and gradient to be re-evaluated before applying another step.
    A point away from the surface cannot be projected when its gradient is
    effectively zero, so that case raises instead of silently leaving it in an
    invalid position.
    """
    points = np.asarray(points)
    scalar_field_values = np.asarray(scalar_field_values)
    gx, gy, gz = gradient_fields
    grad = np.stack([np.asarray(gx), np.asarray(gy), np.asarray(gz)], axis=-1)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if scalar_field_values.shape != (len(points),) or grad.shape != points.shape:
        raise ValueError("scalar and gradient fields must contain one value per point")
    if gradient_tolerance < 0 or surface_tolerance < 0:
        raise ValueError("projection tolerances must be non-negative")

    grad_norm_sq = np.sum(grad ** 2, axis=-1)
    residual = scalar_field_values - target_scalar_value
    near_zero_gradient = grad_norm_sq <= gradient_tolerance ** 2
    unprojectable = near_zero_gradient & (np.abs(residual) > surface_tolerance)
    if np.any(unprojectable):
        count = int(np.count_nonzero(unprojectable))
        raise ValueError(f"Cannot project {count} point(s) with near-zero scalar-field gradient")

    safe_grad_norm_sq = np.where(near_zero_gradient, 1.0, grad_norm_sq)
    correction = residual[:, np.newaxis] * grad / safe_grad_norm_sq[:, np.newaxis]
    return points - correction


def cubic_hermite_taper(d: np.ndarray) -> np.ndarray:
    """S(d) = 1 - (3d^2 - 2d^3) for d < 1, else 0."""
    s = 1 - (3 * d ** 2 - 2 * d ** 3)
    return np.where(d < 1, s, 0.0)


def quadratic_taper(d: np.ndarray) -> np.ndarray:
    """S(d) = (1 - d^2)^2 for d < 1, else 0."""
    s = (1 - d ** 2) ** 2
    return np.where(d < 1, s, 0.0)


def spline_taper(d: np.ndarray, control_points: np.ndarray) -> np.ndarray:
    """
    S(d) = B-Spline(d) for d < 1, else 0.
    
    Args:
        d: Normalized distance from the center.
        control_points: Array of shape (N, 2) defining the spline curve.
            X-axis (control_points[:, 0]) should be in range [0, 1].
            Y-axis (control_points[:, 1]) is the multiplier.
    """
    x = control_points[:, 0]
    y = control_points[:, 1]
    k = min(3, len(x) - 1)
    if k < 1:
        # Fallback for single point or empty, though shouldn't happen with valid input
        return np.where(d < 1, y[0] if len(y) > 0 else 0.0, 0.0)
    spline = make_interp_spline(x, y, k=k)
    s = spline(d)
    return np.where(d < 1, s, 0.0)
