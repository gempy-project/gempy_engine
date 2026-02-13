import numpy as np

from scipy.interpolate import make_interp_spline

def get_local_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a local orthonormal basis (u, v, w) given a normal vector w.
    w is the normal to the fault surface.
    u is the strike vector (horizontal, perpendicular to w and global Z).
    v is the dip vector (perpendicular to u and w).
    """
    w = normal / np.linalg.norm(normal)
    z_axis = np.array([0, 0, 1.0])
    if np.abs(np.dot(w, z_axis)) > 0.999:
        u = np.cross(w, np.array([1, 0, 0.0]))
    else:
        u = np.cross(w, z_axis)
    u /= np.linalg.norm(u)
    v = np.cross(u, w)
    v /= np.linalg.norm(v)
    return u, v, w

def get_ellipsoid_distance(points: np.ndarray, center: np.ndarray, u: np.ndarray, v: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Calculate normalized distance d from the center in the local (u, v) plane.
    d = sqrt((x_local/a)^2 + (y_local/b)^2)
    """
    relative_points = points - center
    x_local = np.dot(relative_points, u)
    y_local = np.dot(relative_points, v)
    d = np.sqrt((x_local / a) ** 2 + (y_local / b) ** 2)
    return d

def project_points_onto_surface(points: np.ndarray, scalar_field_values: np.ndarray, gradient_fields: tuple[np.ndarray, np.ndarray, np.ndarray], target_scalar_value: float = 0.0) -> np.ndarray:
    """
    Project points onto the surface F(x,y,z) = target_scalar_value.
    Formula: P' = P - (F(P) - target) * grad(F) / ||grad(F)||^2
    """
    gx, gy, gz = gradient_fields
    grad = np.stack([gx, gy, gz], axis=-1)
    grad_norm_sq = np.sum(grad ** 2, axis=-1)
    grad_norm_sq = np.where(grad_norm_sq < 1e-12, 1.0, grad_norm_sq)
    f_p = scalar_field_values - target_scalar_value
    projection = points - 0.5 * (f_p[:, np.newaxis] * grad) / grad_norm_sq[:, np.newaxis]
    return projection

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
    spline = make_interp_spline(x, y, k=3)
    s = spline(d)
    return np.where(d < 1, s, 0.0)
