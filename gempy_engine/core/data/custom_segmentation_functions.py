import functools
from typing import Callable

import numpy as np


def ellipsoid_3d_factory(center: np.ndarray, radius: np.ndarray, max_slope: float, min_slope: float) -> callable:
    """
    Implicit 3D ellipsoid.
    """

    implicit_ellipsoid = functools.partial(_implicit_3d_ellipsoid_to_slope,
                                           center=center,
                                           radius=radius,
                                           max_slope=max_slope,
                                           min_slope=min_slope)
    
    return implicit_ellipsoid


def _implicit_3d_ellipsoid_to_slope(xyz: np.ndarray, center: np.ndarray, radius: np.ndarray,
                                    max_slope: float=1000, min_slope: float=1):
    """
    Implicit 3D ellipsoid.
    """
    scalar = - np.sum((xyz - center) ** 2.00 / (radius ** 2),  axis=1) - 1.0
    scalar_shifted = scalar - scalar.min()

    sigmoid_slope = 10 # ? This probably should be also public
    Z_x = scalar_shifted
    drift_0 = 4 # ? Making it a %. It depends on the radius
    scale_0 = max_slope
    scalar_final = scale_0 / (1 + np.exp(-sigmoid_slope * (Z_x - drift_0)))
    return scalar_final
    # # cap scalar
    # shifted_scalar[shifted_scalar > 10] = 10
    # 
    # # map scalar between 0 and 1 but heavily skewed with high values
    # scalar_skewed = np.power(shifted_scalar, 10)
    # 
    # scalar_skewed_scalar = (scalar_skewed - scalar_skewed.min()) / (scalar_skewed.max() - scalar_skewed.min()) * (max_slope - min_slope) + min_slope
    # 
    # return shifted_scalar



    
