from dataclasses import dataclass
from typing import Union

import numpy as np
try:
    import tensorflow as tf
    tensor_types = Union[np.ndarray, tf.Tensor, tf.Variable]
    tensor_types_or_scalar = Union[np.ndarray, tf.Tensor, tf.Variable, float]
except ImportError:
    tensor_types = np.ndarray
    tensor_types_or_scalar = Union[np.ndarray, float]

@dataclass
class OrientationsInternals:
    dip_positions_tiled: tensor_types = np.empty((0, 3))


@dataclass
class SurfacePointsInternals:
    ref_surface_points: tensor_types = np.empty((0, 3))
    rest_surface_points: tensor_types = np.empty((0, 3))
    #ref_nugget: tensor_types = np.empty((0, 1))
    #rest_nugget: tensor_types = np.empty((0, 1))
    nugget_effect_ref_rest: tensor_types_or_scalar = np.empty((0, 1))


@dataclass
class OrientationsGradients:
    gx: np.array = np.empty((0, 3))
    gy: np.array = np.empty((0, 3))
    gz: np.array = np.empty((0, 3))


@dataclass
class InterpolationOptions:
    number_dimensions: int = 3

