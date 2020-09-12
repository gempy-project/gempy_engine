from dataclasses import dataclass
from typing import Union

import numpy as np
try:
    import tensorflow as tf
    tensor_types = Union[np.ndarray, tf.Tensor, tf.Variable]
except ImportError:
    tensor_types = np.ndarray


@dataclass
class OrientationsInternals:
    dip_positions_tiled: tensor_types = np.empty((0, 3))


@dataclass
class SurfacePointsInternals:
    ref_surface_points: tensor_types = np.empty((0, 3))
    rest_surface_points: tensor_types = np.empty((0, 3))
    ref_nugget: tensor_types = np.empty((0, 1))
    rest_nugget: tensor_types = np.empty((0, 1))


@dataclass
class InterpolationOptions:
    number_dimensions: int = 3

