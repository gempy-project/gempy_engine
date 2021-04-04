from dataclasses import dataclass
from typing import Union

from gempy_engine.core.backend_tensor import BackendTensor
import numpy as np

tensor_types = BackendTensor.tensor_types


@dataclass
class Orientations:
    dip_positions: np.ndarray
    dip_gradients: np.ndarray

    nugget_effect_grad: Union[np.ndarray, float] = 0.01

    def __post_init__(self):
        if type(self.nugget_effect_grad) is float:
            self.nugget_effect_grad = np.ones(self.n_items) * self.nugget_effect_grad

    @property
    def gx(self):
        return self.dip_gradients[:, 0]

    @property
    def gy(self):
        return self.dip_gradients[:, 1]

    @property
    def gz(self):
        return self.dip_gradients[:, 2]

    @property
    def n_dimensions(self):
        return self.dip_positions.shape[1]

    @property
    def n_items(self):
        return self.dip_positions.shape[0]


@dataclass
class OrientationsGradients:
    gx: np.array = np.empty((0, 3))
    gy: np.array = np.empty((0, 3))
    gz: np.array = np.empty((0, 3))


@dataclass
class OrientationsInternals:
    orientations: Orientations
    dip_positions_tiled: tensor_types
    gradients_tiled: tensor_types

    @property
    def gx_tiled(self):
        return self.gradients_tiled[:, 0]

    @property
    def gy_tiled(self):
        return self.gradients_tiled[:, 1]

    @property
    def gz_tiled(self):
        return self.gradients_tiled[:, 2]

    @property
    def n_orientations_tiled(self):
        return self.dip_positions_tiled.shape[0]

    @property
    def n_orientations(self):
        return int(self.dip_positions_tiled.shape[0]/self.dip_positions_tiled.shape[1])