from dataclasses import dataclass
from typing import Union

from gempy_engine.config import BackendConf
import numpy as np

tensor_types = BackendConf.tensor_types


@dataclass
class Orientations:
    dip_positions: np.ndarray
    nugget_effect_grad: Union[np.ndarray, float] = 0.01
    dip_gradients: np.ndarray = None
    dip: np.ndarray = None
    azimuth: np.ndarray = None
    polarity: np.ndarray = None

    def __post_init__(self):
        if type(self.nugget_effect_grad) is float:
            self.nugget_effect_grad = np.ones(self.n_items) * self.nugget_effect_grad

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

