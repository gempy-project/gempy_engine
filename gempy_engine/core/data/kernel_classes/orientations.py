from dataclasses import dataclass
from typing import Union

from gempy_engine.core.backend_tensor import BackendTensor
import numpy as np

from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.input_data_descriptor import StacksStructure

tensor_types = BackendTensor.tensor_types


@dataclass
class Orientations:
    dip_positions: np.ndarray
    dip_gradients: np.ndarray

    nugget_effect_grad: Union[np.ndarray, float] = 0.01

    def __post_init__(self):
        if type(self.nugget_effect_grad) is float:
            self.nugget_effect_grad = np.ones(self.n_items) * self.nugget_effect_grad
    
    @classmethod
    def from_orientations_subset(cls, orientations: "Orientations", data_structure: StacksStructure):
        stack_n = data_structure.stack_number
        cum_o_l0 = data_structure.nov_stack[stack_n]
        cum_o_l1 = data_structure.nov_stack[stack_n + 1]

        # TODO: Add nugget selection
        o = Orientations(orientations.dip_positions[cum_o_l0:cum_o_l1], orientations.dip_gradients[cum_o_l0:cum_o_l1])
        return o
    
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
    def gx_tiled(self) -> tensor_types:
        return self.gradients_tiled[:, 0]

    @property
    def gy_tiled(self) -> tensor_types:
        return self.gradients_tiled[:, 1]

    @property
    def gz_tiled(self) -> tensor_types:
        return self.gradients_tiled[:, 2]

    @property
    def n_orientations_tiled(self) -> int:
        return self.dip_positions_tiled.shape[0]

    @property
    def n_orientations(self) -> int:
        return int(self.dip_positions_tiled.shape[0]/self.dip_positions_tiled.shape[1])