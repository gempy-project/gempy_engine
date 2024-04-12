from dataclasses import dataclass, field
from typing import Union

from gempy_engine.core.backend_tensor import BackendTensor
import numpy as np

from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.core.data.kernel_classes.server.input_parser import OrientationsSchema
from gempy_engine.core.utils import cast_type_inplace

tensor_types = BackendTensor.tensor_types


@dataclass
class Orientations:
    dip_positions: np.ndarray
    dip_gradients: np.ndarray  #: Initialize this re-normalizes implicitly

    nugget_effect_grad: Union[np.ndarray, float] = 0.01

    def __post_init__(self):
        if type(self.nugget_effect_grad) is float or type(self.nugget_effect_grad) is int:
            self.nugget_effect_grad = np.ones(self.n_items) * self.nugget_effect_grad
        cast_type_inplace(self, requires_grad=BackendTensor.COMPUTE_GRADS) # TODO: This has to be grabbed from options
        
        
    def renormalize_gradients(self):
        # * In principle we do this in the Transform class in the main gempy repo
        magnitudes = np.linalg.norm(self.dip_gradients, axis=1, keepdims=True)
        magnitudes[magnitudes == 0] = 1
        self.dip_gradients = self.dip_gradients / magnitudes
    
    @classmethod
    def from_orientations_subset(cls, orientations: "Orientations", data_structure: StacksStructure):
        stack_n = data_structure.stack_number
        cum_o_l0 = data_structure.nov_stack[stack_n]
        cum_o_l1 = data_structure.nov_stack[stack_n + 1]

        # TODO: Add nugget selection
        o = Orientations(
            dip_positions=orientations.dip_positions[cum_o_l0:cum_o_l1], 
            dip_gradients=orientations.dip_gradients[cum_o_l0:cum_o_l1],
            nugget_effect_grad=orientations.nugget_effect_grad[cum_o_l0:cum_o_l1]
        )
        return o

    @classmethod
    def from_schema(cls, schema: OrientationsSchema):
        return cls(dip_positions=np.array(schema.dip_positions), dip_gradients=np.array(schema.dip_gradients))
    
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
    gx: np.array = field(default_factory=lambda: np.empty((0, 3)))
    gy: np.array = field(default_factory=lambda: np.empty((0, 3)))
    gz: np.array = field(default_factory=lambda: np.empty((0, 3)))


@dataclass
class OrientationsInternals:
    orientations: Orientations
    dip_positions_tiled: tensor_types
    gradients_tiled: tensor_types
    nugget_effect_grad: tensor_types

    def __hash__(self):
        return hash(self.__repr__())

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