from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import numpy as np

from gempy_engine.config import AvailableBackends
from ..backend_tensor import BackendTensor as b


def _cast_type_inplace(struct_data_instance: "TensorStructure"):
    for key, val in struct_data_instance.__dict__.items():
        if type(val) != np.ndarray: continue
        struct_data_instance.__dict__[key] = val.astype(struct_data_instance.dtype)


@dataclass
class TensorsStructure:
    number_of_points_per_surface: np.ndarray
    dtype: Type = np.int32  # ? Isn't this more for options?

    _reference_sp_position: np.ndarray =  field(default_factory=lambda: np.ones(1))

    def __post_init__(self):  # TODO: Move this to init
        _cast_type_inplace(self)

        # Set _number_of_points_per_surface_vector
        self.number_of_points_per_surface = self.number_of_points_per_surface[self.number_of_points_per_surface != 0]  # remove 0s
        per_surface_cumsum = self.number_of_points_per_surface.cumsum()

        self._reference_sp_position = np.concatenate([np.array([0]), per_surface_cumsum])[:-1]

    def __hash__(self):
        return hash(656)  # TODO: Make a proper hash

    @classmethod
    def from_tensor_structure_subset(cls, data_descriptor: "InputDataDescriptor", stack_number: int) -> TensorsStructure:
        ts = data_descriptor.tensors_structure
        l0 = data_descriptor.stack_structure.number_of_surfaces_per_stack_vector[stack_number]
        l1 = data_descriptor.stack_structure.number_of_surfaces_per_stack_vector[stack_number + 1]

        n_points_per_surface = ts.number_of_points_per_surface[l0:l1]

        return cls(n_points_per_surface, dtype=ts.dtype)

    @property
    def reference_sp_position(self):
        """This is used to find a point on each surface"""
        return self._reference_sp_position

    @property
    def total_number_sp(self):
        return self.number_of_points_per_surface.sum()

    @property
    def n_surfaces(self):
        return self.number_of_points_per_surface.shape[0]

    @property
    def partitions_bool(self):
        ref_positions = self.reference_sp_position

        res = np.eye(self.total_number_sp, dtype='int32')[np.array(ref_positions).reshape(-1)]
        one_hot_ = res.reshape(list(ref_positions.shape) + [self.total_number_sp])
        
        one_hot_ = b.tfnp.array(one_hot_)
            
        partitions = b.tfnp.sum(one_hot_, axis=0, dtype=bool)
        return partitions
