from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Type, List, Callable, Optional, Iterable

import numpy as np

from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
from gempy_engine.core.backend_tensor import BackendTensor as b
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.kernel_classes.server.input_parser import InputDataDescriptorSchema


def _cast_type_inplace(struct_data_instance: "TensorStructure"):
    for key, val in struct_data_instance.__dict__.items():
        if type(val) != np.ndarray: continue
        struct_data_instance.__dict__[key] = val.astype(struct_data_instance.dtype)


class StackRelationType(enum.Enum):
    ERODE = enum.auto()
    ONLAP = enum.auto()
    FAULT = enum.auto()


# noinspection PyArgumentList
@dataclass(frozen=True)
class InputDataDescriptor:
    tensors_structure: TensorsStructure
    stack_structure: StacksStructure = None

    @property
    def stack_relation(self) -> StackRelationType | List[StackRelationType]:
        return self.stack_structure.masking_descriptor

    @classmethod
    def from_schema(cls, schema: InputDataDescriptorSchema):
        tensor_structure = TensorsStructure(
            number_of_points_per_surface=np.array(schema.number_of_points_per_surface)
        )
        
        # Convert list of ints into list of StackRelationType
        list_relations: list[StackRelationType] =  [StackRelationType(x) for x in schema.masking_descriptor]
        stack_structure = StacksStructure(
            number_of_points_per_stack=np.array(schema.number_of_points_per_stack),
            number_of_orientations_per_stack=np.array(schema.number_of_orientations_per_stack),
            number_of_surfaces_per_stack=np.array(schema.number_of_surfaces_per_stack),
            masking_descriptor=list_relations
        )
        return cls(tensors_structure=tensor_structure, stack_structure=stack_structure)


@dataclass(frozen=False)
class StacksStructure:
    number_of_points_per_stack: np.ndarray  # * These fields are the same in all copies of TensorStructure
    number_of_orientations_per_stack: np.ndarray
    number_of_surfaces_per_stack: np.ndarray
    masking_descriptor: List[StackRelationType | False]
    faults_input_data: List[FaultsData] = None
    faults_relations: np.ndarray = None
    interp_functions_per_stack: List[CustomInterpolationFunctions] = None

    segmentation_functions_per_stack: Optional[List[Callable[[np.ndarray], float]]] = None

    number_of_points_per_stack_vector: np.ndarray = np.ones(1)
    number_of_orientations_per_stack_vector: np.ndarray = np.ones(1)
    number_of_surfaces_per_stack_vector: np.ndarray = np.ones(1)

    stack_number: int = -1

    def __post_init__(self):
        per_stack_cumsum = self.number_of_points_per_stack.cumsum()
        per_stack_orientation_cumsum = self.number_of_orientations_per_stack.cumsum()
        per_stack_surface_cumsum = self.number_of_surfaces_per_stack.cumsum()
        self.number_of_points_per_stack_vector = np.concatenate([np.array([0]), per_stack_cumsum])
        self.number_of_orientations_per_stack_vector = np.concatenate([np.array([0]), per_stack_orientation_cumsum])
        self.number_of_surfaces_per_stack_vector = np.concatenate([np.array([0]), per_stack_surface_cumsum])

    @property
    def active_masking_descriptor(self) -> StackRelationType:
        return self.masking_descriptor[self.stack_number]

    @property
    def active_faults_input_data(self) -> FaultsData:
        if self.faults_input_data is None:
            self.faults_input_data = [None] * self.n_stacks
        return self.faults_input_data[self.stack_number]

    @property
    def active_faults_relations(self) -> Iterable[bool]:
        if self.faults_relations is None:
            return [False] * self.n_stacks
        return self.faults_relations[:, self.stack_number]

    @property
    def nspv_stack(self):
        return self.number_of_points_per_stack_vector

    @property
    def nov_stack(self):
        return self.number_of_orientations_per_stack_vector

    @property
    def n_stacks(self):
        return self.number_of_points_per_stack.shape[0]

    @property
    def interp_function(self):
        if self.interp_functions_per_stack is None:
            return None
        return self.interp_functions_per_stack[self.stack_number]

    @property
    def segmentation_function(self):
        if self.segmentation_functions_per_stack is None:
            return None
        return self.segmentation_functions_per_stack[self.stack_number]


@dataclass
class TensorsStructure:
    number_of_points_per_surface: np.ndarray
    dtype: Type = np.int32

    _reference_sp_position: np.ndarray = np.ones(1)

    def __post_init__(self):  # TODO: Move this to init
        _cast_type_inplace(self)

        # Set _number_of_points_per_surface_vector
        per_surface_cumsum = self.number_of_points_per_surface.cumsum()

        self._reference_sp_position = np.concatenate([np.array([0]), per_surface_cumsum])[:-1]

    def __hash__(self):
        return hash(656)  # TODO: Make a proper hash

    @classmethod
    def from_tensor_structure_subset(cls, data_descriptor: InputDataDescriptor, stack_number: int) -> TensorsStructure:
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

        partitions = b.tfnp.sum(one_hot_, axis=0, dtype=bool)
        return partitions
