from __future__ import annotations

import pprint
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .stack_relation_type import StackRelationType
from .tensors_structure import TensorsStructure
from .kernel_classes.server.input_parser import InputDataDescriptorSchema
from .stacks_structure import StacksStructure


# noinspection PyArgumentList
@dataclass(frozen=True)
class InputDataDescriptor:
    tensors_structure: TensorsStructure
    stack_structure: StacksStructure = None

    def __repr__(self):
        return pprint.pformat(self.__dict__)
    
    @property
    def stack_relation(self) -> StackRelationType | List[StackRelationType]:
        return self.stack_structure.masking_descriptor

    @classmethod
    def from_schema(cls, schema: InputDataDescriptorSchema):
        tensor_structure = TensorsStructure(
            number_of_points_per_surface=np.array(schema.number_of_points_per_surface)
        )

        # Convert list of ints into list of StackRelationType
        list_relations: list[StackRelationType] = [StackRelationType(x) for x in schema.masking_descriptor]
        stack_structure = StacksStructure(
            number_of_points_per_stack=np.array(schema.number_of_points_per_stack),
            number_of_orientations_per_stack=np.array(schema.number_of_orientations_per_stack),
            number_of_surfaces_per_stack=np.array(schema.number_of_surfaces_per_stack),
            masking_descriptor=list_relations
        )
        return cls(tensors_structure=tensor_structure, stack_structure=stack_structure)

    @classmethod
    def from_structural_frame(cls, structural_frame: "gempy.StructuralFrame",
                              making_descriptor: list[StackRelationType | False],
                              faults_relations: Optional[np.ndarray] = None):
        tensor_struct = TensorsStructure(
            number_of_points_per_surface=structural_frame.number_of_points_per_element
        )

        stack_structure = StacksStructure(
            number_of_points_per_stack=structural_frame.number_of_points_per_group,
            number_of_orientations_per_stack=structural_frame.number_of_orientations_per_group,
            number_of_surfaces_per_stack=structural_frame.number_of_elements_per_group,
            masking_descriptor=making_descriptor,
            faults_relations=faults_relations
        )

        input_data_descriptor = cls(
            tensors_structure=tensor_struct,
            stack_structure=stack_structure
        )

        return input_data_descriptor
