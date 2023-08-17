from __future__ import annotations

import pprint
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .kernel_classes.faults import FaultsData
from .stack_relation_type import StackRelationType
from .tensors_structure import TensorsStructure
from .kernel_classes.server.input_parser import InputDataDescriptorSchema
from .stacks_structure import StacksStructure


# noinspection PyArgumentList
@dataclass(frozen=True)
class InputDataDescriptor:
    """
    Class representing a descriptor for input data in a geological model.

    This class provides a structure for the input data, including tensors and stack structure.

    Attributes:
        tensors_structure (TensorsStructure): The structure of tensors used in the model.
        stack_structure (StacksStructure, optional): The structure of stacks used in the model.

    Methods:
        stack_relation (property): Retrieves the masking descriptor from the stack_structure.
        from_schema (classmethod): Constructs an InputDataDescriptor from a given InputDataDescriptorSchema.

    Note:
        This class is immutable, i.e., once an instance is created, it cannot be changed.
    """
    
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
                              faults_relations: Optional[np.ndarray] = None,
                              faults_input_data: Optional[List[FaultsData]] = None
                              ):
        tensor_struct = TensorsStructure(
            number_of_points_per_surface=structural_frame.number_of_points_per_element
        )

        stack_structure = StacksStructure(
            number_of_points_per_stack=structural_frame.number_of_points_per_group,
            number_of_orientations_per_stack=structural_frame.number_of_orientations_per_group,
            number_of_surfaces_per_stack=structural_frame.number_of_elements_per_group,
            masking_descriptor=making_descriptor,
            faults_relations=faults_relations,
            faults_input_data=faults_input_data
        )

        input_data_descriptor = cls(
            tensors_structure=tensor_struct,
            stack_structure=stack_structure
        )

        return input_data_descriptor
