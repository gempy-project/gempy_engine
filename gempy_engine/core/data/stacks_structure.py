from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Iterable

import numpy as np

from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
from gempy_engine.core.data.kernel_classes.faults import FaultsData


@dataclass(frozen=False)
class StacksStructure:
    # @off
    number_of_points_per_stack      : np.ndarray  # * These fields are the same in all copies of TensorStructure
    number_of_orientations_per_stack: np.ndarray
    number_of_surfaces_per_stack    : np.ndarray
    masking_descriptor              : List[StackRelationType | False]
    faults_relations                : Optional[np.ndarray]               = None
    faults_input_data               : Optional[List[FaultsData]]          = None  # ? Shouldn't be private?
    
    # * These two fields are optional but public
    interp_functions_per_stack      : List[CustomInterpolationFunctions] = None
    segmentation_functions_per_stack: Optional[List[Callable[[np.ndarray], float]]] = None

    _number_of_points_per_stack_vector      : np.ndarray = field(default_factory=lambda: np.ones(1))
    _number_of_orientations_per_stack_vector: np.ndarray = field(default_factory=lambda: np.ones(1))
    _number_of_surfaces_per_stack_vector    : np.ndarray = field(default_factory=lambda: np.ones(1))

    stack_number: int = -1

    def __post_init__(self):
                
        self.number_of_points_per_stack       = self.number_of_points_per_stack[self.number_of_points_per_stack != 0]
        self.number_of_orientations_per_stack = self.number_of_orientations_per_stack[self.number_of_orientations_per_stack != 0]
        self.number_of_surfaces_per_stack     = self.number_of_surfaces_per_stack[self.number_of_surfaces_per_stack != 0]
        
        consistent_shapes: bool =  len(self.number_of_points_per_stack) == \
                                   len(self.number_of_orientations_per_stack) == \
                                   len(self.number_of_surfaces_per_stack) == \
                                   len(self.masking_descriptor)

        if not consistent_shapes:
            raise ValueError("Inconsistent shapes in StacksStructure")

        # check fault relations
        if self.faults_relations is not None:
            consistent_shapes = consistent_shapes and self.faults_relations.shape[0] == self.faults_relations.shape[1] == len(self.number_of_points_per_stack)
            if not consistent_shapes:
                # Slice self.faults_relations to the correct shape
                self.faults_relations = self.faults_relations[:len(self.number_of_points_per_stack), :len(self.number_of_points_per_stack)]
        
        per_stack_cumsum                             = self.number_of_points_per_stack.cumsum()
        per_stack_orientation_cumsum                 = self.number_of_orientations_per_stack.cumsum()
        per_stack_surface_cumsum                     = self.number_of_surfaces_per_stack.cumsum()
        self._number_of_points_per_stack_vector       = np.concatenate([np.array([0])                 , per_stack_cumsum])
        self._number_of_orientations_per_stack_vector = np.concatenate([np.array([0])                 , per_stack_orientation_cumsum])
        self._number_of_surfaces_per_stack_vector     = np.concatenate([np.array([0])                 , per_stack_surface_cumsum])
    
    # @on

    @property
    def number_of_surfaces_per_stack_vector(self):
        return self._number_of_surfaces_per_stack_vector
    
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
        return self._number_of_points_per_stack_vector

    @property
    def nov_stack(self):
        return self._number_of_orientations_per_stack_vector

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
