from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Iterable, Tuple

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
    interpolation_options_per_stack : Optional[List[InterpolationOptions]] = None

    # * Null-space stack configuration
    ignored_grid_types_per_stack    : Optional[List[Tuple[str, ...]]] = None
    null_space_id: int = -1
    #: Stacks whose region is void (``null_space_id``, no mesh). Their ERODE or ONLAP relation still shapes that region.
    #: The last stack may be null too, with its BASEMENT relation: it voids what lies above its lowest surface and
    #: the basement fills below. A legacy ``NULL_SPACE`` relation becomes ERODE with this flag set; None when no
    #: stack is null.
    null_space_per_stack            : Optional[List[bool]] = None
    #: Elements (units) per stack, which slice ``unit_values``; null stacks own none. None falls back to the surface
    #: count, which leaves out the elements of stacks without points (external functions).
    number_of_elements_per_stack    : Optional[np.ndarray] = None

    _number_of_points_per_stack_vector      : np.ndarray = field(default_factory=lambda: np.ones(1))
    _number_of_orientations_per_stack_vector: np.ndarray = field(default_factory=lambda: np.ones(1))
    _number_of_surfaces_per_stack_vector    : np.ndarray = field(default_factory=lambda: np.ones(1))

    stack_number: int = -1

    def __post_init__(self):
        # Zero-count entries are allowed (e.g. null-space stacks with external functions)
        # They create degenerate cumsum entries [x, x] in the vectors but that's harmless
        
        consistent_shapes: bool =  len(self.number_of_points_per_stack) == \
                                   len(self.number_of_orientations_per_stack) == \
                                   len(self.number_of_surfaces_per_stack) == \
                                   len(self.masking_descriptor)

        if not consistent_shapes:
            raise ValueError("Inconsistent shapes in StacksStructure")

        self._normalize_null_space()
        if self.number_of_elements_per_stack is not None and len(self.number_of_elements_per_stack) != len(self.masking_descriptor):
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

    def _normalize_null_space(self):
        n_stacks = len(self.masking_descriptor)
        flags = [False] * n_stacks if self.null_space_per_stack is None else [bool(f) for f in self.null_space_per_stack]
        if len(flags) != n_stacks:
            raise ValueError("Inconsistent shapes in StacksStructure")
        relations = list(self.masking_descriptor)
        for i, relation in enumerate(relations):
            if relation is StackRelationType.NULL_SPACE:
                relations[i], flags[i] = StackRelationType.ERODE, True
            elif flags[i] and i == n_stacks - 1 and relation in (False, StackRelationType.BASEMENT):
                continue
            elif flags[i] and relation not in (StackRelationType.ERODE, StackRelationType.ONLAP):
                raise ValueError(f"Null-space stack {i} must erode or onlap, or be the last stack, not {relation}.")
        self.masking_descriptor = relations
        self.null_space_per_stack = flags if any(flags) else None

    def is_null_space(self, stack: int) -> bool:
        return self.null_space_per_stack is not None and self.null_space_per_stack[stack]

    def is_bottom_null_space(self, stack: int) -> bool:
        """The last stack is null with its BASEMENT relation: void above its lowest surface, basement below."""
        return (self.is_null_space(stack) and stack == self.n_stacks - 1
                and self.masking_descriptor[stack] in (False, StackRelationType.BASEMENT))

    @property
    def active_is_null_space(self) -> bool:
        return self.is_null_space(self.stack_number)

    @property
    def has_null_space(self) -> bool:
        return self.null_space_per_stack is not None

    @property
    def null_space_flags(self) -> List[bool]:
        return list(self.null_space_per_stack) if self.null_space_per_stack is not None else [False] * self.n_stacks

    @property
    def number_of_units_per_stack(self) -> np.ndarray:
        """Units each stack slices from ``unit_values``: its elements, or its surfaces when they are not given."""
        counts = self.number_of_surfaces_per_stack if self.number_of_elements_per_stack is None else self.number_of_elements_per_stack
        return np.asarray(counts, dtype=np.int64)

    @property
    def unit_offsets(self) -> np.ndarray:
        """Start of each stack's slice of ``unit_values``; null stacks take no units."""
        units = self.number_of_units_per_stack * ~np.array(self.null_space_flags, dtype=bool)
        return np.concatenate([[0], np.cumsum(units)[:-1]])

    @property
    def number_of_surfaces_per_stack_vector(self):
        return self._number_of_surfaces_per_stack_vector

    @property
    def active_masking_descriptor(self) -> StackRelationType:
        return self.masking_descriptor[self.stack_number]

    @property
    def active_ignored_grid_types(self) -> Tuple[str, ...]:
        if self.ignored_grid_types_per_stack is None:
            return ()
        return self.ignored_grid_types_per_stack[self.stack_number]

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

    @property
    def interpolation_options(self):
        if self.interpolation_options_per_stack is None:
            return None
        return self.interpolation_options_per_stack[self.stack_number]
