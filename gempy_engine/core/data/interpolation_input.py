import dataclasses
from dataclasses import dataclass
from typing import List

import numpy as np

from . import SurfacePoints, Orientations, FaultsData
from .grid import Grid
from .input_data_descriptor import StackRelationType, StacksStructure


@dataclass
class InterpolationInput:
    surface_points: SurfacePoints
    orientations: Orientations
    grid: Grid
    unit_values: np.ndarray
    fault_values: FaultsData = None
    _fault_values: FaultsData = dataclasses.field(init=True, repr=False, default=None)
    stack_relation: StackRelationType | List[StackRelationType] = StackRelationType.ERODE # ? Should be here or in the descriptor
    
    @classmethod
    def from_interpolation_input_subset(cls, interpolation_input: "InterpolationInput", stack_structure: StacksStructure):

        sp = SurfacePoints.from_suraface_points_subset(interpolation_input.surface_points, stack_structure)
        o = Orientations.from_orientations_subset(interpolation_input.orientations, stack_structure)
        grid = interpolation_input.grid
        
        cum_number_surfaces_l0 = stack_structure.number_of_surfaces_per_stack[:stack_structure.stack_number].sum()
        cum_number_surfaces_l1 = stack_structure.number_of_surfaces_per_stack[:stack_structure.stack_number + 1].sum() + 1  # * we need to take one unit extra for the basement
        
        unit_values = interpolation_input.unit_values[cum_number_surfaces_l0:cum_number_surfaces_l1]
        
        return cls(
            surface_points=sp,
            orientations=o,
            grid=grid,
            unit_values=unit_values,
            stack_relation=stack_structure.active_masking_descriptor
        )
    
    @property
    def fault_values(self):
        if self._fault_values is None:
            return np.zeros((self.surface_points.n_points, 0))
        return self._fault_values
    
    @fault_values.setter
    def fault_values(self, value):
        self._fault_values = value
        
    
