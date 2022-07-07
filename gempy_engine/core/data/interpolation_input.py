import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import SurfacePoints, Orientations
from .grid import Grid
from .input_data_descriptor import StackRelationType, StacksStructure
from .kernel_classes.faults import FaultsData


@dataclass
class InterpolationInput:
    surface_points: SurfacePoints
    orientations: Orientations
    grid: Grid
    unit_values: np.ndarray
    segmentation_function: Optional[callable] = None
    
    _all_surface_points: SurfacePoints = None#dataclasses.field(init=True, repr=False, default=None)
    
    # region per model
    
    _fault_values: FaultsData = None
    stack_relation: StackRelationType = StackRelationType.ERODE  # ? Should be here or in the descriptor
    # endregion
    
    @classmethod
    def from_interpolation_input_subset(cls, all_interpolation_input: "InterpolationInput",
                                        stack_structure: StacksStructure) -> "InterpolationInput":
        stack_number = stack_structure.stack_number

        sp = SurfacePoints.from_suraface_points_subset(all_interpolation_input.surface_points, stack_structure)
        o = Orientations.from_orientations_subset(all_interpolation_input.orientations, stack_structure)

        cum_number_surfaces_l0 = stack_structure.number_of_surfaces_per_stack[:stack_number].sum()
        cum_number_surfaces_l1 = stack_structure.number_of_surfaces_per_stack[:stack_number + 1].sum() + 1  # * we need to take one unit extra for the basement

        unit_values = all_interpolation_input.unit_values[cum_number_surfaces_l0:cum_number_surfaces_l1]

        grid = all_interpolation_input.grid

        ii_subset = cls(
            surface_points=sp,
            orientations=o,
            grid=grid,
            unit_values=unit_values,
            stack_relation=stack_structure.active_masking_descriptor,
        )

        # ! Setting this on the constructor does not work with data classes.
        ii_subset.fault_values = stack_structure.active_faults_input_data
        ii_subset.all_surface_points = all_interpolation_input.surface_points
        
        return ii_subset
    
    @property
    def slice_feature(self):
        return self.surface_points.slice_feature
    
    
    @property
    def fault_values(self):
        if self._fault_values is None:
            empty_fault_values_on_sp = np.zeros((0, self.surface_points.n_points))
            empty_fault_values_on_grid = np.zeros((0, self.grid.len_all_grids))
            return FaultsData(empty_fault_values_on_grid, empty_fault_values_on_sp)
        return self._fault_values

    @property
    def not_fault_input(self):
        return self._fault_values is None

    @fault_values.setter
    def fault_values(self, value):
        self._fault_values = value

    @property
    def all_surface_points(self):
        if self._all_surface_points is None:
            return self.surface_points # * This is for backwards compatibility with some tests
        else:
            return self._all_surface_points

    @all_surface_points.setter
    def all_surface_points(self, value):
        self._all_surface_points = value
