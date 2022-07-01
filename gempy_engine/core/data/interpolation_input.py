import dataclasses
from dataclasses import dataclass

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

    all_surface_points: SurfacePoints = None
    _all_surface_points: SurfacePoints = dataclasses.field(init=False, repr=False, default=None)
    
    # region per model
    fault_values: FaultsData = None
    _fault_values: FaultsData = dataclasses.field(init=False, repr=False, default=None)
    stack_relation: StackRelationType = StackRelationType.ERODE  # ? Should be here or in the descriptor
    # endregion
    
    # def __post_init__(self):
    #     if self._fault_values is None:
    #         empty_fault_values_on_sp = np.zeros((0, self.surface_points.n_points))
    #         empty_fault_values_on_grid = np.zeros((0, self.grid.len_all_grids))
    #         self._fault_values = [FaultsData(empty_fault_values_on_grid, empty_fault_values_on_sp)] * len(self.stack_relation)

    @classmethod
    def from_interpolation_input_subset(cls, all_interpolation_input: "InterpolationInput",
                                        stack_structure: StacksStructure) -> "InterpolationInput":

        sp = SurfacePoints.from_suraface_points_subset(all_interpolation_input.surface_points, stack_structure)
        o = Orientations.from_orientations_subset(all_interpolation_input.orientations, stack_structure)

        cum_number_surfaces_l0 = stack_structure.number_of_surfaces_per_stack[:stack_structure.stack_number].sum()
        cum_number_surfaces_l1 = stack_structure.number_of_surfaces_per_stack[:stack_structure.stack_number + 1].sum() + 1  # * we need to take one unit extra for the basement

        unit_values = all_interpolation_input.unit_values[cum_number_surfaces_l0:cum_number_surfaces_l1]

        grid = all_interpolation_input.grid

        ii_subset = cls(
            surface_points=sp,
            orientations=o,
            grid=grid,
            unit_values=unit_values,
            stack_relation=stack_structure.active_masking_descriptor,
            fault_values=stack_structure.active_faults_input_data,
            all_surface_points=all_interpolation_input.surface_points
        )

        #ii_subset.fault_values = all_interpolation_input._fault_values  # ! Setting this on the constructor does not work God knows why.

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

    @fault_values.setter
    def fault_values(self, value):
        self._fault_values = value

    @property
    def all_surface_points(self):
        if self._all_surface_points is None:
            return None
        else:
            return self._all_surface_points

    @all_surface_points.setter
    def all_surface_points(self, value):
        self._all_surface_points = value
