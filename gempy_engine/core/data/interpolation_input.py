import pprint
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from . import SurfacePoints, Orientations
from gempy_engine.core.data.generic_grid import GenericGrid
from gempy_engine.core.data.centered_grid import CenteredGrid
from .engine_grid import EngineGrid
from .regular_grid import RegularGrid
from .stack_relation_type import StackRelationType
from .stacks_structure import StacksStructure
from .kernel_classes.faults import FaultsData
from .kernel_classes.server.input_parser import InterpolationInputSchema


@dataclass
class InterpolationInput:
    # @ off
    surface_points: SurfacePoints
    orientations: Orientations
    _original_grid: EngineGrid
    _grid: EngineGrid

    weights: Union[list[np.ndarray] | np.ndarray] = field(default_factory=lambda: [])
    _unit_values: Optional[np.ndarray] = None
    segmentation_function: Optional[callable] = None  # * From scalar field to values

    _all_surface_points: SurfacePoints = None

    # region per model ? Not sure what I mean here

    _fault_values: FaultsData = None  # ! This field is internal. If you want to pass fault data use StacksStructure
    stack_relation: StackRelationType = StackRelationType.ERODE  # ? Should be here or in the descriptor

    # endregion

    def __init__(self, surface_points: SurfacePoints, orientations: Orientations, grid: EngineGrid,
                 unit_values: Optional[np.ndarray] = None, segmentation_function: Optional[callable] = None,
                 stack_relation: StackRelationType = StackRelationType.ERODE, weights: list[np.ndarray] = None):
        if weights is None:
            weights = []
        
        self.surface_points = surface_points
        self._original_grid = grid
        self._grid = grid
        self.orientations = orientations
        self.unit_values = unit_values
        self.segmentation_function = segmentation_function
        self.stack_relation = stack_relation
        self.weights = weights

    # @ on

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    @classmethod
    def from_interpolation_input_subset(cls, all_interpolation_input: "InterpolationInput",
                                        stack_structure: StacksStructure) -> "InterpolationInput":
        """
        This is the constructor used to extract subsets for each feature/series
        """

        stack_number = stack_structure.stack_number

        sp = SurfacePoints.from_suraface_points_subset(all_interpolation_input.surface_points, stack_structure)
        o = Orientations.from_orientations_subset(all_interpolation_input.orientations, stack_structure)

        cum_number_surfaces_l0 = stack_structure.number_of_surfaces_per_stack[:stack_number].sum()
        cum_number_surfaces_l1 = stack_structure.number_of_surfaces_per_stack[:stack_number + 1].sum() + 1  # * we need to take one unit extra for the basement

        unit_values = all_interpolation_input.unit_values[cum_number_surfaces_l0:cum_number_surfaces_l1]

        grid = all_interpolation_input.grid

        # * (miguel 24) This interpolation input goes on the InterpOutput so we are not computing its gradients
        ii_subset: InterpolationInput = cls(
            surface_points=sp,
            orientations=o,
            grid=grid,
            unit_values=unit_values,
            stack_relation=stack_structure.active_masking_descriptor,
            weights=(all_interpolation_input.weights[stack_number] if stack_number < len(all_interpolation_input.weights) else None)
        )

        # ! Setting this on the constructor does not work with data classes.
        ii_subset.fault_values = stack_structure.active_faults_input_data
        ii_subset.all_surface_points = all_interpolation_input.surface_points

        return ii_subset

    @classmethod
    def from_schema(cls, schema: InterpolationInputSchema) -> "InterpolationInput":

        octree_grid = RegularGrid(
            orthogonal_extent=schema.grid.extent,
            regular_grid_shape=np.array([2, 2, 2])
        )
        grid: EngineGrid = EngineGrid(  # * Here we convert the GemPy grid to the
            octree_grid=octree_grid
        )
        return cls(
            surface_points=SurfacePoints.from_schema(schema.surface_points),
            orientations=Orientations.from_schema(schema.orientations),
            grid=grid
        )

    @property
    def original_grid(self):
        return self._original_grid

    def set_grid_to_original(self):
        self._grid = self._original_grid

    @property
    def grid(self):
        return self._grid

    def set_temp_grid(self, value):
        self._grid = value

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
            return self.surface_points  # * This is for backwards compatibility with some tests
        else:
            return self._all_surface_points

    @all_surface_points.setter
    def all_surface_points(self, value):
        self._all_surface_points = value

    @property
    def unit_values(self):
        if self._unit_values is None:
            return np.arange(1000, dtype=np.int16) + 1
        else:
            return self._unit_values.astype(np.int16)

    @unit_values.setter
    def unit_values(self, value):
        self._unit_values = value
