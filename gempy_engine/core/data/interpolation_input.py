import pprint
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import SurfacePoints, Orientations
from gempy_engine.core.data.generic_grid import GenericGrid
from .grid import Grid
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
    grid: Grid

    _unit_values: Optional[np.ndarray] = None
    segmentation_function: Optional[callable] = None  # * From scalar field to values

    _all_surface_points: SurfacePoints = None

    # region per model ? Not sure what I mean here

    _fault_values: FaultsData = None  # ! This field is internal. If you want to pass fault data use StacksStructure
    stack_relation: StackRelationType = StackRelationType.ERODE  # ? Should be here or in the descriptor

    # endregion

    def __init__(self, surface_points: SurfacePoints, orientations: Orientations, grid: Grid,
                 unit_values: Optional[np.ndarray] = None, segmentation_function: Optional[callable] = None,
                 stack_relation: StackRelationType = StackRelationType.ERODE):
        self.surface_points = surface_points
        self.grid = grid
        self.orientations = orientations
        self.unit_values = unit_values
        self.segmentation_function = segmentation_function
        self.stack_relation = stack_relation

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

    @classmethod
    def from_schema(cls, schema: InterpolationInputSchema) -> "InterpolationInput":
        return cls(
            surface_points=SurfacePoints.from_schema(schema.surface_points),
            orientations=Orientations.from_schema(schema.orientations),
            grid=schema.grid,
        )

    @classmethod
    def from_structural_frame(cls, structural_frame: "gempy.StructuralFrame", grid: "gempy.Grid",
                              transform: "gempy.Transfrom", octrees: bool) -> "InterpolationInput":
        _legacy_factor = 0

        if LEGACY_COORDS := False:
            _legacy_factor = 0.5

        surface_points: SurfacePoints = SurfacePoints(
            sp_coords=transform.apply(structural_frame.surface_points.xyz) + _legacy_factor,
            nugget_effect_scalar=structural_frame.surface_points.nugget
        )

        orientations: Orientations = Orientations(
            dip_positions=transform.apply(structural_frame.orientations.xyz) + _legacy_factor,
            dip_gradients=(transform.transform_gradient(structural_frame.orientations.grads)),
            nugget_effect_grad=structural_frame.orientations.nugget
        )
        
        # TODO: if octrees we should deactivate the regular grid resolution?
        if octrees:
            interpolation_resolution = np.array([2, 2, 2])
        else:
            interpolation_resolution = grid.regular_grid.resolution

        # region Transforming the grid

        # DE- BUG
        actual_scaling = transform.apply(grid.regular_grid.values)
        
        transformed = transform.apply(grid.regular_grid.bounding_box)
        new_extents = np.array([transformed[:, 0].min(), transformed[:, 0].max(),
                                transformed[:, 1].min(), transformed[:, 1].max(),
                                transformed[:, 2].min(), transformed[:, 2].max()])
        
        regular_grid: RegularGrid = RegularGrid(
            extent=new_extents,
            regular_grid_shape=interpolation_resolution,
        )
        
        foo = regular_grid.values
        topography_values: GenericGrid = GenericGrid( values=transform.apply(grid.topography.values)) if grid.topography is not None else None
        section_values: GenericGrid = GenericGrid(values=transform.apply(grid.sections.values)) if grid.sections is not None else None
        custom_values: GenericGrid = GenericGrid(values=transform.apply(grid.custom_grid.values)) if grid.custom_grid is not None else None

        grid: Grid = Grid(
            regular_grid=regular_grid,
            topography=topography_values,
            sections=section_values,
            custom_grid=custom_values
        )

        # endregion
        
        interpolation_input: InterpolationInput = cls(
            surface_points=surface_points,
            orientations=orientations,
            grid=grid,
            unit_values=structural_frame.elements_ids  # TODO: Here we will need to pass densities etc.
        )

        return interpolation_input

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
