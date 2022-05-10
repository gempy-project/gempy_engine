from dataclasses import dataclass

import numpy as np
from scoping import scoping

from gempy_engine.core.data import SurfacePoints, Orientations, TensorsStructure
from gempy_engine.core.data.grid import Grid


@dataclass
class InterpolationInput:
    surface_points: SurfacePoints
    orientations: Orientations
    grid: Grid
    unit_values: np.ndarray
    
    @classmethod
    def from_interpolation_input_subset(cls, interpolation_input: "InterpolationInput", data_structure:TensorsStructure):
        # Instantiate surface points and orientations:
        
        sp = SurfacePoints.from_suraface_points_subset(interpolation_input.surface_points, data_structure)
        o = Orientations.from_orientations_subset(interpolation_input.orientations, data_structure)
        grid = interpolation_input.grid
        
        ts = data_structure
        cum_number_surfaces_l0 = ts.stack_structure.number_of_surfaces_per_stack[:data_structure.stack_number].sum()
        cum_number_surfaces_l1 = ts.stack_structure.number_of_surfaces_per_stack[:data_structure.stack_number + 1].sum() + 1  # * we need to take one unit extra for the basement
        
        unit_values = interpolation_input.unit_values[cum_number_surfaces_l0:cum_number_surfaces_l1]
        
        return cls(sp, o, grid, unit_values)
        