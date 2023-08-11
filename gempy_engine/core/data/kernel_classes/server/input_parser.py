# Check if pydantic is installed and import it
from typing import Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object


class SurfacePointsSchema(BaseModel):
    sp_coords: list


class OrientationsSchema(BaseModel):
    dip_positions: list
    dip_gradients: list


class GridSchema(BaseModel):
    extent: list
    octree_levels: int 
    
    
class InterpolationInputSchema(BaseModel):
    surface_points: SurfacePointsSchema
    orientations: OrientationsSchema
    grid: GridSchema


class InputDataDescriptorSchema(BaseModel):
    number_of_points_per_surface: list[int]
    number_of_points_per_stack: list[int]
    number_of_orientations_per_stack: list[int]
    number_of_surfaces_per_stack: list[int]
    masking_descriptor: list[int]  # * StackRelationType


class GemPyInput(BaseModel):
    interpolation_input: InterpolationInputSchema
    input_data_descriptor: InputDataDescriptorSchema
    interpolation_options: Optional[dict] = None
