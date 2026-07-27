# Check if pydantic is installed and import it
from typing import Optional

from ...finite_fault import FiniteFault

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


class FaultsDataSchema(BaseModel):
    thickness: Optional[float] = None
    finite_fault: Optional[FiniteFault] = None


class InputDataDescriptorSchema(BaseModel):
    number_of_points_per_surface: list[int]
    number_of_points_per_stack: list[int]
    number_of_orientations_per_stack: list[int]
    number_of_surfaces_per_stack: list[int]
    masking_descriptor: list[int]  # * StackRelationType
    faults_relations: Optional[list[list[bool]]] = None
    faults_input_data: Optional[list[Optional[FaultsDataSchema]]] = None


class GemPyInput(BaseModel):
    interpolation_input: InterpolationInputSchema
    input_data_descriptor: InputDataDescriptorSchema
    interpolation_options: Optional[dict] = None
