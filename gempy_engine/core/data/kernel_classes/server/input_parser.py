# Check if pydantic is installed and import it
from typing import Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    print("Pydantic is not installed. No server capabilities will be available.")
    BaseModel = object


class SurfacePointsSchema(BaseModel):
    sp_coords: list


class OrientationsSchema(BaseModel):
    dip_positions: list
    dip_gradients: list


class InterpolationInputSchema(BaseModel):
    surface_points: SurfacePointsSchema
    orientations: OrientationsSchema
    grid: Optional[dict] = None # !!TODO: This has to also come from the json


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
