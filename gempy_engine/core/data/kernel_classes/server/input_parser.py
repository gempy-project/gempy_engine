from pydantic import BaseModel

from gempy_engine.core.data.grid import Grid


class SurfacePointsSchema(BaseModel):
    sp_coords: list


class OrientationsSchema(BaseModel):
    dip_positions: list
    dip_gradients: list


class InterpolationInputSchema(BaseModel):
    surface_points: SurfacePointsSchema
    orientations: OrientationsSchema
    grid: None # !!TODO: This has to also come from the json


class InputDataDescriptorSchema(BaseModel):
    number_of_points_per_surface: list[int]
    number_of_points_per_stack: list[int]
    number_of_orientations_per_stack: list[int]
    number_of_surfaces_per_stack: list[int]
    masking_descriptor: list[int]  # * StackRelationType


class GemPyInput(BaseModel):
    interpolation_input: InterpolationInputSchema
    input_data_descriptor: InputDataDescriptorSchema
