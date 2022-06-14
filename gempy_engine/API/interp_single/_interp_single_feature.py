import copy
import enum
from typing import Optional

import numpy as np

from ...core.data.exported_structs import ExportedFields, MaskMatrices, ScalarFieldOutput
from ...core.data.input_data_descriptor import StackRelationType, TensorsStructure
from ...core.data.interpolation_functions import InterpolationFunctions, CustomInterpolationFunctions
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.options import KernelOptions

from ...modules.activator import activator_interface
from ._interp_scalar_field import interpolate_scalar_field, Buffer


def interpolate_feature(interpolation_input: InterpolationInput, options: KernelOptions,
                        data_shape: TensorsStructure, interp_funct: Optional[CustomInterpolationFunctions] = None,
                        clean_buffer: bool = True) -> ScalarFieldOutput:
    
    grid = copy.deepcopy(interpolation_input.grid)
    
    if interp_funct is None:
        weights, exported_fields = interpolate_scalar_field(interpolation_input, options, data_shape)
    else:
        weights = None
        xyz = grid.values

        exported_fields = ExportedFields(
            _scalar_field=interp_funct.implicit_function(xyz),
            _gx_field=interp_funct.gx_function(xyz),
            _gy_field=interp_funct.gy_function(xyz),
            _gz_field=interp_funct.gz_function(xyz),
            _scalar_field_at_surface_points=interp_funct.scalar_field_at_surface_points
        )

    values_block = _segment_scalar_field(exported_fields, interpolation_input.unit_values)
    mask_components = _compute_mask_components(exported_fields, interpolation_input.stack_relation)

    output = ScalarFieldOutput(
        weights=weights,
        grid=grid,
        exported_fields=exported_fields,
        values_block=values_block,
        mask_components=mask_components
    )

    if clean_buffer: Buffer.clean()
    return output


def _segment_scalar_field(exported_fields: ExportedFields, unit_values: np.ndarray) -> np.ndarray:
    return activator_interface.activate_formation_block(exported_fields, unit_values, sigmoid_slope=50000)


def _compute_mask_components(exported_fields: ExportedFields, stack_relation: StackRelationType):
    # ! This is how I am setting the stackRelation in gempy
    # is_erosion = self.series.df['BottomRelation'].values[self.non_zero] == 'Erosion'
    # is_onlap = np.roll(self.series.df['BottomRelation'].values[self.non_zero] == 'Onlap', 1)
    # ! if len(is_erosion) != 0:
    # !     is_erosion[-1] = False

    # * This are the default values
    mask_erode = np.ones_like(exported_fields.scalar_field)
    mask_onlap = None  # ! it is the mask of the previous stack (from gempy: mask_matrix[n_series - 1, shift:x_to_interpolate_shape + shift])

    match stack_relation:
        case StackRelationType.ERODE:
            erode_limit_value = exported_fields.scalar_field_at_surface_points.min()
            mask_lith = exported_fields.scalar_field > erode_limit_value
        case StackRelationType.ONLAP:
            onlap_limit_value = exported_fields.scalar_field_at_surface_points.max()
            mask_lith = exported_fields.scalar_field > onlap_limit_value
        case StackRelationType.FAULT:
            mask_lith = np.zeros_like(exported_fields.scalar_field)
        case False:
            mask_lith = np.ones_like(exported_fields.scalar_field)
        case _:
            raise ValueError("Stack relation type is not supported")

    return MaskMatrices(mask_lith, None)
