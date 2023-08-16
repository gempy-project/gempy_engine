import copy
from typing import Optional, List, Callable

import numpy as np

import gempy_engine.config
from ._interp_scalar_field import interpolate_scalar_field, WeightsBuffer
from ...core.data import SurfacePoints, SurfacePointsInternals, Orientations, OrientationsInternals, TensorsStructure
from ...core.data.exported_fields import ExportedFields
from ...core.data.exported_structs import MaskMatrices
from ...core.data.stack_relation_type import StackRelationType
from ...core.data.internal_structs import SolverInput
from ...core.data.interpolation_functions import CustomInterpolationFunctions
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.kernel_classes.faults import FaultsData
from ...core.data.options import InterpolationOptions
from ...core.data.scalar_field_output import ScalarFieldOutput
from ...modules.activator import activator_interface
from ...modules.data_preprocess import data_preprocess_interface


def interpolate_feature(interpolation_input: InterpolationInput,
                        options: InterpolationOptions,
                        data_shape: TensorsStructure,
                        external_interp_funct: Optional[CustomInterpolationFunctions] = None,
                        external_segment_funct: Optional[Callable[[np.ndarray], float]] = None,
                        clean_buffer: bool = True) -> ScalarFieldOutput:
    
    grid = copy.deepcopy(interpolation_input.grid)

    # region Interpolate scalar field
    solver_input = input_preprocess(data_shape, interpolation_input)
    xyz = solver_input.xyz_to_interpolate

    if external_interp_funct is None:  # * EXTERNAL INTERPOLATION FUNCTION branching
        weights, exported_fields = interpolate_scalar_field(solver_input, options)

        exported_fields.set_structure_values(
            reference_sp_position = data_shape.reference_sp_position,
            slice_feature         = interpolation_input.slice_feature,
            grid_size             = interpolation_input.grid.len_all_grids
        )

        exported_fields.debug = solver_input.debug
    else:
        weights = None
        xyz = grid.values
        exported_fields: ExportedFields = _interpolate_external_function(external_interp_funct, xyz)
        exported_fields.set_structure_values(
            reference_sp_position=None,
            slice_feature=None,
            grid_size=xyz.shape[0]
        )

    # endregion

    # region segmentation
    unit_values = interpolation_input.unit_values
    if external_segment_funct is not None:
        sigmoid_slope = external_segment_funct(xyz)
    else:
        sigmoid_slope = 50000

    values_block: np.ndarray = activator_interface.activate_formation_block(exported_fields, unit_values, sigmoid_slope=sigmoid_slope)

    # endregion
    
    output = ScalarFieldOutput(
        weights=weights,
        grid=grid,
        exported_fields=exported_fields,
        values_block=values_block,
        mask_components=None,
        stack_relation=interpolation_input.stack_relation
    )

    if gempy_engine.config.TENSOR_DTYPE:
        # Check matrices have the right dtype:
        assert values_block.dtype == gempy_engine.config.TENSOR_DTYPE, f"Wrong dtype for values_bloc: {values_block.dtype}. should be {gempy_engine.config.TENSOR_DTYPE}"
        assert exported_fields.scalar_field.dtype == gempy_engine.config.TENSOR_DTYPE, f"Wrong dtype for scalar_field: {exported_fields.scalar_field.dtype}. should be {gempy_engine.config.TENSOR_DTYPE}"

    return output


def input_preprocess(data_shape: TensorsStructure, interpolation_input: InterpolationInput) -> SolverInput:
    grid = interpolation_input.grid
    surface_points: SurfacePoints = interpolation_input.surface_points
    orientations: Orientations = interpolation_input.orientations

    sp_internal: SurfacePointsInternals = data_preprocess_interface.prepare_surface_points(surface_points, data_shape)
    ori_internal: OrientationsInternals = data_preprocess_interface.prepare_orientations(orientations)

    # * We need to interpolate in ALL the surface points not only the surface points of the stack
    grid_internal: np.ndarray = data_preprocess_interface.prepare_grid(grid.values, interpolation_input.all_surface_points)

    fault_values: FaultsData = interpolation_input.fault_values
    faults_on_sp: np.ndarray = fault_values.fault_values_on_sp
    fault_ref, fault_rest = data_preprocess_interface.prepare_faults(faults_on_sp, data_shape)
    fault_values.fault_values_ref, fault_values.fault_values_rest = fault_ref, fault_rest

    solver_input = SolverInput(
        sp_internal=sp_internal,
        ori_internal=ori_internal,
        xyz_to_interpolate=grid_internal,
        fault_internal=fault_values
    )

    return solver_input


def _interpolate_external_function(interp_funct, xyz):
    exported_fields = ExportedFields(
        _scalar_field=interp_funct.implicit_function(xyz),
        _gx_field=interp_funct.gx_function(xyz),
        _gy_field=interp_funct.gy_function(xyz),
        _gz_field=interp_funct.gz_function(xyz),
        _scalar_field_at_surface_points=interp_funct.scalar_field_at_surface_points
    )
    return exported_fields


def _compute_mask_components(exported_fields: ExportedFields, stack_relation: StackRelationType,
                             fault_thickness: Optional[float] = None) -> MaskMatrices:
    # ! This is how I am setting the stackRelation in gempy
    # is_erosion = self.series.df['BottomRelation'].values[self.non_zero] == 'Erosion'
    # is_onlap = np.roll(self.series.df['BottomRelation'].values[self.non_zero] == 'Onlap', 1)
    # ! if len(is_erosion) != 0:
    # !     is_erosion[-1] = False

    # * These are the default values
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
            # TODO [x] Prototyping thickness for faults
            if fault_thickness is not None:
                fault_limit_value = exported_fields.scalar_field_at_surface_points.min()
                thickness_1 = fault_limit_value - fault_thickness
                thickness_2 = fault_limit_value + fault_thickness

                f1 = exported_fields.scalar_field > thickness_1
                f2 = exported_fields.scalar_field < thickness_2

                exported_fields.scalar_field_at_fault_shell = np.array([thickness_1, thickness_2])
                mask_lith = f1 * f2
            else:
                # TODO:  This branch should be like
                # erode_limit_value = exported_fields.scalar_field_at_surface_points.min()
                # mask_lith = exported_fields.scalar_field > erode_limit_value
                
                mask_lith = np.zeros_like(exported_fields.scalar_field)
        case False | StackRelationType.BASEMENT:
            mask_lith = np.ones_like(exported_fields.scalar_field)
        case _:
            raise ValueError("Stack relation type is not supported")

    return MaskMatrices(mask_lith, None)
