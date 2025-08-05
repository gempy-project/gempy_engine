import copy
from typing import Optional, Callable

import numpy as np

from gempy_engine.config import AvailableBackends, NOT_MAKE_INPUT_DEEP_COPY
from ...core.backend_tensor import BackendTensor
from ._interp_scalar_field import interpolate_scalar_field
from ...core.data import SurfacePoints, SurfacePointsInternals, Orientations, OrientationsInternals, TensorsStructure
from ...core.data.exported_fields import ExportedFields
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
                        solver_input: SolverInput,
                        external_interp_funct: Optional[CustomInterpolationFunctions] = None,
                        external_segment_funct: Optional[Callable[[np.ndarray], float]] = None,
                        stack_number: Optional[int] = None) -> ScalarFieldOutput:
    
    if BackendTensor.engine_backend is not AvailableBackends.PYTORCH and NOT_MAKE_INPUT_DEEP_COPY is False:
        grid = copy.deepcopy(interpolation_input.grid)
    else:
        grid = interpolation_input.grid

    # region Interpolate scalar field
    xyz = solver_input.xyz_to_interpolate

    if external_interp_funct is None:  # * EXTERNAL INTERPOLATION FUNCTION branching
        weights, exported_fields = interpolate_scalar_field(solver_input, options, stack_number)

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
        sigmoid_slope = options.sigmoid_slope

    values_block: np.ndarray = activator_interface.activate_formation_block(exported_fields, unit_values, sigmoid_slope=sigmoid_slope)
    
    # endregion
    
    output = ScalarFieldOutput(
        weights=weights,
        grid=grid,
        exported_fields=exported_fields,
        values_block=values_block, # TODO: Check value
        stack_relation=interpolation_input.stack_relation
    )

    if BackendTensor.dtype and BackendTensor.engine_backend != AvailableBackends.PYTORCH:
        # Check matrices have the right dtype:
        assert values_block.dtype == BackendTensor.dtype, f"Wrong dtype for values_bloc: {values_block.dtype}. should be {BackendTensor.dtype}"
        assert exported_fields.scalar_field.dtype == BackendTensor.dtype, f"Wrong dtype for scalar_field: {exported_fields.scalar_field.dtype}. should be {BackendTensor.dtype}"

    return output


def input_preprocess(data_shape: TensorsStructure, interpolation_input: InterpolationInput) -> SolverInput:
    grid = interpolation_input.grid
    surface_points: SurfacePoints = interpolation_input.surface_points
    orientations: Orientations = interpolation_input.orientations

    sp_internal: SurfacePointsInternals = data_preprocess_interface.prepare_surface_points(surface_points, data_shape)
    ori_internal: OrientationsInternals = data_preprocess_interface.prepare_orientations(orientations)
    
    # * We need to interpolate in ALL the surface points not only the surface points of the stack
    grid_internal: np.ndarray = data_preprocess_interface.prepare_grid(
        grid=grid.values,
        surface_points=interpolation_input.all_surface_points
    )

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
    solver_input.weights_x0 = interpolation_input.weights

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
