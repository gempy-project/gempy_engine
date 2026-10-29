import copy
from typing import Optional, Callable, Any

import numpy as np
from numpy import dtype, ndarray

from gempy_engine.config import AvailableBackends, NOT_MAKE_INPUT_DEEP_COPY
from ._interp_scalar_field import compute_weights, _evaluate_sys_eq
from ...core.backend_tensor import BackendTensor
from ...core.compute_scope import checking, StackCache, assert_same
from ...core.data import SurfacePoints, SurfacePointsInternals, Orientations, OrientationsInternals, TensorsStructure
from ...core.data.engine_grid import EngineGrid
from ...core.data.exported_fields import ExportedFields
from ...core.data.internal_structs import SolverInput
from ...core.data.interpolation_functions import CustomInterpolationFunctions
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.kernel_classes.faults import FaultsData
from ...core.data.options import InterpolationOptions
from ...core.data.scalar_field_output import ScalarFieldOutput
from ...core.data.stack_relation_type import StackRelationType
from ...modules.activator import activator_interface
from ...modules.data_preprocess import data_preprocess_interface
from ...modules.evaluator.micro_correction import fit_micro_fields, micro_evaluation_options


def interpolate_feature_with_cokrig(interpolation_input: InterpolationInput,
                                    options: InterpolationOptions,
                                    data_shape: TensorsStructure,
                                    solver_input: SolverInput,
                                    external_segment_funct: Optional[Callable[[np.ndarray], float]] = None,
                                    stack_number: Optional[int] = None) -> ScalarFieldOutput:
    if BackendTensor.engine_backend is not AvailableBackends.PYTORCH and NOT_MAKE_INPUT_DEEP_COPY is False:
        grid = copy.deepcopy(interpolation_input.grid)
    else:
        grid = interpolation_input.grid

    # region Interpolate scalar field
    xyz = solver_input.xyz_to_interpolate

    weights = compute_weights(solver_input, stack_number, options)
    # Retain actual evaluation provenance independently of mutable cache storage.
    if isinstance(weights, np.ndarray):
        solved_weights = weights.copy()
        solved_weights.flags.writeable = False
    else:
        solved_weights = weights.clone()  # Preserve the Torch autograd graph.
    exported_fields: ExportedFields = _evaluate_sys_eq(solver_input, weights,
                                                       micro_evaluation_options(options, interpolation_input), grid=grid)
    fit_micro_fields(interpolation_input, exported_fields, options,
                     data_shape.number_of_points_per_surface, xyz, options.compute_scalar_gradient)

    exported_fields.set_structure_values(
        reference_sp_position=data_shape.reference_sp_position,
        slice_feature=interpolation_input.slice_feature,
        grid_size=interpolation_input.grid.len_all_grids,
        macro_reference_size=interpolation_input.macro_reference_size
    )

    exported_fields.debug = solver_input.debug

    # endregion

    # region segmentation

    output = _segment(exported_fields, external_segment_funct, grid, interpolation_input, options, xyz)
    output.weights = solved_weights
    return output


def interpolate_feature_with_external_function(interpolation_input: InterpolationInput,
                                               options: InterpolationOptions,
                                               external_interp_funct: Optional[CustomInterpolationFunctions] = None,
                                               external_segment_funct: Optional[Callable[[np.ndarray], float]] = None,
                                               ) -> ScalarFieldOutput:
    if BackendTensor.engine_backend is not AvailableBackends.PYTORCH and NOT_MAKE_INPUT_DEEP_COPY is False:
        grid = copy.deepcopy(interpolation_input.grid)
    else:
        grid = interpolation_input.grid

    # region Interpolate scalar field
    shared_queries = (interpolation_input.evaluation_micro_points is not None or
                      interpolation_input.stack_relation is StackRelationType.FAULT)
    xyz = (data_preprocess_interface.prepare_grid(grid.values, interpolation_input.all_surface_points,
                                                   interpolation_input.evaluation_micro_points)
           if shared_queries else grid.values)

    exported_fields: ExportedFields = _interpolate_external_function(
        interp_funct=external_interp_funct,
        xyz=xyz
    )

    exported_fields.set_structure_values(
        reference_sp_position=None,
        slice_feature=None,
        grid_size=grid.len_all_grids,
        macro_reference_size=interpolation_input.macro_reference_size if shared_queries else None
    )
    output = _segment(exported_fields, external_segment_funct, grid, interpolation_input, options, xyz)

    return output


def input_preprocess(data_shape: TensorsStructure, interpolation_input: InterpolationInput,
                     cache: Optional[StackCache] = None) -> SolverInput:
    grid = interpolation_input.grid
    sp_internal, ori_internal, fault_values = preprocess_stack_data(data_shape, interpolation_input, cache)

    # * We need to interpolate in ALL the surface points not only the surface points of the stack
    grid_internal: np.ndarray = data_preprocess_interface.prepare_grid(
        grid=grid.values,
        surface_points=interpolation_input.all_surface_points,
        micro_points=interpolation_input.evaluation_micro_points
    )

    solver_input = SolverInput(
        sp_internal=sp_internal,
        ori_internal=ori_internal,
        xyz_to_interpolate=grid_internal,
        fault_internal=fault_values
    )
    solver_input.weights_x0 = interpolation_input.weights
    solver_input.stack_cache = cache

    return solver_input


def preprocess_stack_data(data_shape: TensorsStructure, interpolation_input: InterpolationInput,
                          cache: Optional[StackCache] = None) -> tuple[SurfacePointsInternals, OrientationsInternals, FaultsData]:
    """Surface point and orientation internals, and the fault values at the reference/rest points.

    None of them depends on the grid: within a compute they are built once per stack and reused from
    ``cache``. The fault values are set on the input's own (per-level) ``FaultsData``.
    """
    fault_values: FaultsData = interpolation_input.fault_values
    if cache is not None and cache.sp_internal is not None and not checking():
        fault_values.fault_values_ref, fault_values.fault_values_rest = cache.fault_values_ref, cache.fault_values_rest
        return cache.sp_internal, cache.ori_internal, fault_values

    sp_internal: SurfacePointsInternals = data_preprocess_interface.prepare_surface_points(interpolation_input.surface_points, data_shape)
    ori_internal: OrientationsInternals = data_preprocess_interface.prepare_orientations(interpolation_input.orientations)
    fault_values.fault_values_ref, fault_values.fault_values_rest = data_preprocess_interface.prepare_faults(
        faults_values_on_sp=fault_values.fault_values_on_sp,
        tensors_structure=data_shape
    )
    if cache is None:
        return sp_internal, ori_internal, fault_values
    if cache.sp_internal is not None:  # checking(): compare, then keep serving the cached objects
        for name in ('ref_surface_points', 'rest_surface_points', 'nugget_effect_ref_rest'):
            assert_same(getattr(cache.sp_internal, name), getattr(sp_internal, name), f"sp_internal.{name}")
        for name in ('dip_positions_tiled', 'gradients_tiled', 'nugget_effect_grad'):
            assert_same(getattr(cache.ori_internal, name), getattr(ori_internal, name), f"ori_internal.{name}")
        # Each level reads them from that level's fault-stack evaluation: equal up to float32 rounding.
        assert_same(cache.fault_values_ref, fault_values.fault_values_ref, "fault_values_ref", rtol=1e-6)
        assert_same(cache.fault_values_rest, fault_values.fault_values_rest, "fault_values_rest", rtol=1e-6)
        fault_values.fault_values_ref, fault_values.fault_values_rest = cache.fault_values_ref, cache.fault_values_rest
        return cache.sp_internal, cache.ori_internal, fault_values
    cache.sp_internal, cache.ori_internal = sp_internal, ori_internal
    cache.fault_values_ref, cache.fault_values_rest = fault_values.fault_values_ref, fault_values.fault_values_rest
    return sp_internal, ori_internal, fault_values


def _segment(exported_fields: ExportedFields, external_segment_funct: Callable[[ndarray[tuple[Any, ...], dtype[Any]]], float] | None,
             grid: EngineGrid, interpolation_input: InterpolationInput, options: InterpolationOptions,
             xyz: ndarray[tuple[Any, ...], dtype[Any]]) -> ScalarFieldOutput:
    # endregion

    # region segmentation
    values_block = _scalar_field_segmentation(exported_fields=exported_fields, external_segment_funct=external_segment_funct, unit_values=interpolation_input.unit_values, xyz=xyz, sigmoid_slope=options.sigmoid_slope)

    # endregion

    output = ScalarFieldOutput(
        weights=None,
        grid=grid,
        exported_fields=exported_fields,
        values_block=values_block,  # TODO: Check value
        stack_relation=interpolation_input.stack_relation
    )

    if BackendTensor.dtype and BackendTensor.engine_backend != AvailableBackends.PYTORCH:
        # Check matrices have the right dtype:
        assert values_block.dtype == BackendTensor.dtype, f"Wrong dtype for values_bloc: {values_block.dtype}. should be {BackendTensor.dtype}"
        assert exported_fields.scalar_field.dtype == BackendTensor.dtype, f"Wrong dtype for scalar_field: {exported_fields.scalar_field.dtype}. should be {BackendTensor.dtype}"
    return output


def _scalar_field_segmentation(exported_fields: ExportedFields, external_segment_funct: Callable[[ndarray[tuple[Any, ...], dtype[Any]]], float] | None,
                               unit_values: np.ndarray, xyz: ndarray | None, sigmoid_slope: float) -> ndarray[tuple[Any, ...], dtype[Any]]:
    if external_segment_funct is not None:  # * This branch is used in finite faults
        sigmoid_slope = external_segment_funct(xyz)
    else:
        sigmoid_slope = sigmoid_slope

    values_block: np.ndarray = activator_interface.activate_formation_block(exported_fields, unit_values, sigmoid_slope=sigmoid_slope)
    return values_block


def _interpolate_external_function(interp_funct, xyz):
    def field(func):
        return BackendTensor.t.array(func(xyz), dtype=BackendTensor.dtype) if func is not None else None

    exported_fields = ExportedFields(
        _scalar_field=field(interp_funct.implicit_function),
        _gx_field=field(interp_funct.gx_function),
        _gy_field=field(interp_funct.gy_function),
        _gz_field=field(interp_funct.gz_function),
        _scalar_field_at_surface_points=interp_funct.scalar_field_at_surface_points
    )
    return exported_fields
