import copy
from typing import Optional, Callable, Any

import numpy as np
from numpy import dtype, ndarray

from gempy_engine.config import AvailableBackends, NOT_MAKE_INPUT_DEEP_COPY
from ._interp_scalar_field import compute_weights, _evaluate_sys_eq
from ...core.backend_tensor import BackendTensor
from ...core.data import SurfacePoints, SurfacePointsInternals, Orientations, OrientationsInternals, TensorsStructure
from ...core.data.engine_grid import EngineGrid
from ...core.data.exported_fields import ExportedFields
from ...core.data.internal_structs import SolverInput
from ...core.data.interpolation_functions import CustomInterpolationFunctions
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.kernel_classes.faults import FaultsData
from ...core.data.options import InterpolationOptions
from ...core.data.scalar_field_output import ScalarFieldOutput
from ...modules.activator import activator_interface
from ...modules.data_preprocess import data_preprocess_interface


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
    eval_options = _options_with_micro_correction(interpolation_input, solver_input, weights, options,
                                                  data_shape.number_of_points_per_surface)
    exported_fields: ExportedFields = _evaluate_sys_eq(solver_input, weights, eval_options, grid=grid)

    exported_fields.set_structure_values(
        reference_sp_position=data_shape.reference_sp_position,
        slice_feature=interpolation_input.slice_feature,
        grid_size=interpolation_input.grid.len_all_grids
    )

    exported_fields.debug = solver_input.debug

    # endregion

    # region segmentation

    output = _segment(exported_fields, external_segment_funct, grid, interpolation_input, options, xyz)
    return output


def _options_with_micro_correction(interpolation_input, solver_input, weights, options, surface_sizes):
    micro_data = interpolation_input.micro_points
    if micro_data is None or not options.evaluation_options.micro_anisotropic.enabled:
        return options
    if len(micro_data.points) == 0:
        local_options = options.model_copy(deep=True)
        local_options.evaluation_options.micro_anisotropic.enabled = False
        return local_options
    if solver_input.fault_internal.n_faults:
        raise NotImplementedError("Authored micro-point correction with fault-coupled stacks is not supported")

    from ...modules.evaluator.micro_anisotropic_evaluator import build_micro_design_matrix

    local_options = options.model_copy(deep=True)
    micro = local_options.evaluation_options.micro_anisotropic
    if (not np.isfinite(micro.kernel_range) or micro.kernel_range <= 0
            or not np.isfinite(micro.nugget) or micro.nugget < 0):
        raise ValueError("micro kernel_range must be positive and nuggets nonnegative and finite")

    points = micro_data.points
    macro_sp = interpolation_input.surface_points.sp_coords
    if BackendTensor.engine_backend is AvailableBackends.PYTORCH:
        import torch
        macro_points = macro_sp.detach().cpu().numpy()
    else:
        macro_points = np.asarray(macro_sp)
    sizes = (surface_sizes.detach().cpu().numpy() if BackendTensor.engine_backend is AvailableBackends.PYTORCH
             else np.asarray(surface_sizes))
    if len(sizes) == 0 or sizes.sum() != len(macro_points):
        raise ValueError("Micro targets require the stack's surface point counts")

    macro_options = options.model_copy(deep=True)
    macro_options.evaluation_options.micro_anisotropic.enabled = False
    macro_xyz = np.vstack((macro_points, points))
    if BackendTensor.engine_backend is AvailableBackends.PYTORCH:
        macro_xyz = torch.as_tensor(macro_xyz, device=weights.device, dtype=weights.dtype)
    proxy = SolverInput(solver_input.sp_internal, solver_input.ori_internal, macro_xyz,
                        solver_input.fault_internal)
    macro_values = _evaluate_sys_eq(proxy, weights, macro_options).scalar_field_everywhere
    if BackendTensor.engine_backend is AvailableBackends.PYTORCH:
        macro_values = macro_values.detach().cpu().numpy()
    means = [np.mean(chunk) for chunk in np.split(macro_values[:len(macro_points)], np.cumsum(sizes)[:-1])]
    residuals = np.asarray(means)[micro_data.surface_indices] - macro_values[len(macro_points):]

    centers = points
    matrices = micro_data.anisotropy_matrices
    if micro.preserve_macro_points:
        # Borrow the nearest authored metric for added macro constraint centers.
        nearest = np.argmin(np.sum((macro_points[:, None, :] - points[None, :, :]) ** 2, axis=2), axis=1)
        centers = np.vstack((points, macro_points))
        matrices = np.concatenate((matrices, matrices[nearest]))
    design = build_micro_design_matrix(centers, centers, matrices, micro.kernel_range, micro.kernel_type)
    design[np.arange(len(points)), np.arange(len(points))] += micro_data.nuggets + micro.nugget
    rhs = np.concatenate((residuals, np.zeros(len(macro_points)))) if micro.preserve_macro_points else residuals
    try:
        fitted = np.linalg.solve(design, rhs)
    except np.linalg.LinAlgError:
        fitted = np.linalg.lstsq(design, rhs, rcond=None)[0]
        import warnings
        warnings.warn("Micro-point system is singular; using least-squares fit", RuntimeWarning, stacklevel=2)
    if not np.isfinite(fitted).all():
        raise ValueError("Micro-point fit produced non-finite weights")
    micro.points = centers
    micro.anisotropy_matrices = matrices
    micro.weights = fitted * micro.strength
    return local_options


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
    xyz = grid.values

    exported_fields: ExportedFields = _interpolate_external_function(
        interp_funct=external_interp_funct,
        xyz=xyz
    )

    exported_fields.set_structure_values(
        reference_sp_position=None,
        slice_feature=None,
        grid_size=xyz.shape[0]
    )
    output = _segment(exported_fields, external_segment_funct, grid, interpolation_input, options, xyz)

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
    fault_values.fault_values_ref, fault_values.fault_values_rest = data_preprocess_interface.prepare_faults(
        faults_values_on_sp=fault_values.fault_values_on_sp,
        tensors_structure=data_shape
    )

    solver_input = SolverInput(
        sp_internal=sp_internal,
        ori_internal=ori_internal,
        xyz_to_interpolate=grid_internal,
        fault_internal=fault_values
    )
    solver_input.weights_x0 = interpolation_input.weights

    return solver_input


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
    exported_fields = ExportedFields(
        _scalar_field=interp_funct.implicit_function(xyz),
        _gx_field=interp_funct.gx_function(xyz) if interp_funct.gx_function is not None else None,
        _gy_field=interp_funct.gy_function(xyz) if interp_funct.gy_function is not None else None,
        _gz_field=interp_funct.gz_function(xyz) if interp_funct.gz_function is not None else None,
        _scalar_field_at_surface_points=interp_funct.scalar_field_at_surface_points
    )
    return exported_fields
