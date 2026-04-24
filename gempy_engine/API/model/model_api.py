import copy
import time
from typing import Optional, Any

import numpy as np

from ..dual_contouring.multi_scalar_dual_contouring import dual_contouring_multi_scalar
from ..interp_single.interp_features import interpolate_n_octree_levels
from ...config import NOT_MAKE_INPUT_DEEP_COPY, AvailableBackends
from ...core.backend_tensor import BackendTensor
from ...core.data import InterpolationOptions
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.geophysics_input import GeophysicsInput
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interp_output import InterpOutput
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.octree_level import OctreeLevel
from ...core.data.solutions import Solutions
from ...core.utils import gempy_profiler_decorator
from ...core.exceptions import GemPyEngineInputError
from ...modules.geophysics.fw_gravity import compute_gravity
from ...modules.geophysics.fw_magnetic import compute_tmi
from ...modules.weights_cache.weights_cache_interface import WeightCache


@gempy_profiler_decorator
def compute_model(interpolation_input: InterpolationInput, options: InterpolationOptions,
                  data_descriptor: InputDataDescriptor, *, geophysics_input: Optional[GeophysicsInput] = None) -> Solutions:
    try:
        WeightCache.initialize_cache_dir()
        options.temp_interpolation_values.start_computation_ts = int(time.time())

        # ! If we inline this it seems the deepcopy does not work
        if BackendTensor.engine_backend is not AvailableBackends.PYTORCH and NOT_MAKE_INPUT_DEEP_COPY is False:
            interpolation_input = copy.deepcopy(interpolation_input)

        # Check input is valid
        _check_input_validity(interpolation_input, options, data_descriptor)  # TODO

        output: list[OctreeLevel] = interpolate_n_octree_levels(
            interpolation_input=interpolation_input,
            options=options,
            data_descriptor=data_descriptor
        )
        # region Geophysics
        # ---------------------

        gravity = None
        magnetics = None
        if geophysics_input is not None:
            gravity, magnetics = _compute_geophysics(geophysics_input, output)

        # endregion

        meshes: Optional[list[DualContouringMesh]] = None
        if options.mesh_extraction:
            meshes: list[DualContouringMesh] = dual_contouring_multi_scalar(
                data_descriptor=data_descriptor,
                interpolation_input=interpolation_input,
                options=options,
                octree_list=output[:options.number_octree_levels_surface]
            )

        solutions = Solutions(
            octrees_output=output,
            dc_meshes=meshes,
            fw_gravity=gravity,
            fw_magnetics=magnetics,
            block_solution_type=options.block_solutions_type
        )

        if options.debug:
            solutions.debug_input_data["stack_interpolation_input"] = interpolation_input
    except Exception as e:
        raise e
    finally:
        options.temp_interpolation_values.start_computation_ts = -1
        WeightCache.clear_cache()
        BackendTensor.clear_gpu_memory()

    return solutions


def _compute_geophysics(geophysics_input: GeophysicsInput, output: list[OctreeLevel]) -> tuple[Any, Any]:
    first_level_last_field: InterpOutput = output[0].outputs[-1]

    # Gravity (optional)
    if getattr(geophysics_input, 'tz', None) is not None and getattr(geophysics_input, 'densities', None) is not None:
        gravity = compute_gravity(
            geophysics_input=geophysics_input,
            root_ouput=first_level_last_field
        )
    else:
        gravity = None

    # Magnetics (optional)
    try:
        if getattr(geophysics_input, 'magnetics_input', None) is not None:
            magnetics = compute_tmi(
                geophysics_input=geophysics_input.magnetics_input,
                root_output=first_level_last_field
            )
        else:
            magnetics = None
    except Exception:
        # Keep gravity working even if magnetics paths are incomplete
        magnetics = None
    return gravity, magnetics


def _check_input_validity(interpolation_input: InterpolationInput, options: InterpolationOptions, data_descriptor: InputDataDescriptor):
    # 1. Check internal consistency of InterpolationInput
    # 1.1 Orientations: dip_positions and dip_gradients must have same number of items
    if interpolation_input.orientations.dip_positions.shape[0] != interpolation_input.orientations.dip_gradients.shape[0]:
        raise GemPyEngineInputError(
            f"Consistency Error: Orientations dip_positions ({interpolation_input.orientations.dip_positions.shape[0]}) "
            f"and dip_gradients ({interpolation_input.orientations.dip_gradients.shape[0]}) must have the same number of items."
        )

    # 2. Check consistency with TensorsStructure
    # 2.1 Total surface points consistency
    expected_sp_count = data_descriptor.tensors_structure.number_of_points_per_surface.sum()
    actual_sp_count = interpolation_input.surface_points.n_points
    if expected_sp_count != actual_sp_count:
        raise GemPyEngineInputError(
            f"Consistency Error: Total surface points in InterpolationInput ({actual_sp_count}) "
            f"does not match the sum of points in TensorsStructure ({expected_sp_count})."
        )

    # 3. Check consistency with StacksStructure
    if data_descriptor.stack_structure is not None:
        # 3.1 Total orientations consistency
        expected_orientations_count = data_descriptor.stack_structure.number_of_orientations_per_stack.sum()
        actual_orientations_count = interpolation_input.orientations.n_items
        if expected_orientations_count != actual_orientations_count:
            raise GemPyEngineInputError(
                f"Consistency Error: Total orientations in InterpolationInput ({actual_orientations_count}) "
                f"does not match the sum of orientations in StacksStructure ({expected_orientations_count})."
            )

        # 3.2 Total surface points in StacksStructure vs TensorsStructure
        expected_sp_from_stacks = data_descriptor.stack_structure.number_of_points_per_stack.sum()
        if expected_sp_from_stacks != expected_sp_count:
            raise GemPyEngineInputError(
                f"Consistency Error: Total points in StacksStructure ({expected_sp_from_stacks}) "
                f"does not match the total points in TensorsStructure ({expected_sp_count})."
            )

        # 3.3 Total surfaces in StacksStructure vs TensorsStructure
        expected_surfaces_from_stacks = data_descriptor.stack_structure.number_of_surfaces_per_stack.sum()
        actual_surfaces_count = data_descriptor.tensors_structure.n_surfaces
        if expected_surfaces_from_stacks != actual_surfaces_count:
            raise GemPyEngineInputError(
                f"Consistency Error: Total surfaces in StacksStructure ({expected_surfaces_from_stacks}) "
                f"does not match the number of surfaces in TensorsStructure ({actual_surfaces_count})."
            )

        # 3.4 Each stack must have at least one surface point
        if (data_descriptor.stack_structure.number_of_points_per_stack == 0).any():
            stack_index = np.where(data_descriptor.stack_structure.number_of_points_per_stack == 0)[0][0]
            raise GemPyEngineInputError(
                f"Validation Error: Stack {stack_index} has no surface points. "
                f"Each stack must have at least one surface point."
            )

        # 3.5 Each stack must have at least one orientation
        if (data_descriptor.stack_structure.number_of_orientations_per_stack == 0).any():
            stack_index = np.where(data_descriptor.stack_structure.number_of_orientations_per_stack == 0)[0][0]
            raise GemPyEngineInputError(
                f"Validation Error: Stack {stack_index} has no orientations. "
                f"Each stack must have at least one orientation."
            )

    # 4. Check surfaces consistency
    # 4.1 Each surface must have at least one surface point
    if (data_descriptor.tensors_structure.number_of_points_per_surface == 0).any():
        surface_index = np.where(data_descriptor.tensors_structure.number_of_points_per_surface == 0)[0][0]
        raise GemPyEngineInputError(
            f"Validation Error: Surface {surface_index} has no surface points. "
            f"Each surface must have at least one surface point."
        )
