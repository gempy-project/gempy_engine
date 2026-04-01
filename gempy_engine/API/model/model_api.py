import copy
import time
from typing import Optional, Any

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
        _check_input_validity(interpolation_input, options, data_descriptor) # TODO

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
    first_level_last_field: InterpOutput = output[0].outputs_centers[-1]

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
    pass
