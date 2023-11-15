import copy
from typing import List, Optional

from ...core.backend_tensor import BackendTensor
from ...config import COMPUTE_GRADIENTS, AvailableBackends
from ...core.data.interp_output import InterpOutput
from ...core.data.geophysics_input import GeophysicsInput
from ...modules.geophysics.fw_gravity import compute_gravity
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ..dual_contouring.multi_scalar_dual_contouring import dual_contouring_multi_scalar
from ..interp_single._interp_scalar_field import WeightsBuffer
from ..interp_single.interp_features import interpolate_n_octree_levels
from ...core.data import InterpolationOptions
from ...core.data.solutions import Solutions
from ...core.data.octree_level import OctreeLevel
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interpolation_input import InterpolationInput
from ...core.utils import gempy_profiler_decorator


@gempy_profiler_decorator
def compute_model(interpolation_input: InterpolationInput, options: InterpolationOptions,
                  data_descriptor: InputDataDescriptor, *, geophysics_input: Optional[GeophysicsInput] = None) -> Solutions:
    WeightsBuffer.clean()

    # TODO: Make sure if this works with TF
    # ! If we inline this it seems the deepcopy does not work

    if BackendTensor.engine_backend is AvailableBackends.PYTORCH and COMPUTE_GRADIENTS is False:
        interpolation_input = copy.deepcopy(interpolation_input)

    output: list[OctreeLevel] = interpolate_n_octree_levels(
        interpolation_input=interpolation_input,
        options=options,
        data_descriptor=data_descriptor
    )

    # region Geophysics
    # ---------------------

    # TODO: [ ] Gravity

    # TODO: [ ] Magnetics

    if geophysics_input is not None:
        first_level_last_field: InterpOutput = output[0].outputs_centers[-1]
        gravity = compute_gravity(
            geophysics_input=geophysics_input,
            root_ouput=first_level_last_field
        )
    else:
        gravity = None

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
        fw_gravity=gravity
    )

    if options.debug:
        solutions.debug_input_data["stack_interpolation_input"] = interpolation_input

    return solutions
