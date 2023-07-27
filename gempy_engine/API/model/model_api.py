from __future__ import annotations

import copy
from typing import List, Optional

from core.data.dual_contouring_mesh import DualContouringMesh
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
                  data_descriptor: InputDataDescriptor) -> Solutions:
    
    WeightsBuffer.clean()
    
    interpolation_input = copy.deepcopy(interpolation_input)  # TODO: Make sure if this works with TF
    output: list[OctreeLevel] = _interpolate(interpolation_input, options, data_descriptor)
    
    meshes: Optional[list[DualContouringMesh]] = None
    if options.dual_contouring:
        meshes: list[DualContouringMesh] = dual_contouring_multi_scalar(
            data_descriptor=data_descriptor,
            interpolation_input=interpolation_input,
            options=options,
            octree_list= output
        )
        
    # ---------------------
    # TODO: [ ] Gravity

    # TODO: [ ] Magnetics
    
    solutions = Solutions(
        octrees_output=output,
        dc_meshes=meshes,
    )
    
    
    return solutions


def _interpolate(stack_interpolation_input: InterpolationInput, options: InterpolationOptions,
                 data_descriptor: InputDataDescriptor) -> list[OctreeLevel]:
    output: List[OctreeLevel] = interpolate_n_octree_levels(stack_interpolation_input, options, data_descriptor)
    return output
    solutions: Solutions = Solutions(octrees_output=output)
    
    if options.debug:
        solutions.debug_input_data["stack_interpolation_input"] = stack_interpolation_input
    return solutions
