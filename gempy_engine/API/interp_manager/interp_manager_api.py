from __future__ import annotations

import copy
from typing import List

from ..dual_contouring.multi_scalar_dual_contouring import dual_contouring_multi_scalar
from ..interp_single.interp_single_interface import interpolate_n_octree_levels
from ...core.data import InterpolationOptions
from ...core.data.exported_structs import OctreeLevel, Solutions
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interpolation_input import InterpolationInput


def interpolate_model(interpolation_input: InterpolationInput, options: InterpolationOptions,
                      data_descriptor: InputDataDescriptor) -> Solutions:
    solutions: Solutions = _interpolate(interpolation_input, options, data_descriptor)

    meshes = dual_contouring_multi_scalar(data_descriptor, interpolation_input, options, solutions)
    solutions.dc_meshes = meshes

    # ---------------------
    # TODO: [ ] Gravity

    # TODO: [ ] Magnetics
    return solutions


def _interpolate(stack_interpolation_input: InterpolationInput, options: InterpolationOptions,
                 data_descriptor: InputDataDescriptor) -> Solutions:
    output: List[OctreeLevel] = interpolate_n_octree_levels(stack_interpolation_input, options, data_descriptor)
    solutions: Solutions = Solutions(octrees_output=output)

    if options.debug:
        solutions.debug_input_data = stack_interpolation_input
    return solutions
