from typing import List

from gempy_engine.core.data import InterpolationOptions, TensorsStructure
from gempy_engine.core.data.exported_structs import OctreeLevel
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.integrations.interp_single.interp_single_interface import \
    interpolate_single_scalar, compute_n_octree_levels



def interpolate_model(interpolation_input: InterpolationInput, options: InterpolationOptions,
                      data_shape: TensorsStructure):

    # TODO: [ ] Looping scalars
    s = interpolate_single_scalar(interpolation_input, options, data_shape)
    number_octree_levels = options.number_octree_levels
    output: List[OctreeLevel] = compute_n_octree_levels(number_octree_levels, interpolation_input,
                                                       options, data_shape)

    # --------------------
    # The following operations are applied on the FINAL lith block:

    # This should happen only on the leaf of an octree
    # TODO: [ ] Dual contouring. This method only make one vertex per voxel. It is possible to make water tight surfaces?
    #compute_dual_contouring

    # ---------------------
    # TODO: [ ] Gravity

    # TODO: [ ] Magnetics
    return output