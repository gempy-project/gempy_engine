import copy
import warnings
from typing import List

import numpy as np

from ...core.data.octree_level import OctreeLevel
from ._experimental_water_tight_DC_1 import _experimental_water_tight
from ._interpolate_on_edges import interpolate_on_edges_for_dual_contouring, _mask_generation
from ._mask_buffer import MaskBuffer
from ...core.data import InterpolationOptions
from ...core.data.dual_contouring_data import DualContouringData
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interpolation_input import InterpolationInput

from ._dual_contouring import compute_dual_contouring
from ...core.utils import gempy_profiler_decorator
from ...modules.dual_contouring.fancy_triangulation import get_left_right_array


@gempy_profiler_decorator
def dual_contouring_multi_scalar(data_descriptor: InputDataDescriptor, interpolation_input: InterpolationInput,
                                 options: InterpolationOptions, octree_list: list[OctreeLevel]) -> List[DualContouringMesh]:
    # Dual Contouring prep:
    MaskBuffer.clean()

    octree_leaves = octree_list[-1]
    all_meshes: List[DualContouringMesh] = []

    dual_contouring_options = copy.copy(options)
    dual_contouring_options.compute_scalar_gradient = True

    if options.debug_water_tight:
        _experimental_water_tight(all_meshes, data_descriptor, interpolation_input, octree_leaves, dual_contouring_options)
        return all_meshes

    # region new triangulations
    is_pure_octree = bool(np.all(octree_list[0].grid_centers.regular_grid_shape == 2))
    match (options.dual_contouring_fancy, is_pure_octree):
        case (True, True):
            left_right_codes = get_left_right_array(octree_list)
        case (True, False):
            left_right_codes = None
            warnings.warn("Fancy triangulation only works with regular grid of resolution [2,2,2]. Defaulting to regular triangulation")
        case (False, _):
            left_right_codes = None
        case _:
            raise ValueError("Invalid combination of options")
    # endregion

    for n_scalar_field in range(data_descriptor.stack_structure.n_stacks):
        previous_stack_is_onlap = data_descriptor.stack_relation[n_scalar_field - 1] == 'Onlap'
        was_erosion_before = data_descriptor.stack_relation[n_scalar_field - 1] == 'Erosion'
        if previous_stack_is_onlap and was_erosion_before:  # ? (July, 2023) Is this still valid? I thought we have all the combinations
            raise NotImplementedError("Erosion and Onlap are not supported yet")
            pass

        mask: np.ndarray = _mask_generation(
            n_scalar_field=n_scalar_field,
            octree_leaves=octree_leaves,
            masking_option=options.dual_contouring_masking_options
        )
        
        if mask is not None and left_right_codes is not None:
            left_right_codes_per_stack = left_right_codes[mask]
        else:
            left_right_codes_per_stack = left_right_codes

        # @off
        dc_data: DualContouringData = interpolate_on_edges_for_dual_contouring(
            data_descriptor     = data_descriptor,
            interpolation_input = interpolation_input,
            options             = dual_contouring_options,
            n_scalar_field      = n_scalar_field,  
            octree_leaves       = octree_leaves,
            mask                = mask
        )

        meshes: List[DualContouringMesh] = compute_dual_contouring(
            dc_data_per_stack = dc_data,
            left_right_codes  = left_right_codes_per_stack,
            debug             = options.debug
        )
        
        # ! If the order of the meshes does not match the order of scalar_field_at_surface points we need to reorder them HERE

        if meshes is not None:
            all_meshes.extend(meshes)
        # @on

    return all_meshes
