import copy
from typing import List

import numpy as np

from ._experimental_water_tight_DC_1 import _experimental_water_tight
from ._interpolate_on_edges import interpolate_on_edges_for_dual_contouring
from ._mask_buffer import MaskBuffer
from ...core.data import InterpolationOptions
from ...core.data.solutions import Solutions
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interpolation_input import InterpolationInput

from ._dual_contouring import compute_dual_contouring
from ...core.utils import gempy_profiler_decorator
from ...modules.dual_contouring.fancy_triangulation import get_left_right_array


@gempy_profiler_decorator
def dual_contouring_multi_scalar(data_descriptor: InputDataDescriptor, interpolation_input: InterpolationInput,
                                 options: InterpolationOptions, solutions: Solutions) -> List[DualContouringMesh]:
    # Dual Contouring prep:
    MaskBuffer.clean()

    octree_leaves = solutions.octrees_output[-1]
    all_meshes: List[DualContouringMesh] = []

    dual_contouring_options = copy.copy(options)
    dual_contouring_options.compute_scalar_gradient = True

    if options.debug_water_tight:
        _experimental_water_tight(all_meshes, data_descriptor, interpolation_input, octree_leaves, dual_contouring_options)
        return all_meshes

    # region new triangulations
    if options.dual_contouring_fancy:
        is_pure_octree = np.all(solutions.octrees_output[0].grid_centers.regular_grid_shape == 2)
        if not is_pure_octree:  # Check if regular grid is [2,2,2]
            raise ValueError("Fancy triangulation only works with regular grid of resolution [2,2,2]")
        left_right_codes = get_left_right_array(solutions.octrees_output)
    else:
        left_right_codes = None
    # endregion

    for n_scalar_field in range(data_descriptor.stack_structure.n_stacks):
        # @off
        dc_data = interpolate_on_edges_for_dual_contouring(
            data_descriptor     = data_descriptor,
            interpolation_input = interpolation_input,
            n_scalar_field      = n_scalar_field,
            octree_leaves       = octree_leaves,
            options             = dual_contouring_options
        )
        # ? This has been moved to constructor. dc_data.tree_depth = options.number_octree_levels  # TODO: Once we have move to the fancy triangulation. Set this value in a better location

        meshes: List[DualContouringMesh] = compute_dual_contouring(
            dc_data_per_stack = dc_data,
            left_right_codes  = left_right_codes,
            debug             = options.debug
        )
        
        if meshes is not None:
            all_meshes.extend(meshes)
        # @on

    return all_meshes
