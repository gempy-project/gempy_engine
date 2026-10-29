import warnings
from typing import Callable, List, Optional

import numpy as np

from . import _multi_scalar_field_manager as ms
from ._octree_generation import interpolate_on_octree
from ...core import data
from ...core.backend_tensor import BackendTensor
from ...core.data import InterpolationOptions
from ...core.data.engine_grid import EngineGrid
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interp_output import InterpOutput
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.octree_level import OctreeLevel
from ...modules.octrees_topology.octrees_topology_interface import get_next_octree_grid


OctreeLevelCallback = Callable[[int, List[OctreeLevel]], None]


def interpolate_n_octree_levels(interpolation_input: InterpolationInput, options: data.InterpolationOptions,
                                data_descriptor: InputDataDescriptor,
                                on_octree_level: Optional[OctreeLevelCallback] = None) -> List[OctreeLevel]:
    """Evaluate every octree level, refining from the previous one.

    ``on_octree_level(level, octree_list)`` is called once a level is complete (evaluated
    and, unless it is the last one, used to build the next grid), so ``octree_list`` holds
    levels ``0..level``. The list is live: read it inside the callback and do not mutate it.
    An exception raised by the callback aborts the computation.
    """
    n_levels = options.number_octree_levels
    _warn_dense_leaf_level(interpolation_input, options, n_levels)

    octree_list = []
    for i in range(0, n_levels):
        options.temp_interpolation_values.current_octree_level = i
        
        # * Here it goes all the different grids
        next_octree: OctreeLevel = interpolate_on_octree(interpolation_input, options, data_descriptor)
        
        if not options.is_last_octree_level:
            # * This is only a Grid with a Regular grid
            grid_1_centers: EngineGrid = get_next_octree_grid(
                prev_octree=next_octree,
                evaluation_options=options.evaluation_options,
                current_octree_level=i,
                debug=options.debug
            )
            interpolation_input.set_temp_grid(grid_1_centers)
        octree_list.append(next_octree)
        if on_octree_level is not None:
            on_octree_level(i, octree_list)

    interpolation_input.set_grid_to_original()
    return octree_list


def _warn_dense_leaf_level(interpolation_input: InterpolationInput, options: InterpolationOptions, n_levels: int):
    """Every level below ``octree_min_level`` refines all cells: at ``n_levels - 1`` the leaf level is dense."""
    min_level = options.evaluation_options.octree_min_level
    root = interpolation_input.grid.octree_grid
    if n_levels < 2 or min_level < n_levels - 1 or root is None:
        return
    shape = np.asarray(BackendTensor.t.to_numpy(root.regular_grid_shape)).astype(int) * 2 ** (n_levels - 1)
    warnings.warn(
        f"octree_min_level={min_level} with number_octree_levels={n_levels} refines every cell: the leaf level is "
        f"a dense {'x'.join(map(str, shape))} grid ({int(np.prod(shape)):,} cells) and the octree gives no "
        f"speed-up. Use octree_min_level <= {n_levels - 2} to keep the leaf level adaptive.",
        UserWarning, stacklevel=3)


def interpolate_all_fields_no_octree(interpolation_input: InterpolationInput, options: InterpolationOptions,
                                     data_descriptor: InputDataDescriptor) -> List[InterpOutput]:
    """Used in dual contouring"""
    # TODO (miguel March 2026): These copies I do not think are necessary since a while
    # if BackendTensor.engine_backend is not AvailableBackends.PYTORCH and NOT_MAKE_INPUT_DEEP_COPY is False:
    #     temp_interpolation_input = copy.deepcopy(interpolation_input)
    # else:
    #     temp_interpolation_input = interpolation_input

    temp_interpolation_input = interpolation_input
    return ms.interpolate_all_fields(temp_interpolation_input, options, data_descriptor)
