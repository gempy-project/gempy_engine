from typing import List, Tuple, Optional

import numpy as np

from ._mask_buffer import MaskBuffer
from ..interp_single.interp_features import interpolate_all_fields_no_octree
from ...core.data import InterpolationOptions
from ...core.data.dual_contouring_data import DualContouringData
from ...core.data.grid import Grid
from ...core.data.input_data_descriptor import InputDataDescriptor, StackRelationType
from ...core.data.interp_output import InterpOutput
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.octree_level import OctreeLevel
from ...core.data.options import DualContouringMaskingOptions
from ...core.utils import gempy_profiler_decorator
from ...modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge


@gempy_profiler_decorator
def interpolate_on_edges_for_dual_contouring(
        data_descriptor: InputDataDescriptor,
        interpolation_input: InterpolationInput,
        options: InterpolationOptions,
        n_scalar_field: int,
        octree_leaves: OctreeLevel,
        mask: Optional[np.ndarray] = None

) -> DualContouringData:
    # region define location where we need to interpolate the gradients for dual contouring
    output_corners: InterpOutput = octree_leaves.outputs_corners[n_scalar_field]
    intersection_xyz, valid_edges = _get_intersection_on_edges(octree_leaves, output_corners, mask)
    interpolation_input.grid = Grid(intersection_xyz)
    # endregion

    # ! (@miguel 21 June) I think by definition in the function `interpolate_all_fields_no_octree`
    # ! we just need to interpolate up to the n_scalar_field, but I am not sure about this. I need to test it
    output_on_edges: List[InterpOutput] = interpolate_all_fields_no_octree(interpolation_input, options, data_descriptor)  # ! This has to be done with buffer weights otherwise is a waste

    # * We need this general way because for example for fault we extract two surfaces from one surface input
    dc_data = DualContouringData(
        xyz_on_edge=intersection_xyz,
        valid_edges=valid_edges,
        xyz_on_centers=octree_leaves.grid_centers.values if mask is None else octree_leaves.grid_centers.values[mask],
        dxdydz=octree_leaves.grid_centers.dxdydz,
        exported_fields_on_edges=output_on_edges[n_scalar_field].exported_fields,
        n_surfaces_to_export=output_corners.scalar_field_at_sp.shape[0],
        tree_depth=options.number_octree_levels,
    )
    return dc_data


# TODO: These two functions could be moved to the module
def _get_intersection_on_edges(octree_level: OctreeLevel, output_corners: InterpOutput, mask: Optional[np.ndarray] = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    # First find xyz on edges:
    xyz_corners = octree_level.grid_corners.values

    scalar_field_corners = output_corners.exported_fields.scalar_field
    scalar_field_at_all_sp = output_corners.scalar_field_at_sp

    intersection_xyz, valid_edges = find_intersection_on_edge(xyz_corners, scalar_field_corners, scalar_field_at_all_sp, mask)
    return intersection_xyz, valid_edges


def _mask_generation(n_scalar_field, octree_leaves, masking_option: DualContouringMaskingOptions) -> np.ndarray | None:
    output_corners: InterpOutput = octree_leaves.outputs_corners[n_scalar_field]
    stack_relation: StackRelationType = output_corners.scalar_fields.stack_relation

    match (masking_option, stack_relation):
        case DualContouringMaskingOptions.RAW:
            return None
        case (DualContouringMaskingOptions.DISJOINT | DualContouringMaskingOptions.INTERSECT, StackRelationType.FAULT):
            return None
        case DualContouringMaskingOptions.DISJOINT, _:
            mask_scalar = output_corners.squeezed_mask_array.reshape((1, -1, 8)).sum(-1, bool)[0]
            if MaskBuffer.previous_mask is None:
                mask = mask_scalar
            else:
                mask = (MaskBuffer.previous_mask ^ mask_scalar) * mask_scalar
            MaskBuffer.previous_mask = mask
            return mask
        case DualContouringMaskingOptions.INTERSECT:
            mask = output_corners.squeezed_mask_array.reshape((1, -1, 8)).sum(-1, bool)[0]
            return mask
