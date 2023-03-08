from typing import List

from gempy_engine.API.dual_contouring._dual_contouring import get_intersection_on_edges
from gempy_engine.API.dual_contouring.mask_buffer import MaskBuffer
from gempy_engine.API.interp_single.interp_features import interpolate_all_fields_no_octree
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.dual_contouring_data import DualContouringData
from gempy_engine.core.data.grid import Grid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.options import DualContouringMaskingOptions
from gempy_engine.core.utils import gempy_profiler_decorator


@gempy_profiler_decorator
def _interpolate_on_edges_for_dual_contouring(data_descriptor: InputDataDescriptor, interpolation_input: InterpolationInput,
                                              n_scalar_field: int, octree_leaves: OctreeLevel, options: InterpolationOptions,
                                              ) -> DualContouringData:
    # TODO: [ ]  _mask_generation is not working with fault StackRelationType

    mask = _mask_generation(n_scalar_field, octree_leaves, options.dual_contouring_masking_options)

    # region define location where we need to interpolate the gradients for dual contouring
    output_corners: InterpOutput = octree_leaves.outputs_corners[n_scalar_field]
    intersection_xyz, valid_edges = get_intersection_on_edges(octree_leaves, output_corners, mask)
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


def _mask_generation(n_scalar_field, octree_leaves, masking_option: DualContouringMaskingOptions):
    match masking_option:
        case DualContouringMaskingOptions.DISJOINT:
            mask_scalar = octree_leaves.outputs_corners[n_scalar_field].squeezed_mask_array.reshape((1, -1, 8)).sum(-1, bool)[0]
            if MaskBuffer.previous_mask is None:
                mask = mask_scalar
            else:
                mask = (MaskBuffer.previous_mask ^ mask_scalar) * mask_scalar
            MaskBuffer.previous_mask = mask
            return mask
        case DualContouringMaskingOptions.INTERSECT:
            mask = octree_leaves.outputs_corners[n_scalar_field].squeezed_mask_array.reshape((1, -1, 8)).sum(-1, bool)[0]
            return mask
        case DualContouringMaskingOptions.RAW:
            return None

