import copy
import warnings
from typing import List

import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor
from ._dual_contouring import compute_dual_contouring
from ._experimental_water_tight_DC_1 import _experimental_water_tight
from ._interpolate_on_edges import interpolate_on_edges_for_dual_contouring
from ._mask_buffer import MaskBuffer
from ...core.data import InterpolationOptions
from ...core.data.dual_contouring_data import DualContouringData
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interp_output import InterpOutput
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.octree_level import OctreeLevel
from ...core.data.options import MeshExtractionMaskingOptions
from ...core.data.stack_relation_type import StackRelationType
from ...core.utils import gempy_profiler_decorator
from ...modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge
from ...modules.dual_contouring.fancy_triangulation import get_left_right_array


@gempy_profiler_decorator
def dual_contouring_multi_scalar(data_descriptor: InputDataDescriptor, interpolation_input: InterpolationInput,
                                 options: InterpolationOptions, octree_list: list[OctreeLevel]) -> List[DualContouringMesh]:
    # Dual Contouring prep:
    MaskBuffer.clean()

    octree_leaves = octree_list[-1]
    all_meshes: List[DualContouringMesh] = []

    dual_contouring_options = copy.deepcopy(options)
    dual_contouring_options.evaluation_options.compute_scalar_gradient = True

    if options.debug_water_tight:
        _experimental_water_tight(all_meshes, data_descriptor, interpolation_input, octree_leaves, dual_contouring_options)
        return all_meshes

    # region new triangulations
    is_pure_octree = bool(np.all(octree_list[0].grid_centers.octree_grid_shape == 2))
    match (options.evaluation_options.mesh_extraction_fancy, is_pure_octree):
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

    all_mask_arrays: np.ndarray = _mask_generation(
        octree_leaves=octree_leaves,
        masking_option=options.evaluation_options.mesh_extraction_masking_options
    )

    all_stack_intersection = []
    all_valid_edges = []
    all_left_right_codes = []

    for n_scalar_field in range(data_descriptor.stack_structure.n_stacks):
        previous_stack_is_onlap = data_descriptor.stack_relation[n_scalar_field - 1] == 'Onlap'
        was_erosion_before = data_descriptor.stack_relation[n_scalar_field - 1] == 'Erosion'
        if previous_stack_is_onlap and was_erosion_before:  # ? (July, 2023) Is this still valid? I thought we have all the combinations
            raise NotImplementedError("Erosion and Onlap are not supported yet")
            pass

        mask: np.ndarray = all_mask_arrays[n_scalar_field]

        if mask is not None and left_right_codes is not None:
            left_right_codes_per_stack = left_right_codes[mask]
        else:
            left_right_codes_per_stack = left_right_codes

        output: InterpOutput = octree_leaves.outputs_centers[n_scalar_field]
        intersection_xyz, valid_edges = find_intersection_on_edge(
            _xyz_corners=octree_leaves.grid_centers.corners_grid.values,
            scalar_field_on_corners=output.exported_fields.scalar_field[output.grid.corners_grid_slice],
            scalar_at_sp=output.scalar_field_at_sp,
            masking=mask
        )

        all_stack_intersection.append(intersection_xyz)
        all_valid_edges.append(valid_edges)
        all_left_right_codes.append(left_right_codes_per_stack)

    from gempy_engine.core.data.engine_grid import EngineGrid
    from gempy_engine.core.data.generic_grid import GenericGrid
    from gempy_engine.API.interp_single.interp_features import interpolate_all_fields_no_octree
    interpolation_input.set_temp_grid(
        EngineGrid(
            custom_grid=GenericGrid(
                values=BackendTensor.t.concatenate(all_stack_intersection, axis=0)
            )
        )
    )
    # endregion

    # ! (@miguel 21 June) I think by definition in the function `interpolate_all_fields_no_octree`
    # ! we just need to interpolate up to the n_scalar_field, but I am not sure about this. I need to test it
    output_on_edges: List[InterpOutput] = interpolate_all_fields_no_octree(
        interpolation_input=interpolation_input,
        options=dual_contouring_options,
        data_descriptor=data_descriptor
    )  # ! This has to be done with buffer weights otherwise is a waste
    interpolation_input.set_grid_to_original()

    for n_scalar_field in range(data_descriptor.stack_structure.n_stacks):
        output: InterpOutput = octree_leaves.outputs_centers[n_scalar_field]
        dc_data = DualContouringData(
            xyz_on_edge=all_stack_intersection[n_scalar_field],
            valid_edges=all_valid_edges[n_scalar_field],
            xyz_on_centers=octree_leaves.grid_centers.octree_grid.values if mask is None else octree_leaves.grid_centers.octree_grid.values[mask],
            dxdydz=octree_leaves.grid_centers.octree_dxdydz,
            exported_fields_on_edges=output_on_edges[n_scalar_field].exported_fields,
            n_surfaces_to_export=output.scalar_field_at_sp.shape[0],
            tree_depth=options.number_octree_levels,
        )
        meshes: List[DualContouringMesh] = compute_dual_contouring(
            dc_data_per_stack=dc_data,
            left_right_codes=all_left_right_codes[n_scalar_field],
            debug=options.debug
        )

        # ! If the order of the meshes does not match the order of scalar_field_at_surface points we need to reorder them HERE

        if meshes is not None:
            all_meshes.extend(meshes)
            # @on

    return all_meshes


def _mask_generation(octree_leaves, masking_option: MeshExtractionMaskingOptions) -> np.ndarray | None:
    all_scalar_fields_outputs: list[InterpOutput] = octree_leaves.outputs_centers
    n_scalar_fields = len(all_scalar_fields_outputs)
    outputs_ = all_scalar_fields_outputs[0]
    slice_corners = outputs_.grid.corners_grid_slice
    grid_size = outputs_.cornersGrid_values.shape[0]
    mask_matrix = BackendTensor.t.zeros((n_scalar_fields, grid_size // 8), dtype=bool)
    onlap_chain_counter = 0

    for i in range(n_scalar_fields):
        stack_relation = all_scalar_fields_outputs[i].scalar_fields.stack_relation
        match (masking_option, stack_relation):
            case MeshExtractionMaskingOptions.RAW, _:
                mask_matrix[i] = BackendTensor.t.ones(grid_size // 8, dtype=bool)
            case MeshExtractionMaskingOptions.DISJOINT, _:
                raise NotImplementedError("Disjoint is not supported yet. Not even sure if there is anything to support")
            # case (MeshExtractionMaskingOptions.DISJOINT | MeshExtractionMaskingOptions.INTERSECT, StackRelationType.FAULT):
            #     mask_matrix[i] = np.ones(grid_size//8, dtype=bool)
            # case MeshExtractionMaskingOptions.DISJOINT, StackRelationType.ERODE | StackRelationType.BASEMENT:
            #     mask_scalar = all_scalar_fields_outputs[i - 1].squeezed_mask_array.reshape((1, -1, 8)).sum(-1, bool)[0]
            #     if MaskBuffer.previous_mask is None:
            #         mask = mask_scalar
            #     else:
            #         mask = (MaskBuffer.previous_mask ^ mask_scalar) * mask_scalar
            #     MaskBuffer.previous_mask = mask
            # case MeshExtractionMaskingOptions.DISJOINT, StackRelationType.ONLAP:
            #     raise NotImplementedError("Onlap is not supported yet")
            #     return octree_leaves.outputs_corners[n_scalar_field].squeezed_mask_array.reshape((1, -1, 8)).sum(-1, bool)[0]
            case MeshExtractionMaskingOptions.INTERSECT, StackRelationType.ERODE:
                x = all_scalar_fields_outputs[i + onlap_chain_counter].squeezed_mask_array[slice_corners].reshape((1, -1, 8))
                mask_matrix[i] = BackendTensor.t.sum(x, -1, bool)[0]
                onlap_chain_counter = 0
            case MeshExtractionMaskingOptions.INTERSECT, StackRelationType.BASEMENT:
                x = all_scalar_fields_outputs[i].squeezed_mask_array[slice_corners].reshape((1, -1, 8))
                mask_matrix[i] = BackendTensor.t.sum(x, -1, bool)[0]
                onlap_chain_counter = 0
            case MeshExtractionMaskingOptions.INTERSECT, StackRelationType.ONLAP:
                x = all_scalar_fields_outputs[i].squeezed_mask_array[slice_corners].reshape((1, -1, 8))
                mask_matrix[i] = BackendTensor.t.sum(x, -1, bool)[0]
                onlap_chain_counter += 1
            case _, StackRelationType.FAULT:
                mask_matrix[i] = BackendTensor.t.ones(grid_size // 8, dtype=bool)
            case _:
                raise ValueError("Invalid combination of options")

    return mask_matrix
