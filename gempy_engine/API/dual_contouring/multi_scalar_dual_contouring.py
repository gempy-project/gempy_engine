from typing import List

import numpy as np

from ...core.data import InterpolationOptions
from ...core.data.solutions import Solutions
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.dual_contouring_data import DualContouringData
from ...core.data.octree_level import OctreeLevel
from ...core.data.interp_output import InterpOutput
from ...core.data.exported_fields import ExportedFields
from ...core.data.grid import Grid
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.options import DualContouringMaskingOptions

from ._dual_contouring import get_intersection_on_edges, compute_dual_contouring
from ..interp_single.interp_features import interpolate_all_fields_no_octree


class MaskBuffer:
    previous_mask = None

    @classmethod
    def clean(cls):
        cls.previous_mask = None


def dual_contouring_multi_scalar(data_descriptor: InputDataDescriptor, interpolation_input: InterpolationInput,
                                 options: InterpolationOptions, solutions: Solutions) -> List[DualContouringMesh]:
    # Dual Contouring prep:
    MaskBuffer.clean()
    octree_leaves = solutions.octrees_output[-1]
    all_meshes: List[DualContouringMesh] = []
    if options.debug_water_tight is False:
        for n_scalar_field in range(data_descriptor.stack_structure.n_stacks):
            dc_data = _independent_dual_contouring(data_descriptor, interpolation_input, n_scalar_field, octree_leaves, options)
            meshes: List[DualContouringMesh] = compute_dual_contouring(dc_data, options.debug)
            all_meshes.append(*meshes)
    else:
        all_dc: List[DualContouringData] = []
        for n_scalar_field in range(data_descriptor.stack_structure.n_stacks):
            dc_data = _independent_dual_contouring(data_descriptor, interpolation_input, n_scalar_field, octree_leaves,
                                                   options)
            all_dc.append(dc_data)

        merged_dc = _merge_dc_data([all_dc[0], all_dc[1]])

        meshes: List[DualContouringMesh] = compute_dual_contouring(merged_dc, options.debug)
        all_meshes.append(*meshes)
        MaskBuffer.clean()

    return all_meshes


def _independent_dual_contouring(data_descriptor: InputDataDescriptor, interpolation_input: InterpolationInput,
                                 n_scalar_field: int, octree_leaves: OctreeLevel, options: InterpolationOptions,
                                 ) -> DualContouringData:
    mask = _mask_generation(n_scalar_field, octree_leaves, options.dual_contouring_masking_options)

    output_corners: InterpOutput = octree_leaves.outputs_corners[n_scalar_field]
    intersection_xyz, valid_edges = get_intersection_on_edges(octree_leaves, output_corners, mask)

    interpolation_input.grid = Grid(intersection_xyz)
    output_on_edges: List[InterpOutput] = interpolate_all_fields_no_octree(interpolation_input, options, data_descriptor)

    dc_data = DualContouringData(
        xyz_on_edge=intersection_xyz,
        valid_edges=valid_edges,
        xyz_on_centers=octree_leaves.grid_centers.values if mask is None else octree_leaves.grid_centers.values[mask],
        dxdydz=octree_leaves.grid_centers.dxdydz,
        exported_fields_on_edges=output_on_edges[n_scalar_field].exported_fields,
        n_surfaces=data_descriptor.stack_structure.number_of_surfaces_per_stack[n_scalar_field]
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


def _merge_dc_data(dc_data_collection: List[DualContouringData]) -> DualContouringData:
    xyz_on_edge = np.vstack([dc_data.xyz_on_edge for dc_data in dc_data_collection])
    valid_edges = np.vstack([dc_data.valid_edges for dc_data in dc_data_collection])
    xyz_on_centers = np.vstack([dc_data.xyz_on_centers for dc_data in dc_data_collection])
    n_surfaces: int = 1  # ! np.sum([dc_data.n_surfaces for dc_data in dc_data_collection]) Not sure if we should keep trying this route

    gx = np.hstack([dc_data.exported_fields_on_edges.gx_field for dc_data in dc_data_collection])
    gy = np.hstack([dc_data.exported_fields_on_edges.gy_field for dc_data in dc_data_collection])
    gz = np.hstack([dc_data.exported_fields_on_edges.gz_field for dc_data in dc_data_collection])

    exported_fields_on_edges = ExportedFields(None, gx, gy, gz)
    dxdydz = dc_data_collection[0].dxdydz

    dc_data = DualContouringData(
        xyz_on_edge=xyz_on_edge,
        valid_edges=valid_edges,
        xyz_on_centers=xyz_on_centers,
        dxdydz=dxdydz,
        exported_fields_on_edges=exported_fields_on_edges,
        n_surfaces=n_surfaces
    )
    return dc_data