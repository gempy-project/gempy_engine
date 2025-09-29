from typing import List

import numpy as np

from gempy_engine.modules.dual_contouring._dual_contouring import compute_dual_contouring
from gempy_engine.API.dual_contouring._interpolate_on_edges import interpolate_on_edges_for_dual_contouring
from gempy_engine.core.data.dual_contouring_data import DualContouringData
from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.core.data.exported_fields import ExportedFields


def _experimental_water_tight(all_meshes, data_descriptor, interpolation_input, octree_leaves, options):
    all_dc: List[DualContouringData] = []
    for n_scalar_field in range(data_descriptor.stack_structure.n_stacks):
        dc_data = interpolate_on_edges_for_dual_contouring(data_descriptor, interpolation_input, options, n_scalar_field, octree_leaves)
        all_dc.append(dc_data)
    merged_dc = _merge_dc_data([all_dc[0], all_dc[1]])
    meshes: List[DualContouringMesh] = compute_dual_contouring(merged_dc, debug=options.debug)
    all_meshes.append(*meshes)


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
        n_surfaces_to_export=n_surfaces
    )
    return dc_data
