"""Orchestrate extent capping and batched boundary scalar evaluation."""

import copy
import os

import numpy as np

from ..interp_single.interp_features import interpolate_all_fields_no_octree
from ...core.backend_tensor import BackendTensor
from ...core.data.engine_grid import EngineGrid
from ...core.data.generic_grid import GenericGrid
from ...core.data.stack_relation_type import StackRelationType
from ...modules.dual_contouring._extent_capping import boundary_lattice, cap_mesh


def cap_meshes_at_extent(all_meshes, dc_data_per_surface, all_mask_arrays, base_number,
                         orthogonal_extent, interpolation_input, options, data_descriptor):
    """Cap extracted meshes in place, sharing boundary samples across surfaces."""
    skip_triangles = os.getenv("GEMPY_SKIP_TRIANGULATION", "0").lower() in ("true", "1", "t", "y", "yes")
    if not all_meshes or skip_triangles:
        return

    extent = BackendTensor.t.to_numpy(orthogonal_extent)
    skip_reasons = []
    stack_relations = data_descriptor.stack_structure.masking_descriptor
    fault_relations = data_descriptor.stack_structure.faults_relations
    for mesh in all_meshes:
        stack = mesh.stack_index
        mask = all_mask_arrays[stack]
        faulted = (stack_relations[stack] is StackRelationType.FAULT
                   or (fault_relations is not None and np.any(fault_relations[:, stack])))
        masked = mask is not None and not bool(mask.all())
        skip_reasons.append(
            "Capping skipped: fault or extraction-mask boundary ownership is not supported"
            if faulted or masked else None
        )
    geometry = None
    coordinates = points = np.empty((0, 3))
    boundary_scalars = None
    if any(reason is None for reason in skip_reasons):
        geometry = boundary_lattice(base_number, extent)
        coordinates, points, _ = geometry
        boundary_scalars = _interp_on_boundary(points, interpolation_input, options, data_descriptor)
    for mesh, dc_data, reason in zip(all_meshes, dc_data_per_surface, skip_reasons):
        cell_coordinates = BackendTensor.t.to_numpy(dc_data.left_right_codes[dc_data.valid_voxels])
        cap_mesh(mesh, cell_coordinates, base_number, extent, coordinates,
                 boundary_scalars[mesh.stack_index] if reason is None else np.empty(0),
                 mesh.isovalue, boundary_geometry=geometry, skip_reason=reason)
        mesh.capping_report["boundary_scalar_points"] = len(points) if reason is None else 0
        mesh.capping_report["original_vertex_count"] = len(cell_coordinates)


def _interp_on_boundary(points, interpolation_input, options, data_descriptor):
    """Evaluate each stack once per boundary batch, reusing it for its surfaces."""
    boundary_options = copy.deepcopy(options)
    boundary_options.evaluation_options.compute_scalar = True
    boundary_options.evaluation_options.compute_scalar_gradient = False
    saved_grid = interpolation_input.grid
    saved_stack = data_descriptor.stack_structure.stack_number
    scalars = [np.empty(len(points)) for _ in range(data_descriptor.stack_structure.n_stacks)]
    batch_size = max(1, int(options.evaluation_options.evaluation_chunk_size))
    try:
        for start in range(0, len(points), batch_size):
            stop = min(start + batch_size, len(points))
            interpolation_input.set_temp_grid(EngineGrid(custom_grid=GenericGrid(
                values=BackendTensor.t.array(points[start:stop], dtype=BackendTensor.dtype)
            )))
            outputs = interpolate_all_fields_no_octree(interpolation_input, boundary_options, data_descriptor)
            for stack_index, output in enumerate(outputs):
                scalars[stack_index][start:stop] = BackendTensor.t.to_numpy(
                    output.exported_fields.scalar_field[output.grid.custom_grid_slice]
                )
    finally:
        interpolation_input.set_temp_grid(saved_grid)
        data_descriptor.stack_structure.stack_number = saved_stack
    return scalars
