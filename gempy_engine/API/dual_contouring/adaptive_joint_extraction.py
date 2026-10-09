"""Opt-in production bridge to the CPU float64 joint adaptive extractor."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from ..interp_single.interp_features import interpolate_all_fields_no_octree
from ..interp_single._interp_single_feature import input_preprocess, interpolate_feature_with_external_function
from ..interp_single._interp_scalar_field import _evaluate_sys_eq
from ...core.data import TensorsStructure
from ...core.data.interpolation_input import InterpolationInput
from ...core.backend_tensor import BackendTensor as BT
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.engine_grid import EngineGrid
from ...core.data.generic_grid import GenericGrid
from ...core.data.options.evaluation_options import MeshExtentCapping, MeshExtractionMaskingOptions
from ...core.data.stack_relation_type import StackRelationType
from ...core.data.options import InterpolationOptions
from ...modules.weights_cache.weights_cache_interface import WeightCache
from ...modules.weights_cache.weight_cache_policy import resolve_weight_cache, WeightCacheRoute
from ...modules.solver.interpolation_solver import pykeops_solver_requested
from ...modules.evaluator.micro_correction import fit_micro_fields, micro_evaluation_options


def _numpy(values):
    return np.asarray(BT.t.to_numpy(values))


def _collect_joint_leaves(octree_list, n_stacks):
    if not octree_list:
        raise ValueError("joint requires octree levels")
    root = octree_list[0].grid.octree_grid
    if root is None:
        raise ValueError("joint requires an octree grid at every level")
    extent = _numpy(root.orthogonal_extent).astype(np.float64)
    root_shape = _numpy(root.regular_grid_shape).astype(np.int64)
    domain_shape = root_shape * 2 ** (len(octree_list) - 1)
    origins, spans, samples = [], [], []
    metadata = None
    for level_index, level in enumerate(octree_list):
        grid = level.grid.octree_grid
        if grid is None:
            raise ValueError("joint requires an octree grid at every level")
        if not np.array_equal(_numpy(grid.orthogonal_extent), extent):
            raise ValueError("joint requires the same extent at every octree level")
        if len(level.outputs) != n_stacks:
            raise ValueError("joint requires one output per stack at every octree level")
        coords = _numpy(grid.integer_coordinates).astype(np.int64)
        count = len(grid.values)
        if coords.shape != (count, 3):
            raise ValueError("joint octree coordinate rows do not match cells")
        shape = root_shape * 2 ** level_index
        spacing = (extent[1::2] - extent[::2]) / shape
        physical = (_numpy(grid.values) - extent[::2]) / spacing - 0.5
        canonical = np.rint(physical).astype(np.int64)
        if not np.allclose(physical, canonical, rtol=0, atol=1e-7):
            raise ValueError("joint octree centers are not on the canonical lattice")
        # Signed/offset integer conventions must differ by a constant per level.
        offset = canonical - coords
        if count and not np.all(offset == offset[0]):
            raise ValueError("joint integer coordinates disagree with physical centers")
        keep = np.ones(count, dtype=bool)
        if level_index + 1 < len(octree_list):
            refined = _numpy(octree_list[level_index + 1].grid.octree_grid.active_cells)
            if refined.dtype != np.bool_ or refined.size != count:
                raise ValueError("joint next-level active_cells must be a parent refinement mask")
            keep = ~refined.reshape(-1)
        level_metadata, level_samples = [], []
        for stack_index, output in enumerate(level.outputs):
            isovalues = _numpy(output.scalar_field_at_sp).reshape(-1)
            raw = _numpy(output.exported_fields.scalar_field[output.grid.corners_grid_slice])
            if raw.size != count * 8:
                raise ValueError("joint requires eight raw corner samples per cell")
            for surface_index, isovalue in enumerate(isovalues):
                level_metadata.append((stack_index, surface_index, float(isovalue)))
                level_samples.append(raw.reshape(count, 8)[keep])
        if metadata is None:
            metadata = level_metadata
        elif (len(metadata) != len(level_metadata) or
              any(a[:2] != b[:2] or not np.isclose(a[2], b[2], rtol=0, atol=1e-12)
                  for a, b in zip(metadata, level_metadata))):
            raise ValueError("joint surface metadata/isovalues differ across octree levels")
        if not level_metadata:
            raise ValueError("joint requires at least one exported surface")
        span = 2 ** (len(octree_list) - 1 - level_index)
        origins.append(canonical[keep] * span)
        spans.append(np.full(np.count_nonzero(keep), span, dtype=np.int64))
        samples.append(np.asarray(level_samples, dtype=np.float64).reshape(len(level_metadata), -1, 8))
    return (np.concatenate(origins), np.concatenate(spans), domain_shape, extent,
            np.concatenate(samples, axis=1), metadata)


def _check_joint_options(descriptor, options, allow_gpu=False):
    evaluation = options.evaluation_options
    if MeshExtentCapping(evaluation.mesh_extraction_extent_capping) != MeshExtentCapping.NONE:
        raise ValueError("joint extent capping is unsupported")
    if evaluation.mesh_extraction_masking_options != MeshExtractionMaskingOptions.INTERSECT:
        raise ValueError("joint only supports INTERSECT mesh masking")
    stacks = descriptor.stack_structure
    relations = stacks.masking_descriptor
    if StackRelationType.NULL_SPACE in relations:
        raise NotImplementedError("joint null-space stacks are unsupported")
    if BT.use_gpu and not allow_gpu:
        raise NotImplementedError("joint requires a CPU backend; CUDA fallback is disabled")
    if BT.dtype != "float64":
        raise NotImplementedError("joint requires float64 production fields")


def _has_faults(descriptor, interpolation_input):
    stacks = descriptor.stack_structure
    return (StackRelationType.FAULT in stacks.masking_descriptor or
            (stacks.faults_relations is not None and np.any(_numpy(stacks.faults_relations))) or
            (stacks.faults_input_data is not None and any(bank is not None for bank in stacks.faults_input_data)) or
            interpolation_input._fault_values is not None)


def extract_joint_contacts_octree(descriptor, interpolation_input, options, octree_list):
    """Joint erosion/onlap contacts with pretty's fault vertex merge; no fault banks.

    Without faults this is exactly ``joint``.
    """
    if not _has_faults(descriptor, interpolation_input):
        return extract_joint_octree(descriptor, interpolation_input, options, octree_list)
    _check_joint_options(descriptor, options, allow_gpu=True)
    from .adaptive_fault_extraction import extract_fault_merge_octree
    return extract_fault_merge_octree(descriptor, interpolation_input, options, octree_list)


def extract_joint_octree(descriptor, interpolation_input, options, octree_list):
    """Query private fixed solves; cokriging with COMPUTE_GRADS=True is unsupported."""
    _check_joint_options(descriptor, options)
    stacks = descriptor.stack_structure
    relations = stacks.masking_descriptor
    if _has_faults(descriptor, interpolation_input):
        from .adaptive_fault_extraction import extract_fault_joint_octree
        return extract_fault_joint_octree(descriptor, interpolation_input, options, octree_list)
    origins, spans, domain_shape, extent, samples, metadata = _collect_joint_leaves(
        octree_list, stacks.n_stacks
    )
    surface_to_stack = np.array([m[0] for m in metadata], dtype=np.int64)
    surface_indices = np.array([m[1] for m in metadata], dtype=np.int64)
    isovalues = [np.array([m[2] for m in metadata if m[0] == stack_index], dtype=np.float64)
                 for stack_index in range(stacks.n_stacks)]
    query_input = copy.copy(interpolation_input)
    # Retain the final production solve, not weights re-solved on query points.
    query_input.weights = [output.weights for output in octree_list[-1].outputs]
    query_options = copy.deepcopy(options)
    query_options.evaluation_options.compute_scalar = True
    query_options.evaluation_options.compute_scalar_gradient = True
    query_descriptor = replace(descriptor, stack_structure=copy.copy(stacks))
    query_descriptor.stack_structure.faults_input_data = (
        None if stacks.faults_input_data is None else list(stacks.faults_input_data))
    if stacks.interpolation_options_per_stack is not None:
        query_descriptor.stack_structure.interpolation_options_per_stack = copy.deepcopy(
            stacks.interpolation_options_per_stack)
        for override in query_descriptor.stack_structure.interpolation_options_per_stack:
            if override is not None:
                override.evaluation_options.compute_scalar = True
                override.evaluation_options.compute_scalar_gradient = True
    fixed_weights = {}
    for stack_index in range(stacks.n_stacks):
        external = stacks.interp_functions_per_stack
        if external is not None and external[stack_index] is not None:
            continue
        overrides = query_descriptor.stack_structure.interpolation_options_per_stack
        stack_options = overrides[stack_index] if overrides and overrides[stack_index] is not None else query_options
        if BT.COMPUTE_GRADS:
            raise NotImplementedError("joint cokriging with COMPUTE_GRADS=True is unsupported")
        if (stack_options.cache_mode not in (
                InterpolationOptions.CacheMode.CACHE, InterpolationOptions.CacheMode.IN_MEMORY_CACHE) or
                stack_options.temp_interpolation_values.start_computation_ts == -1 or
                WeightCache.load_weights(f"{stack_options.cache_model_name}.{stack_index}", False) is None):
            raise NotImplementedError("joint requires cached fixed production weights for cokriging")
        stack_options.cache_mode = InterpolationOptions.CacheMode.IN_MEMORY_CACHE
        query_descriptor.stack_structure.stack_number = stack_index
        subset = InterpolationInput.from_interpolation_input_subset(query_input, query_descriptor.stack_structure)
        shape = TensorsStructure.from_tensor_structure_subset(query_descriptor, stack_index)
        solver_input = input_preprocess(shape, subset)
        decision = resolve_weight_cache(stack_options, stack_index, pykeops_solver_requested(), solver_input)
        if decision.route is not WeightCacheRoute.CACHED:
            raise ValueError(f"joint fixed production weights fingerprint mismatch for stack {stack_index}; no solve allowed")
        weights = decision.weights
        if hasattr(weights, "detach"):
            weights = weights.detach().clone()
        else:
            weights = np.array(weights, copy=True)
            weights.flags.writeable = False
        fixed_weights[stack_index] = weights
        query_input.weights[stack_index] = weights

    def sample_fields(points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not len(points):
            return np.empty((len(metadata), 0)), np.empty((len(metadata), 0, 3))
        previous_grid = query_input.grid
        try:
            query_input.set_temp_grid(EngineGrid(custom_grid=GenericGrid(
                values=BT.t.array(points, dtype=BT.dtype_obj))))
            if not fixed_weights:
                outputs = interpolate_all_fields_no_octree(query_input, query_options, query_descriptor)
            else:
                outputs = []
                for stack_index in range(stacks.n_stacks):
                    query_descriptor.stack_structure.stack_number = stack_index
                    subset = InterpolationInput.from_interpolation_input_subset(query_input, query_descriptor.stack_structure)
                    overrides = query_descriptor.stack_structure.interpolation_options_per_stack
                    stack_options = overrides[stack_index] if overrides and overrides[stack_index] is not None else query_options
                    if stack_index not in fixed_weights:
                        output = interpolate_feature_with_external_function(
                            subset, stack_options, query_descriptor.stack_structure.interp_function,
                            query_descriptor.stack_structure.segmentation_function)
                        fields = output.exported_fields
                    else:
                        shape = TensorsStructure.from_tensor_structure_subset(query_descriptor, stack_index)
                        solver_input = input_preprocess(shape, subset)
                        fields = _evaluate_sys_eq(solver_input, fixed_weights[stack_index],
                                                 micro_evaluation_options(stack_options, subset), grid=subset.grid)
                        fit_micro_fields(subset, fields, stack_options, shape.number_of_points_per_surface,
                                         solver_input.xyz_to_interpolate, True)
                    # Only raw query fields are needed; segmentation/masking never affects them.
                    outputs.append(SimpleNamespace(exported_fields=fields, grid=subset.grid))
            values, gradients = [], []
            for stack_index in surface_to_stack:
                output = outputs[stack_index]
                fields = output.exported_fields
                slicer = output.grid.custom_grid_slice
                if any(field is None for field in (fields.gx_field, fields.gy_field, fields.gz_field)):
                    raise ValueError("joint requires actual production gradient callbacks")
                values.append(_numpy(fields.scalar_field[slicer]))
                gradients.append(np.stack([_numpy(field[slicer]) for field in
                                           (fields.gx_field, fields.gy_field, fields.gz_field)], axis=-1))
            return np.asarray(values, dtype=np.float64), np.asarray(gradients, dtype=np.float64)
        finally:
            query_input.set_temp_grid(previous_grid)

    from .adaptive_topology_extraction import extract_adaptive_topology
    result = extract_adaptive_topology(
        origins, spans, domain_shape, extent, samples, surface_to_stack, surface_indices,
        relations, isovalues, sample_fields=sample_fields, ownership=None, include_reference=False
    )
    vertices = np.asarray(result["vertices"], dtype=np.float64)
    meshes = []
    for index, (stack_index, surface_index, isovalue) in enumerate(metadata):
        faces = np.asarray(result["faces"][index], dtype=np.int64).reshape(-1, 3)
        ids, inverse = np.unique(faces, return_inverse=True)
        mesh = DualContouringMesh(
            vertices=vertices[ids].copy(), edges=inverse.reshape(-1, 3),
            stack_index=stack_index, surface_index=surface_index,
            exported_surface_index=index, isovalue=isovalue,
            contact_report=result["diagnostics"],
        )
        mesh.joint_vertex_ids = ids
        mesh.joint_vertex_keys = [result["vertex_keys"][i] for i in ids]
        mesh.joint_seam_edges = np.asarray(result["seam_edges"], dtype=np.int64).copy()
        meshes.append(mesh)
    return meshes
