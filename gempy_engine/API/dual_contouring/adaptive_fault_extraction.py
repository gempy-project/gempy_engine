"""Production fixed-weight fault banks, assembled without cross-bank welding."""

import numpy as np

from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.kernel_classes.faults import FaultsData
from ...core.data.stack_relation_type import StackRelationType as R
from ...modules.dual_contouring.adaptive_cell_complex import build_adaptive_complex
from ...modules.dual_contouring.adaptive_closed_volumes import build_closed_volumes
from ...modules.dual_contouring.adaptive_joint_topology import emit_adaptive_triangles
from ...modules.dual_contouring.contact_topology import build_contact_relations
from ...modules.dual_contouring.topology_extraction import CORNERS
from .adaptive_joint_extraction import (_collect_joint_leaves, _numpy, _check_closed_audit, _emit_with_closure,
                                        _assemble_closed_volumes)


class _BankFieldMemo:
    """Deduplicated fixed-weight samples keyed by stack, effective bank and exact point.

    Affected stacks depend on the queried bank; fault and unaffected stacks do
    not, so one evaluation serves both banks. Each point is evaluated once.
    """

    def __init__(self, query, affected):
        self._query, self._affected, self._stores = query, set(affected), {}

    def __call__(self, points, stacks, bank):
        unique, inverse = np.unique(points, axis=0, return_inverse=True)
        keys = list(map(tuple, unique))
        raw = np.empty((len(stacks), len(unique)))
        gradients = np.empty((len(stacks), len(unique), 3))
        for row, stack in enumerate(stacks):
            effective = bank if stack in self._affected else None
            index, values, normals = self._stores.setdefault((stack, effective), ({}, [], []))
            missing = [i for i, key in enumerate(keys) if key not in index]
            if missing:
                v, g = self._query(unique[missing], bank=effective, stacks=(stack,))
                index.update(zip((keys[i] for i in missing), range(len(index), len(index) + len(missing))))
                values.append(np.asarray(v, dtype=np.float64)[0])
                normals.append(np.asarray(g, dtype=np.float64)[0])
            lookup = np.fromiter((index[key] for key in keys), dtype=np.int64, count=len(keys))
            raw[row] = np.concatenate(values)[lookup]
            gradients[row] = np.concatenate(normals)[lookup]
        inverse = inverse.reshape(-1)
        return raw[:, inverse], gradients[:, inverse]


def extract_fault_joint_octree(descriptor, interpolation_input, options, octree_list):
    """Export one mesh per original surface, with two independent fault skins.

    Original mixed-bank corners are metadata only. All extraction samples,
    including leaf corners, use the same immutable production bank sampler.
    """
    stacks = descriptor.stack_structure
    relations = stacks.masking_descriptor
    matrix = (np.zeros((stacks.n_stacks, stacks.n_stacks), dtype=bool)
              if stacks.faults_relations is None else _numpy(stacks.faults_relations))
    if matrix.shape != (stacks.n_stacks, stacks.n_stacks) or not np.isin(matrix, (False, True)).all():
        raise ValueError("unsupported_fault_metadata: invalid directed fault matrix")
    matrix = matrix.astype(bool)
    fault_stacks = {i for i, relation in enumerate(relations) if relation is R.FAULT}
    fault_stacks.update(np.flatnonzero(matrix.any(axis=1)).tolist())
    if len(fault_stacks) != 1:
        raise ValueError("unsupported_fault_count: exactly one infinite planar fault required")
    fault_stack = next(iter(fault_stacks))
    if relations[fault_stack] is not R.FAULT or not matrix[fault_stack].any() or matrix[fault_stack, fault_stack]:
        raise ValueError("unsupported_fault_metadata: unknown fault authorization")
    banks = stacks.faults_input_data
    if banks is not None:
        if len(banks) != stacks.n_stacks:
            raise ValueError("unsupported_fault_metadata: invalid fault data count")
        for data in banks:
            if data is not None:
                if not isinstance(data, FaultsData):
                    raise ValueError("unsupported_fault_metadata: unknown fault data")
                if data.finite_fault_defined:
                    raise ValueError("unsupported_finite_fault")
    data = interpolation_input._fault_values
    if data is not None:
        if not isinstance(data, FaultsData):
            raise ValueError("unsupported_fault_metadata: unknown input fault data")
        if data.finite_fault_defined:
            raise ValueError("unsupported_finite_fault")

    origins, spans, domain_shape, extent, _, metadata = _collect_joint_leaves(octree_list, stacks.n_stacks)
    groups = np.array([m[0] for m in metadata], dtype=np.int64)
    indices = np.array([m[1] for m in metadata], dtype=np.int64)
    levels = [_numpy(output.scalar_field_at_sp).astype(np.float64).reshape(-1).copy()
              for output in octree_list[0].outputs]
    separator_ids = np.flatnonzero(groups == fault_stack)
    if len(separator_ids) != 1 or indices[separator_ids[0]] != 0 or len(levels[fault_stack]) != 1:
        raise ValueError("unsupported_fault_surfaces: one separator boundary required")
    affected = set(np.flatnonzero(matrix[fault_stack]).tolist())
    unaffected = set(range(stacks.n_stacks)) - affected - {fault_stack}
    _, _, truncations = build_contact_relations(groups, indices, relations, matrix, levels)
    if any((groups[c] in affected and groups[t] in unaffected) or
           (groups[c] in unaffected and groups[t] in affected) for c, t in truncations):
        raise ValueError("unsupported_cross_partition_dependencies: affected/unaffected truncation")

    from .adaptive_fault_sampler import prepare_fault_bank_sampler
    from .adaptive_topology_extraction import extract_adaptive_topology

    closed = bool(getattr(options.evaluation_options, "mesh_extraction_closed_volumes", False))
    if closed and unaffected:
        raise ValueError("unsupported_closed_volumes: stacks unaffected by the fault are not split into banks")
    sampler = prepare_fault_bank_sampler(descriptor, interpolation_input, options, octree_list)
    if (sampler["fault_stack"] != fault_stack or set(sampler["affected_stacks"]) != affected or
            set(sampler["unaffected_stacks"]) != unaffected):
        raise ValueError("unsupported_fault_metadata: sampler partition disagrees with directed matrix")
    memo = _BankFieldMemo(sampler["query"], affected)
    # Leaf geometry is bank-independent; every partition shares one complex.
    cell_complex = build_adaptive_complex(origins, spans, domain_shape)
    spacing = (extent[1::2] - extent[::2]) / domain_shape
    xyz = extent[::2] + (origins[:, None, :] + spans[:, None, None] * CORNERS) * spacing
    pieces = []
    for bank in (0, 1, -1):
        selected = np.flatnonzero(np.isin(groups, list(unaffected if bank == -1 else affected | {fault_stack})))
        if not len(selected):
            continue
        sign = -1 if bank == 0 else 1

        def sample_fields(points, selected=selected, bank=bank, sign=sign):
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            needed = sorted(set(groups[selected].tolist()))
            raw, gradients = memo(points, needed, None if bank == -1 else bank)
            if raw.shape != (len(needed), len(points)) or gradients.shape != (len(needed), len(points), 3):
                raise ValueError("invalid_fault_sampler: expected raw stack scalar/gradient arrays")
            rows = np.searchsorted(needed, groups[selected])
            values = raw[rows]
            normals = gradients[rows]
            if bank != -1:
                local = np.flatnonzero(groups[selected] == fault_stack)
                values[local] = sign * (values[local] - levels[fault_stack][0])
                normals[local] *= sign
            return values, normals

        samples = sample_fields(xyz.reshape(-1, 3))[0].reshape(len(selected), len(origins), 8)
        if bank != -1:
            separator = int(np.flatnonzero(groups[selected] == fault_stack)[0])
            _, fault_pairs, ordinary = build_contact_relations(
                groups[selected], indices[selected], relations, matrix, levels)
            controllers = {i: [] for i in range(len(selected))}
            for c, t in sorted(ordinary):
                controllers[t].append((c, 1 if groups[selected[c]] > groups[selected[t]] else -1))
            for c, t in sorted(fault_pairs):
                if c != separator or groups[selected[t]] not in affected:
                    raise ValueError("unsupported_fault_metadata: unauthorized separator contact")
                controllers[t].append((separator, -1))
            separator_contacts = dict(separator=separator, bank=bank, fault_pairs=fault_pairs,
                                      controllers=controllers)
            result = extract_adaptive_topology(
                origins, spans, domain_shape, extent, samples, groups[selected], indices[selected],
                relations, levels, sample_fields=sample_fields, include_reference=False, defer_emission=True,
                faults_relations=matrix, separator_contacts=separator_contacts, cell_complex=cell_complex,
                closure=closed)
        else:
            # Removing an unrepresented fault label is safe only if it preserves
            # every ordinary dependency selected from the full original stacks.
            safe_relations = [R.BASEMENT if r is R.FAULT else r for r in relations]
            original = build_contact_relations(groups[selected], indices[selected], relations, matrix, levels)
            sanitized = build_contact_relations(groups[selected], indices[selected], safe_relations, None, levels)
            if original[1] or not np.array_equal(original[0], sanitized[0]) or original[2] != sanitized[2]:
                raise ValueError("unsupported_unaffected_dependencies: fault label cannot be safely omitted")
            result = extract_adaptive_topology(
                origins, spans, domain_shape, extent, samples, groups[selected], indices[selected],
                safe_relations, levels, sample_fields=sample_fields, include_reference=False, defer_emission=True,
                cell_complex=cell_complex)
        pieces.append((bank, selected, result))

    closed_parts = {}
    if closed:
        # Closed unit shells per bank, validated before any bank emits.
        plane = np.asarray(sampler["diagnostics"]["plane"], dtype=float)
        stack_levels = [np.asarray(level, dtype=float) for level in levels]
        all_stacks = list(range(stacks.n_stacks))
        for bank, selected, result in pieces:
            sign = -1 if bank == 0 else 1
            separator = int(np.flatnonzero(groups[selected] == fault_stack)[0])

            def sample_stacks(points, bank=bank):
                return memo(np.asarray(points, dtype=np.float64).reshape(-1, 3), all_stacks, bank)

            part = build_closed_volumes(
                result.pop("closure"), sample_stacks=sample_stacks, relations=relations,
                stack_isovalues=stack_levels, separator=(separator, sign), partition=bank,
                region_volume=_bank_region_volume(extent, plane, levels[fault_stack][0], sign))
            _check_closed_audit(part, bank)
            closed_parts[bank] = part
        volumes = _assemble_closed_volumes(closed_parts, fault_normal=plane[:3])
        _check_closed_audit(volumes.report, "shared fault")

    # A later bank or independent partition may reject otherwise valid geometry.
    # Allocate no triangles until every partition has passed all validations.
    for bank, _, result in pieces:
        _emit_with_closure(result, closed_parts.get(bank), emit_adaptive_triangles)

    # Numeric IDs and canonical keys share the same disjoint bank namespaces.
    seams, offset = [], 0
    components = [[] for _ in metadata]
    for bank, selected, result in pieces:
        vertices = np.asarray(result["vertices"], dtype=np.float64).reshape(-1, 3)
        seams.append(np.asarray(result["seam_edges"], dtype=np.int64).reshape(-1, 2) + offset)
        for local, exported in enumerate(selected):
            faces = np.asarray(result["faces"][local], dtype=np.int64).reshape(-1, 3)
            ids, inverse = np.unique(faces, return_inverse=True)
            components[exported].append((vertices[ids].copy(), inverse.reshape(-1, 3), ids + offset,
                                        [(bank, result["vertex_keys"][i]) for i in ids], bank))
        offset += len(vertices)
    seam_edges = np.concatenate(seams, axis=0)
    report = dict(interface="bank_side_fault_skin_not_single_conforming_fault_mesh",
                  banks={bank: result["diagnostics"] for bank, _, result in pieces},
                  coarsefine_seam_edge_count=sum(result["diagnostics"].get("coarsefine_seam_edge_count", 0)
                                                 for _, _, result in pieces),
                  sampler=sampler.get("diagnostics", {}), no_uniform_fallback=True,
                  no_post_reconciliation=True, cross_bank_id_joins=False)
    meshes = []
    for exported, (stack, surface, isovalue) in enumerate(metadata):
        rows = components[exported]
        vertices, faces, ids, keys, vertex_banks, face_banks = [], [], [], [], [], []
        count = 0
        for v, f, i, k, bank in rows:
            vertices.append(v)
            faces.append(f + count)
            ids.append(i)
            keys.extend(k)
            vertex_banks.append(np.full(len(v), bank, dtype=np.int64))
            face_banks.append(np.full(len(f), bank, dtype=np.int64))
            count += len(v)
        mesh = DualContouringMesh(
            vertices=np.concatenate(vertices), edges=np.concatenate(faces), stack_index=stack,
            surface_index=surface, exported_surface_index=exported, isovalue=isovalue,
            contact_report=report, joint_bank_ids=np.concatenate(vertex_banks),
            joint_face_bank_ids=np.concatenate(face_banks))
        mesh.joint_vertex_ids = np.concatenate(ids)
        mesh.joint_vertex_keys = keys
        mesh.joint_seam_edges = seam_edges.copy()
        meshes.append(mesh)
    if closed_parts:
        report["closed_volumes"] = dict(volumes.report, units=len(volumes.units))
        for mesh in meshes:
            mesh.closed_volumes = volumes
    return meshes


def _bank_region_volume(extent, plane, isovalue, sign):
    """Exact volume of box ∩ {sign * (fault scalar - isovalue) <= 0} (planar fault)."""
    from itertools import product
    from scipy.spatial import ConvexHull

    corners = np.array(list(product(*extent.reshape(3, 2))))
    values = sign*(corners @ plane[:3] + plane[3] - isovalue)
    points = [corners[values <= 0]]
    for a, b in [(i, j) for i in range(8) for j in range(i+1, 8)
                 if np.count_nonzero(corners[i] != corners[j]) == 1]:
        if values[a]*values[b] < 0:
            t = values[a]/(values[a]-values[b])
            points.append((corners[a]+t*(corners[b]-corners[a]))[None])
    points = np.concatenate(points)
    return float(ConvexHull(points).volume) if len(points) >= 4 else 0.


def extract_fault_merge_octree(descriptor, interpolation_input, options, octree_list):
    """``joint_contacts``: one ordinary extraction where affected surfaces borrow fault vertices.

    Fields are the production ones (smooth-fault drift), corners from the octree
    and new points from the fixed-weight production-drift sampler. Returns one
    mesh per original surface and one conforming mesh per fault; no banks.
    """
    from .adaptive_fault_sampler import prepare_fault_drift_sampler
    from .adaptive_topology_extraction import extract_adaptive_topology

    stacks = descriptor.stack_structure
    relations = stacks.masking_descriptor
    matrix = (np.zeros((stacks.n_stacks, stacks.n_stacks), dtype=bool)
              if stacks.faults_relations is None else _numpy(stacks.faults_relations))
    if matrix.shape != (stacks.n_stacks, stacks.n_stacks) or not np.isin(matrix, (False, True)).all():
        raise ValueError("unsupported_fault_metadata: invalid directed fault matrix")
    matrix = matrix.astype(bool)
    origins, spans, domain_shape, extent, samples, metadata = _collect_joint_leaves(octree_list, stacks.n_stacks)
    groups = np.array([m[0] for m in metadata], dtype=np.int64)
    indices = np.array([m[1] for m in metadata], dtype=np.int64)
    levels = [_numpy(output.scalar_field_at_sp).astype(np.float64).reshape(-1).copy()
              for output in octree_list[0].outputs]
    sampler = prepare_fault_drift_sampler(descriptor, interpolation_input, options, octree_list)
    def field_query(requests):
        # One stacked sampler batch: only the requested surfaces' stacks, each
        # request's unique points once.
        prepared = []
        for points, surfaces, kind in requests:
            unique, inverse = np.unique(np.asarray(points, dtype=np.float64).reshape(-1, 3), axis=0,
                                        return_inverse=True)
            stacks_needed = sorted(set(groups[surfaces].tolist()))
            prepared.append((unique, stacks_needed, kind, np.searchsorted(stacks_needed, groups[surfaces]),
                             inverse.reshape(-1)))
        answers = sampler["query_batch"]([(unique, stacks_needed, kind)
                                          for unique, stacks_needed, kind, _, _ in prepared])
        return [answer[rows][:, inverse] for answer, (_, _, _, rows, inverse) in zip(answers, prepared)]

    def sample_fields(points):
        raise ValueError("unexpected_full_field_query: joint_contacts uses targeted field queries")

    _, fault_pairs, _ = build_contact_relations(groups, indices, relations, matrix, levels)
    fault_merge = {int(i): set() for i, g in enumerate(groups) if relations[g] is R.FAULT}
    for c, t in fault_pairs:
        if c not in fault_merge:
            raise ValueError("unsupported_fault_metadata: fault relation from a non-fault stack")
        fault_merge[c].add(int(t))
    result = extract_adaptive_topology(
        origins, spans, domain_shape, extent, samples, groups, indices, relations, levels,
        sample_fields=sample_fields, field_query=field_query, faults_relations=matrix, fault_merge=fault_merge,
        cell_complex=build_adaptive_complex(origins, spans, domain_shape))
    report = dict(result["diagnostics"], interface="pretty_fault_vertex_merge_single_fault_mesh",
                  sampler=sampler.get("diagnostics", {}), no_uniform_fallback=True,
                  no_post_reconciliation=True)
    vertices = np.asarray(result["vertices"], dtype=np.float64).reshape(-1, 3)
    meshes = []
    for exported, (stack, surface, isovalue) in enumerate(metadata):
        faces = np.asarray(result["faces"][exported], dtype=np.int64).reshape(-1, 3)
        ids, inverse = np.unique(faces, return_inverse=True)
        mesh = DualContouringMesh(
            vertices=vertices[ids].copy(), edges=inverse.reshape(-1, 3), stack_index=stack,
            surface_index=surface, exported_surface_index=exported, isovalue=isovalue,
            contact_report=report)
        mesh.joint_vertex_ids = ids
        mesh.joint_vertex_keys = [result["vertex_keys"][i] for i in ids]
        mesh.joint_seam_edges = np.asarray(result["seam_edges"], dtype=np.int64).copy()
        meshes.append(mesh)
    return meshes
