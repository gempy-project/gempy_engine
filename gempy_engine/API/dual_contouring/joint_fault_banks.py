"""Production fixed-weight fault banks (``joint`` with one planar fault), assembled without cross-bank welding."""

import numpy as np

from .fault_bank_sampler import prepare_fault_bank_sampler
from .joint_extraction import _collect_joint_leaves, _numpy
from .joint_topology import extract_adaptive_topology
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.kernel_classes.faults import FaultsData
from ...core.data.stack_relation_type import StackRelationType as R
from ...modules.dual_contouring.joint_cell_branches import CORNERS
from ...modules.dual_contouring.joint_cell_complex import build_adaptive_complex
from ...modules.dual_contouring.joint_contact_relations import build_contact_relations
from ...modules.dual_contouring.joint_ownership import ordinary_controllers
from ...modules.dual_contouring.joint_separator import bank_normalized, separator_controllers
from ...modules.dual_contouring.joint_triangle_plan import emit_adaptive_triangles


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
            if bank == -1:
                return raw[rows], gradients[rows]
            return bank_normalized(raw[rows], gradients[rows], np.flatnonzero(groups[selected] == fault_stack),
                                   sign, levels[fault_stack][0])

        samples = sample_fields(xyz.reshape(-1, 3))[0].reshape(len(selected), len(origins), 8)
        if bank != -1:
            separator = int(np.flatnonzero(groups[selected] == fault_stack)[0])
            _, fault_pairs, ordinary = build_contact_relations(
                groups[selected], indices[selected], relations, matrix, levels)
            if any(c != separator or groups[selected[t]] not in affected for c, t in fault_pairs):
                raise ValueError("unsupported_fault_metadata: unauthorized separator contact")
            identities = list(zip(groups[selected].tolist(), indices[selected].tolist()))
            controllers = separator_controllers(ordinary_controllers(identities, ordinary), fault_pairs)
            separator_contacts = dict(separator=separator, bank=bank, fault_pairs=fault_pairs,
                                      controllers=controllers)
            result = extract_adaptive_topology(
                origins, spans, domain_shape, extent, samples, groups[selected], indices[selected],
                relations, levels, sample_fields=sample_fields, include_reference=False, defer_emission=True,
                faults_relations=matrix, separator_contacts=separator_contacts, cell_complex=cell_complex)
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

    # A later bank or independent partition may reject otherwise valid geometry.
    # Allocate no triangles until every partition has passed all validations.
    for _, _, result in pieces:
        plan, key_to_id = result.pop("_triangle_plan"), result.pop("_key_to_id")
        result["faces"], result["affected_faces"] = emit_adaptive_triangles(plan, key_to_id)

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
    return meshes
