"""Explicit uniform-grid joint DUAL contouring experiment; no production dispatch."""

from itertools import product

import numpy as np

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.dual_contouring_data import DualContouringData
from gempy_engine.core.data.options.evaluation_options import TriangulationMethod
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.modules.dual_contouring._gen_vertices import generate_dual_contouring_vertices
from gempy_engine.modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge
from gempy_engine.modules.dual_contouring.contact_topology import build_contact_relations
from gempy_engine.modules.dual_contouring.topology_extraction import (
    CORNERS, EDGE_START, EDGE_END, classify_cell_branches, plan_joint_representatives,
    plan_dual_quads, validate_joint_incidence, emit_dual_triangles,
)


def extract_topology_aware(axes, scalar_samples, surface_to_stack, surface_indices,
                           stack_relations, isovalues, *, ownership, gradient_samples,
                           faults_relations=None, fault_support=None, fault_slip=None,
                           interior_samples=None, include_reference=False, unsafe_diagnostics=False):
    """Generate branch-aware Hermite/QEF vertices and shared dual seam incidence.

    Node samples/ownership: (S,nx,ny,nz); gradients add a last dimension of 3.
    Corner approximants establish sampled face topology; actual supplied Hermite
    normals drive QEF geometry, including curved fields. Junctions currently need
    cell-affine corner approximants, not exactly affine original geological fields.
    unsafe_diagnostics bypasses selected rejection guards to expose raw defects;
    it does not implement missing topology or make the output valid.
    """
    if BackendTensor.engine_backend is not AvailableBackends.numpy or BackendTensor.use_gpu:
        raise ValueError('unsupported_backend: explicit NumPy CPU experiment')
    allowed, faults, truncations = build_contact_relations(
        surface_to_stack, surface_indices, stack_relations, faults_relations, isovalues,
    )
    if faults or (faults_relations is not None and np.asarray(faults_relations).any()) or \
            any(r is StackRelationType.FAULT for r in stack_relations) or \
            fault_support is not None or fault_slip is not None:
        raise ValueError('unsupported_fault_extraction: no dual footprint/bank incidence adapter')
    axes = tuple(np.asarray(a, dtype=float) for a in axes)
    if len(axes) != 3 or any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all()
                             or np.any(np.diff(a) <= 0) for a in axes):
        raise ValueError('unsupported_grid: need three increasing finite axes')
    if any(not np.allclose(np.diff(a), np.diff(a)[0], rtol=1e-10, atol=1e-12) for a in axes):
        raise ValueError('unsupported_grid: uniform conforming cells only')
    shape = tuple(map(len, axes))
    domain = tuple(s - 1 for s in shape)
    if np.prod(domain) > 4096:
        raise ValueError('unsupported_bound: maximum 4096 cells')
    identities = list(zip(map(int, surface_to_stack), map(int, surface_indices)))
    n = len(identities)
    if not n or len(set(identities)) != n:
        raise ValueError('invalid_surface_ids: unique stack/local surface identities required')
    samples = np.asarray(scalar_samples, dtype=float)
    if np.asarray(ownership).dtype.kind == 'b':
        raise ValueError('unsupported_ownership: bool masks are not signed geological constraints')
    ownership = np.asarray(ownership, dtype=float)
    gradients = np.asarray(gradient_samples, dtype=float)
    if samples.shape != (n, *shape) or ownership.shape != samples.shape or gradients.shape != (*samples.shape, 3):
        raise ValueError('invalid_samples: aligned scalar, signed ownership and gradient nodes required')
    if not all(np.isfinite(a).all() for a in (samples, ownership, gradients)):
        raise ValueError('invalid_samples: all values must be finite')
    levels = np.array([isovalues[g][s] for g, s in identities])
    fields = samples - levels[:, None, None, None]
    coordinates = np.array(list(product(*(range(d) for d in domain))), dtype=np.int64)
    nodes = coordinates[:, None, :] + CORNERS[None, :, :]
    node_index = tuple(nodes[:, :, a] for a in range(3))
    xyz_nodes = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    corners_xyz = xyz_nodes[node_index]
    corner_fields = fields[(slice(None), *node_index)]
    corner_ownership = ownership[(slice(None), *node_index)]
    corner_gradients = gradients[(slice(None), *node_index)]
    valid_edges, edge_xyz, edge_normals, edge_ownership = [], [], [], []
    branch_labels, component_counts, coefficients, affine = [], [], [], []
    hermite_data = []
    ambiguous_faces = 0
    unsafe_warnings = [] if unsafe_diagnostics else None
    spacing = np.array([a[1] - a[0] for a in axes])
    for surface in range(n):
        crossings, flags = find_intersection_on_edge(
            corners_xyz.reshape(-1, 3), corner_fields[surface].reshape(-1), np.array([0.]),
            strict_crossings=True,
        )
        flags = np.asarray(flags, dtype=bool)
        dense_xyz = np.zeros((*flags.shape, 3))
        dense_xyz[flags] = crossings
        a, b = corner_fields[surface, :, EDGE_START].T, corner_fields[surface, :, EDGE_END].T
        fractions = np.divide(a, a - b, out=np.zeros_like(a), where=flags)
        norms = ((1 - fractions[..., None]) * corner_gradients[surface][:, EDGE_START] +
                 fractions[..., None] * corner_gradients[surface][:, EDGE_END])
        owned = ((1 - fractions) * corner_ownership[surface][:, EDGE_START] +
                 fractions * corner_ownership[surface][:, EDGE_END])
        if np.any(np.linalg.norm(norms[flags], axis=1) < 1e-12):
            raise ValueError('degenerate_hermite_normal: nonzero crossing gradients required')
        labels, counts, coeff, is_affine, ambiguous = classify_cell_branches(
            corner_fields[surface], flags, unsafe_warnings)
        ambiguous_faces += ambiguous
        valid_edges.append(flags)
        edge_xyz.append(dense_xyz)
        edge_normals.append(norms)
        edge_ownership.append(owned)
        branch_labels.append(labels)
        component_counts.append(counts)
        coefficients.append(coeff)
        affine.append(is_affine)
        hermite_data.append(DualContouringData(
            crossings, flags, corners_xyz.mean(axis=1), spacing, 1, coordinates,
            gradients=norms[flags], base_number=domain, strict_crossings=True,
            triangulation_method=TriangulationMethod.QUADS, generated_cell_coordinates=coordinates,
        ))
    valid_edges, edge_xyz, edge_normals, edge_ownership = map(np.array, (valid_edges, edge_xyz, edge_normals, edge_ownership))
    if interior_samples is not None:
        interior_samples = np.asarray(interior_samples, dtype=float)
        expected = corner_fields.mean(axis=2).reshape(n, *domain)
        if interior_samples.shape != expected.shape or not np.isfinite(interior_samples).all():
            raise ValueError('invalid_interior_samples: finite aligned cell-center samples required')
        centers = interior_samples.reshape(n, -1)
        hidden_crossing = ((corner_fields.min(axis=2) > 0) & (centers <= 0)) | \
                          ((corner_fields.max(axis=2) < 0) & (centers >= 0))
        if hidden_crossing.any():
            raise ValueError('insufficient_interior_topology: center reveals topology absent from corner crossings')
    # A mixed ownership mask needs a unique geological controller, not coincidence.
    selected_pairs = set()
    for target in range(n):
        own = corner_ownership[target]
        if np.any(own > 0) and not np.all(own > 0):
            matches = [(c, t) for c, t in truncations if t == target and allowed[c, t]
                       and (np.allclose(own, corner_fields[c], atol=1e-10, rtol=0)
                            or np.allclose(own, -corner_fields[c], atol=1e-10, rtol=0))]
            if len(matches) != 1:
                raise ValueError('unsupported_ownership: require one eligible signed controller')
            selected_pairs.update(matches)
    junctions, face_evidence = plan_joint_representatives(
        coordinates, corners_xyz, corner_fields, corner_ownership, branch_labels,
        coefficients, affine, selected_pairs, identities, edge_xyz, edge_normals, valid_edges,
        unsafe_warnings,
    )
    plans = plan_dual_quads(coordinates, domain, valid_edges, branch_labels, edge_ownership, identities, junctions)
    if unsafe_diagnostics:
        # Observe the raw plan, but do not let validation prevent diagnostic emission.
        try:
            seam_keys = validate_joint_incidence(plans, junctions, face_evidence, coordinates, domain)
        except ValueError as error:
            unsafe_warnings.append(str(error))
            seam_keys = set()
    else:
        seam_keys = validate_joint_incidence(plans, junctions, face_evidence, coordinates, domain)

    # Ordinary cells literally use the production Hermite + mass-point QEF.
    regular_positions = {}
    for surface in sorted(range(n), key=lambda s: identities[s]):
        rows, row_cells, row_branches = [], [], []
        for cell, count in enumerate(component_counts[surface]):
            for branch in range(count):
                rows.append(branch_labels[surface][cell] == branch)
                row_cells.append(cell)
                row_branches.append(branch)
        if not rows:
            continue
        rows, row_cells = np.array(rows), np.array(row_cells)
        data = DualContouringData(
            edge_xyz[surface, row_cells][rows], rows, corners_xyz[row_cells].mean(axis=1), spacing,
            1, coordinates[row_cells], gradients=edge_normals[surface, row_cells][rows], strict_crossings=True,
        )
        try:
            qef_vertices = generate_dual_contouring_vertices(data)
        except np.linalg.LinAlgError as error:
            raise ValueError('degenerate_qef_system: production solve needs better conditioned Hermite data') from error
        if not np.isfinite(qef_vertices).all():
            raise ValueError('nonfinite_qef_solution: rescale or refine supplied Hermite data')
        for cell, branch, position in zip(row_cells, row_branches, qef_vertices):
            key = ('regular', (identities[surface],), tuple(map(int, coordinates[cell])), int(branch))
            regular_positions[key] = position
    positions = regular_positions.copy()
    for junction in junctions.values():
        positions[junction['key']] = junction['position']
    used_keys = sorted({k for surface_plans in plans for p in surface_plans for k in p['keys']})
    # Retain unused cell representatives as evidence (e.g. an open one-cell crop).
    used_keys = sorted(set(used_keys) | {k for k in regular_positions if not any(
        j['key'][2] == k[2] and k[1][0] in j['key'][1] for j in junctions.values())} |
        {j['key'] for j in junctions.values()})
    vertices = np.array([positions[k] for k in used_keys]).reshape(-1, 3)
    key_to_id = {k: i for i, k in enumerate(used_keys)}
    faces, primal_edges, affected = emit_dual_triangles(plans, vertices, key_to_id, edge_normals, unsafe_warnings)
    result = dict(vertices=vertices, faces=faces, vertex_keys=used_keys,
                  cell_coordinates=coordinates, cell_components=np.array(component_counts),
                  branch_labels=np.array(branch_labels), primal_edges=primal_edges,
                  affected_faces=affected, junction_cells=np.array(sorted(junctions), dtype=int),
                  seam_edges=np.array([[key_to_id[a], key_to_id[b]] for a, b in sorted(seam_keys)], dtype=int).reshape(-1, 2),
                  hermite_data=hermite_data,
                  diagnostics={'algorithm': 'hermite_qef_dual_contouring', 'triangulation': 'production_quad_1_3',
                               'field_contract': 'sampled_affine_monotone_or_bilinear_extrusion', 'ambiguous_face_count': ambiguous_faces,
                               'hermite_normals': 'supplied_interpolated_not_forced_to_corner_fit',
                               'junction_constraints': 'cell_affine_corner_approximant_seam_actual_hermite_qef',
                               'interior_topology': 'center_checked_not_topology_certified' if interior_samples is not None else 'sampled_boundary_only_not_interior_certified',
                               'center_approximant_max_residual': None if interior_samples is None else float(np.max(np.abs(interior_samples - expected))),
                               'junction_cell_count': len(junctions), 'post_reconciliation': False,
                                'boundary': 'open_complete_quad_crop', 'finite_faults': 'unsupported_no_bank_incidence_adapter',
                                'unsafe_diagnostics': unsafe_diagnostics,
                                'unsafe_warnings': [] if unsafe_warnings is None else unsafe_warnings})
    if include_reference:
        reference_keys = sorted(regular_positions)
        reference_vertices = np.array([regular_positions[k] for k in reference_keys]).reshape(-1, 3)
        reference_plans = [[dict(p, keys=p['regular_keys']) for p in ps] for ps in plans]
        reference_faces, _, _ = emit_dual_triangles(
            reference_plans, reference_vertices, {k: i for i, k in enumerate(reference_keys)}, edge_normals,
            unsafe_warnings,
        )
        result['reference'] = dict(vertices=reference_vertices, faces=reference_faces,
                                   vertex_keys=reference_keys, affected_faces=affected,
                                   description='Independent production QEFs; identical retained primal-edge support and quad split')
    return result
