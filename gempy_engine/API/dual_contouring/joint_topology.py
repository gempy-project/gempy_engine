"""Explicit leaf-native joint dual contouring; no production dispatch or repairs.

Orchestration only: lattice, ownership, edge decisions and triangle planning
are module functions; this file samples fields, solves the production QEFs and
passes data between them.
"""

import numpy as np

from ...config import AvailableBackends
from ...core.backend_tensor import BackendTensor as BT
from ...core.data.dual_contouring_data import DualContouringData
from ...core.data.stack_relation_type import StackRelationType as R
from ...modules.dual_contouring._gen_vertices import generate_dual_contouring_vertices
from ...modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge
from ...modules.dual_contouring.joint_cell_branches import CORNERS, EDGE_START, EDGE_END, classify_cell_branches
from ...modules.dual_contouring.joint_cell_complex import build_adaptive_complex
from ...modules.dual_contouring.joint_contact_relations import build_contact_relations
from ...modules.dual_contouring.joint_edge_decisions import crossing_normals, edge_crossing_decisions
from ...modules.dual_contouring.joint_field_queries import ExactPointCache, TargetedFieldBatch
from ...modules.dual_contouring.joint_lattice import (field_tolerances, lattice_frame, leaf_corner_nodes,
                                                      lookup_node_fields, minimal_edge_nodes, shared_node_fields,
                                                      tile_nodes)
from ...modules.dual_contouring.joint_separator import (authorize_fault_pairs, check_separator_contract,
                                                        check_separator_samples, fit_separator_plane,
                                                        separator_controllers)
from ...modules.dual_contouring.joint_ownership import (borrowed_hermite_rows, borrowed_leaves,
                                                        check_composite_controllers, check_missing_controllers,
                                                        contact_pairs, controllers_from_ownership,
                                                        fallback_junction_leaves, fallback_region,
                                                        fault_merge_targets, ordinary_controllers)
from ...modules.dual_contouring.joint_triangle_plan import (emit_adaptive_triangles, plan_adaptive_junctions,
                                                            plan_adaptive_triangles, plan_edge_rows,
                                                            validate_adaptive_geometry, validate_adaptive_incidence)
from ...modules.dual_contouring.weighted_qef_setup_multicore import DEFAULT_CROSS_SURFACE_WEIGHT


def extract_adaptive_topology(origins, spans, domain_shape, extent, scalar_samples,
                              surface_to_stack, surface_indices, stack_relations, isovalues,
                              *, sample_fields, ownership=None, include_reference=False,
                              faults_relations=None, separator_contacts=None, defer_emission=False,
                              cell_complex=None, fault_merge=None, field_query=None):
    """Extract balanced dyadic leaves using canonical tiles and minimal primal edges.

    ``sample_fields`` supplies raw fields (S,M) and actual gradients (S,M,3).
    Signed synthetic ownership must equal eligible controller fields (or their
    negatives); it is never interpolated independently at hanging edges. Original
    Hermite rows and ordinary production QEF regularization remain untouched.
    A composite controller dict must exactly match derived ordinary ownership.
    ``separator_contacts`` adds only explicitly authorized directed fault pairs;
    its separator is already bank-normalized (positive excluded, level zero),
    with correspondingly oriented actual gradients. Stack metadata stays original.
    ``defer_emission`` returns a fully validated ``_triangle_plan`` and
    ``_key_to_id`` with faces/affected_faces set to None for a caller-owned
    all-bank validation barrier. It cannot be combined with ``include_reference``.
    ``cell_complex`` may supply the read-only ``build_adaptive_complex`` result
    for these exact origins/spans/domain, so bank partitions share one build.
    ``fault_merge`` maps each represented fault surface to exactly the surfaces it
    affects (pretty's fault rule, as key substitution): in leaves where the fault
    has a vertex, an affected surface uses the fault's vertex identity, its
    discarded own geometry there is not validated, all-fault triangles are
    dropped, and a contact junction there or in the adjacent leaf ring is not
    shared: participants keep unshared vertices without ownership certification
    (pretty-style fallback, counted, never silent). A leaf crossed by an
    affected surface and two or more of its faults has no unambiguous borrow:
    the surface keeps its own vertex there, under the same fallback. Faults are
    not ownership controllers.
    ``field_query([(points, surfaces, 'scalar'|'gradient'), ...])`` optionally
    replaces ``sample_fields`` with batched targeted queries, returning per
    request (S, M) scalars or (S, M, 3) gradients: tile and minimal-edge node
    values come from the supplied leaf corners (shared nodes must agree within
    tolerance), gradients are evaluated only for the crossing surface (one batch
    for crossings and crossed-edge endpoints), and vertex checks are one
    scalar-only batch. Not with separator contacts.
    """
    if defer_emission and include_reference:
        raise ValueError('unsupported_deferred_reference: defer_emission and include_reference are incompatible')
    # Topology is host NumPy; Torch tensors for crossings/QEFs live on the active device.
    if BT.dtype != 'float64' or BT.engine_backend not in (AvailableBackends.numpy, AvailableBackends.PYTORCH):
        raise ValueError('unsupported_backend: NumPy/Torch float64 only')

    # region Contracts and controllers
    allowed, faults, truncations = build_contact_relations(
        surface_to_stack, surface_indices, stack_relations, faults_relations, isovalues)
    if separator_contacts is None and fault_merge is None and (faults or any(r is R.FAULT for r in stack_relations)
                                or (faults_relations is not None and np.asarray(faults_relations).any())):
        raise ValueError('unsupported_fault_extraction: adaptive bank incidence unavailable')
    merge_targets = {}
    if fault_merge is not None:
        if separator_contacts is not None or ownership is not None:
            raise ValueError('invalid_fault_merge_contract: exclusive with separator contacts and ownership')
        merge_targets = fault_merge_targets(fault_merge, surface_to_stack, stack_relations, faults)
    separator, bank = None, None
    if separator_contacts is not None:
        separator, bank = check_separator_contract(separator_contacts, surface_to_stack, stack_relations, faults)
        allowed = authorize_fault_pairs(allowed, faults)
    full_identities = [(g, s) for g, values in enumerate(isovalues) for s in range(len(values))]
    _, _, full_truncations = build_contact_relations(
        np.array([g for g, _ in full_identities], dtype=int),
        np.array([s for _, s in full_identities], dtype=int), stack_relations, faults_relations, isovalues)
    check_missing_controllers(full_identities, full_truncations, surface_to_stack, surface_indices, isovalues)
    # endregion

    # region Lattice and samples
    complex_ = build_adaptive_complex(origins, spans, domain_shape) if cell_complex is None else cell_complex
    origins, spans = np.asarray(origins), np.asarray(spans)
    domain = np.asarray(domain_shape)
    bounds, spacing = lattice_frame(extent, domain)
    corners = leaf_corner_nodes(origins, spans, CORNERS)
    xyz = bounds[::2] + corners*spacing
    identities = list(zip(map(int, surface_to_stack), map(int, surface_indices)))
    n, leaves = len(identities), len(origins)
    if not n or len(set(identities)) != n:
        raise ValueError('invalid_surface_ids: unique stack/local identities required')
    samples = _detached(scalar_samples).astype(float)
    if samples.shape != (n, leaves, 8) or not np.isfinite(samples).all():
        raise ValueError('invalid_samples: finite (S,N,8) raw corners required')
    levels = np.array([isovalues[g][s] for g, s in identities])
    if separator is not None:
        levels[separator] = 0.
    fields = samples-levels[:, None, None]
    faces, edges = complex_['faces'], complex_['edges']
    tiles, edge_nodes = tile_nodes(faces), minimal_edge_nodes(edges)
    tile_xyz, minimal_xyz = bounds[::2]+tiles*spacing, bounds[::2]+edge_nodes*spacing
    tolerances = field_tolerances(samples, levels)
    if field_query is not None and (separator_contacts is not None or ownership is not None):
        raise ValueError('invalid_field_query_contract: exclusive with separator contacts and ownership')
    plane = []

    def on_separator_plane(points, raw, gradients):
        if plane:
            check_separator_samples(points, raw[separator], gradients[separator], plane[0])

    query = ExactPointCache(sample_fields, n, on_separator_plane if separator is not None else None)
    batch = TargetedFieldBatch(field_query)
    if field_query is not None:
        # Every tile and minimal-edge node is a leaf corner: reuse the supplied
        # corner values, after checking that leaves sharing a node agree.
        lattice = shared_node_fields(corners, fields, domain, tolerances)
        tile_fields = lookup_node_fields(tiles, *lattice).reshape(n, len(faces), 4)
        edge_fields = lookup_node_fields(edge_nodes, *lattice).reshape(n, len(edges), 2)
        # Endpoint gradients orient ring triangles only on edges a surface
        # crosses; they are queried with the crossing normals in one batch.
        endpoint_gradients = np.zeros((n, len(edges), 2, 3))
    else:
        all_points = np.concatenate((xyz.reshape(-1, 3), tile_xyz.reshape(-1, 3), minimal_xyz.reshape(-1, 3)))
        raw, gradients = query(all_points)
        if separator is not None:
            plane.append(fit_separator_plane(all_points, raw[separator], gradients[separator]))
        canonical, _ = query(xyz)
        if np.any(np.abs(canonical.reshape(samples.shape)-levels[:, None, None]-fields) > tolerances[:, None, None]):
            raise ValueError('inconsistent_corner_samples: callback disagrees with raw original corners')
        tile_fields = query(tile_xyz)[0].reshape(n, len(faces), 4)-levels[:, None, None]
        edge_fields, endpoint_gradients = query(minimal_xyz)
        edge_fields = edge_fields.reshape(n, len(edges), 2)-levels[:, None, None]
        endpoint_gradients = endpoint_gradients.reshape(n, len(edges), 2, 3)
    controllers = ordinary_controllers(identities, truncations)
    if separator is not None:
        controllers = separator_controllers(controllers, faults)
        check_composite_controllers(separator_contacts['controllers'], controllers)
        if ownership is not None:
            raise ValueError('unsupported_ownership: separator contract supplies composite ownership')
    elif isinstance(ownership, dict):
        check_composite_controllers(ownership, controllers)
    elif ownership is not None:
        controllers = controllers_from_ownership(_detached(ownership), fields, truncations)
    pairs = contact_pairs(controllers, allowed)
    # endregion

    # region Crossings, branches and production QEFs
    stricts = (fields[:, :, EDGE_START]*fields[:, :, EDGE_END]) < 0
    merged, multi_fault, borrowed = borrowed_leaves(merge_targets, stricts)
    flags, counts, edge_xyz = _surface_crossings(xyz, samples, levels, stricts, merged, multi_fault, fields,
                                                 strict=separator is not None or fault_merge is not None,
                                                 reject_aligned=separator is not None)
    # A surface never solves its own QEF in borrowed leaves; there its Hermite
    # rows only constrain the fault vertex it borrows (pretty's weighted rule).
    solved = flags & ~merged[:, :, None]
    crossing_points = np.concatenate([edge_xyz[s][flags[s]] for s in range(n)])
    if len(crossing_points) and field_query is None:
        query(crossing_points)
    normals = np.zeros_like(edge_xyz)
    if field_query is None:
        for s in range(n):
            if flags[s].any():
                normals[s][flags[s]] = query(edge_xyz[s][flags[s]])[1][s]
    else:
        crossed_edges = [np.flatnonzero(edge_fields[s, :, 0]*edge_fields[s, :, 1] < 0) for s in range(n)]
        answers = batch([(edge_xyz[s][flags[s]], [s], 'gradient') for s in range(n)] +
                        [(minimal_xyz[crossed_edges[s]].reshape(-1, 3), [s], 'gradient') for s in range(n)])
        for s in range(n):
            normals[s][flags[s]] = answers[s][0]
            endpoint_gradients[s, crossed_edges[s]] = answers[n+s][0].reshape(-1, 2, 3)
    for s in range(n):
        if np.any(np.linalg.norm(normals[s][solved[s]], axis=1) < 1e-12):
            raise ValueError('degenerate_hermite_normal: actual crossing gradient vanishes')

    leaf_origins = list(map(tuple, origins.astype(np.int64).tolist()))
    leaf_spans = spans.astype(np.int64).tolist()

    def regular_key(s, cell):
        return 'regular', (identities[s],), leaf_origins[cell], leaf_spans[cell], 0

    regular, weighted_fault_cells = {}, set()
    for s in range(n):
        own = solved[s]
        if not own.any():
            continue
        cells = np.flatnonzero(own.any(axis=1))
        extra = None
        if fault_merge is not None and s in fault_merge:
            extra = borrowed_hermite_rows(s, cells, fault_merge[s], borrowed, edge_xyz, normals, flags,
                                          DEFAULT_CROSS_SURFACE_WEIGHT)
            if extra is not None:
                weighted_fault_cells.update((s, int(c)) for c in cells[np.any(extra[2] > 0, axis=1)])
        positions = _solve_qef(edge_xyz[s][own], own, xyz, spans, spacing, origins, normals[s][own], extra)
        for cell, position in zip(cells, positions):
            regular[regular_key(s, cell)] = position
    # endregion

    # region Junctions, fallback region and edge rows
    fault_region = bool(borrowed) or multi_fault.any()
    fallback, region = set(), set()
    if fault_region:
        # One region for every surface: a junction given up there leaves both
        # participants, affected or not, with unshared vertices.
        region = fallback_region(edges, (merged | multi_fault).any(axis=0))
        junctions, evidence, fallback = plan_adaptive_junctions(
            faces, tile_xyz, tile_fields, pairs, identities, origins, spans, counts, edge_xyz, normals, flags,
            controllers=controllers, leaf_fields=fields, overridden={s: region for s in range(n)})
        fallback = fallback_junction_leaves(fallback, region, flags.any(axis=2), pairs)
    else:
        junctions, evidence = plan_adaptive_junctions(
            faces, tile_xyz, tile_fields, pairs, identities, origins, spans, counts, edge_xyz, normals, flags,
            controllers=controllers, leaf_fields=fields)
    fallback_leaves = {(tuple(map(int, origins[c])), int(spans[c])) for c in region}

    def vertex_key(s, cell):
        if cell in junctions and s in junctions[cell]['pair']:
            return junctions[cell]['key']
        source = borrowed.get((s, cell))
        return regular_key(s if source is None else source, cell)

    edge_cells = np.array([[-1 if c is None else c for c in edge['cells']] for edge in edges],
                          dtype=np.int64).reshape(-1, 4)
    selected, fraction = edge_crossing_decisions(edge_fields, controllers, edge_cells, counts)
    fault_identities = {identities[c] for c in (fault_merge or {})}
    plans, filtered_triangles = plan_edge_rows(selected, edges, crossing_normals(fraction, endpoint_gradients),
                                               regular_key, vertex_key, borrowed, fault_identities)
    # endregion

    # region Validation and emission
    seams, coarsefine = validate_adaptive_incidence(plans, junctions, evidence, spans)
    positions = regular.copy()
    for cell, junction in junctions.items():
        for s in junction['pair']:
            positions.pop(regular_key(s, cell))
        positions[junction['key']] = junction['position']
    keys = sorted(positions)
    vertices = np.asarray([positions[k] for k in keys]).reshape(-1, 3)
    ids = {k: i for i, k in enumerate(keys)}
    triangles = plan_adaptive_triangles(plans, positions)
    used_regular = sorted({k for rows in triangles for row in rows for k in row['keys'] if k[0] == 'regular'})
    joint_keys = [j['key'] for j in junctions.values()]
    vertex_fields = {}
    if field_query is None:
        if used_regular:
            raw, _ = query([positions[k] for k in used_regular])
            vertex_fields = dict(zip(used_regular, (raw-levels[:, None]).T))
        if junctions:
            raw, _ = query([positions[k] for k in joint_keys])
            vertex_fields.update(zip(joint_keys, (raw-levels[:, None]).T))
    else:
        regular_raw, joint_raw = batch([([positions[k] for k in used_regular], range(n), 'scalar'),
                                        ([positions[k] for k in joint_keys], range(n), 'scalar')])
        vertex_fields = dict(zip(used_regular, (regular_raw-levels[:, None]).T))
        vertex_fields.update(zip(joint_keys, (joint_raw-levels[:, None]).T))
    # Ownership is not certified for borrowed fault vertices nor in fallback
    # leaves, where pretty-style unshared vertices replace a joint junction.
    exempt = ([{k for row in rows for k in row['keys'] if k[0] == 'regular' and (
                   k[1][0] in fault_identities or (k[2], k[3]) in fallback_leaves)}
               for rows in triangles] if fault_region else None)
    validate_adaptive_geometry(triangles, seams, junctions, positions, vertex_fields, tolerances, controllers,
                               exempt=exempt)
    output_faces, affected = (None, None) if defer_emission else emit_adaptive_triangles(triangles, ids)
    # endregion

    result = dict(vertices=vertices, faces=output_faces, vertex_keys=keys,
                  seam_edges=np.asarray([[ids[a], ids[b]] for a, b in sorted(seams)], dtype=int).reshape(-1, 2),
                  junction_cells=np.asarray(sorted(junctions), dtype=int), leaf_origins=origins.copy(),
                  leaf_spans=spans.copy(), affected_faces=affected,
                  diagnostics=dict(algorithm='adaptive_joint_hermite_qef_dual_contouring',
                                   coarsefine_seam_edge_count=len(coarsefine), no_uniform_fallback=True,
                                   no_post_reconciliation=True, original_hermite_rows=12,
                                   sampled_point_count=len(query)+batch.sampled, canonical_face_tile_count=len(faces),
                                   minimal_edge_count=len(edges), finite_faults='unsupported',
                                   interior_topology='sampled_boundary_only_not_interior_certified'))
    if defer_emission:
        result.update(_triangle_plan=triangles, _key_to_id=ids)
    if include_reference:
        reference_keys = sorted(regular)
        reference_plans = [[dict(row, keys=row['regular_keys']) for row in rows] for rows in plans]
        reference_faces, _ = emit_adaptive_triangles(
            plan_adaptive_triangles(reference_plans, regular), {k: i for i, k in enumerate(reference_keys)})
        result['reference'] = dict(vertices=np.asarray([regular[k] for k in reference_keys]).reshape(-1, 3),
                                   faces=reference_faces, vertex_keys=reference_keys,
                                   description='Original production QEFs with identical retained minimal edges')
    if fault_merge is not None:
        result['diagnostics'].update(
            contact_type='pretty_fault_vertex_merge', strict_crossings=True,
            crossing_contract='strict_scalar_sides', fault_surfaces=sorted(int(c) for c in fault_merge),
            fault_cell_count={int(c): int(flags[c].any(axis=1).sum()) for c in fault_merge},
            borrowed_vertex_count=len(borrowed), filtered_fault_triangle_count=filtered_triangles,
            multi_fault_unborrowed_count=int(multi_fault.sum()),
            weighted_fault_vertex_count=len(weighted_fault_cells),
            cross_surface_weight=DEFAULT_CROSS_SURFACE_WEIGHT,
            fault_overridden_junction_cells=sorted(int(c) for c in fallback),
            fallback_region_leaf_count=len(fallback_leaves))
    if separator is not None:
        result['bank'] = bank
        result['diagnostics'].update(bank=bank, separator_contact=separator,
                                     contact_type='explicit_fault_separator', separator_geometry='planar',
                                     strict_crossings=True, crossing_contract='strict_scalar_sides')
    return result


def _detached(a):
    return a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a)


def _tensor(a, boolean=False):
    if BT.engine_backend is AvailableBackends.PYTORCH:
        import torch
        return torch.as_tensor(a, dtype=torch.bool if boolean else torch.float64,
                               device=BT.device if BT.use_gpu else 'cpu')
    return np.asarray(a, dtype=bool if boolean else np.float64)


def _surface_crossings(xyz, samples, levels, stricts, merged, multi_fault, fields, *, strict, reject_aligned=False):
    """Production edge crossings per surface, checked against strict sides, plus branch counts.

    Leaves whose geometry a surface discards (borrowed or multi-fault) are
    neither checked nor classified (count one). ``reject_aligned`` rejects any
    original corner on a surface (bank geometry has no sided rule for it).
    """
    n, leaves = stricts.shape[:2]
    flags, edge_xyz, counts = [], [], []
    for s in range(n):
        if reject_aligned and np.any(np.abs(fields[s]) < 1e-10):
            raise ValueError('sample_aligned_interface: original corner lies on surface')
        crossings, production_flags = find_intersection_on_edge(
            _tensor(xyz.reshape(-1, 3)), _tensor(samples[s].reshape(-1)), _tensor(levels[s:s+1]),
            **({'strict_crossings': True} if strict else {}))
        production_flags = _detached(production_flags).reshape(leaves, 12).astype(bool)
        skipped = merged[s] | multi_fault[s]
        if not np.array_equal(stricts[s][~skipped], production_flags[~skipped]):
            raise ValueError('unsupported_geometric_flags: production tolerance crosses a non-strict edge')
        if merged[s].any():
            # Production crossings follow production flags; keep only strict edges.
            dense_production = np.zeros((leaves, 12, 3))
            dense_production[production_flags] = _detached(crossings)
            crossings = dense_production[stricts[s]]
        _, components, _, _, _ = classify_cell_branches(fields[s], stricts[s], skip=skipped if skipped.any() else None)
        if np.any(components > 1):
            raise ValueError('unsupported_adaptive_branches: only original single-branch leaves supported')
        components[skipped] = 1
        dense = np.zeros((leaves, 12, 3))
        dense[stricts[s]] = _detached(crossings)
        flags.append(stricts[s])
        counts.append(components)
        edge_xyz.append(dense)
    return np.asarray(flags), np.asarray(counts), np.asarray(edge_xyz)


def _solve_qef(crossing_xyz, own, xyz, spans, spacing, origins, crossing_normals, extra):
    """Production QEF vertices of one surface's crossed leaves, with optional extra weighted rows."""
    dc = DualContouringData(
        _tensor(crossing_xyz), _tensor(own, True), _tensor(xyz.mean(axis=1)),
        _tensor(spans[:, None]*spacing), 1, _tensor(origins), gradients=_tensor(crossing_normals),
        strict_crossings=False)
    if extra is not None:
        dc.extra_edge_xyz, dc.extra_edge_normals, dc.extra_weights = map(_tensor, extra)
    positions = _detached(generate_dual_contouring_vertices(dc))
    if not np.isfinite(positions).all():
        raise ValueError('nonfinite_qef_solution: original production solve failed')
    return positions
