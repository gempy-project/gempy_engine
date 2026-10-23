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
from ...modules.dual_contouring.joint_cell_branches import CORNERS, EDGE_START, EDGE_END
from ...modules.dual_contouring.joint_cell_complex import AdaptiveComplex, _paused_gc, adaptive_complex_arrays
from ...modules.dual_contouring.joint_contact_relations import build_contact_relations
from ...modules.dual_contouring.joint_edge_decisions import crossing_normals, edge_crossing_decisions
from ...modules.dual_contouring.joint_field_queries import ExactPointCache, TargetedFieldBatch
from ...modules.dual_contouring.joint_lattice import (lattice_frame, leaf_corner_nodes,
                                                      lookup_node_fields, minimal_edge_nodes, precision_tolerance,
                                                      shared_node_fields, snap_straddling_nodes, tile_nodes)
from ...modules.dual_contouring.joint_separator import (authorize_fault_pairs, check_separator_contract,
                                                        check_separator_samples, fit_separator_plane,
                                                        separator_controllers)
from ...modules.dual_contouring.joint_ownership import (borrowed_hermite_rows, borrowed_leaves,
                                                        check_composite_controllers, check_missing_controllers,
                                                        contact_pairs, controllers_from_ownership,
                                                        fallback_junction_leaves, fallback_region,
                                                        fault_merge_targets, ordinary_controllers)
from ...modules.dual_contouring.joint_triangle_plan import (emit_adaptive_triangles, plan_adaptive_junctions,
                                                            plan_adaptive_triangles, plan_edge_rows, ring_triangles,
                                                            validate_adaptive_incidence)
from ...modules.dual_contouring.joint_xp import host, namespace
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
    ``cell_complex`` may supply the read-only ``adaptive_complex_arrays`` result
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
    if BT.engine_backend not in (AvailableBackends.numpy, AvailableBackends.PYTORCH):
        raise ValueError('unsupported_backend: NumPy/Torch only')

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
    # Leaves set the array library and device (NumPy, or Torch on the engine's device).
    xp = namespace(origins, spans)
    origins, spans = xp.asarray(origins, dtype=np.int64), xp.asarray(spans, dtype=np.int64)
    complex_ = (adaptive_complex_arrays(origins, spans, domain_shape) if cell_complex is None else
                AdaptiveComplex(*(xp.asarray(a) for a in cell_complex)))
    domain = np.asarray(host(domain_shape))
    bounds, spacing = lattice_frame(extent, domain)
    low, step = xp.asarray(bounds[::2]), xp.asarray(spacing)
    corners = leaf_corner_nodes(origins, spans, CORNERS)
    xyz = low + corners*step
    identities = list(zip(map(int, surface_to_stack), map(int, surface_indices)))
    n, leaves = len(identities), len(origins)
    if not n or len(set(identities)) != n:
        raise ValueError('invalid_surface_ids: unique stack/local identities required')
    samples = xp.asarray(scalar_samples, dtype=np.float64)
    if tuple(samples.shape) != (n, leaves, 8) or not bool(xp.isfinite(samples).all()):
        raise ValueError('invalid_samples: finite (S,N,8) raw corners required')
    host_levels = np.array([isovalues[g][s] for g, s in identities], dtype=np.float64)
    if separator is not None:
        host_levels[separator] = 0.
    levels = xp.asarray(host_levels)
    fields = samples-levels[:, None, None]
    n_faces, n_edges = len(complex_.face_axis), len(complex_.edge_axis)
    tiles = tile_nodes(complex_.face_axis, complex_.face_origin, complex_.face_span)
    edge_nodes = minimal_edge_nodes(complex_.edge_axis, complex_.edge_origin, complex_.edge_span)
    tile_xyz, minimal_xyz = low+tiles*step, low+edge_nodes*step
    if field_query is not None and (separator_contacts is not None or ownership is not None):
        raise ValueError('invalid_field_query_contract: exclusive with separator contacts and ownership')
    plane = []

    def on_separator_plane(points, raw, gradients):
        if plane:
            check_separator_samples(points, raw[separator], gradients[separator], plane[0],
                                    precision_tolerance(1e-10, BT.dtype))

    query = ExactPointCache(sample_fields, n, on_separator_plane if separator is not None else None)

    def sample(points):
        raw, gradients = query(host(points).reshape(-1, 3))
        return xp.asarray(raw), xp.asarray(gradients)

    batch = TargetedFieldBatch(field_query)
    all_endpoint_gradients = None
    if field_query is not None or separator is None:
        # Every tile and minimal-edge node is a leaf corner: reuse the supplied
        # corner values, one side per node for every leaf sharing it. A
        # re-queried node (low precision) could sit on the other side of a
        # surface and drop the dual face.
        samples = snap_straddling_nodes(corners, samples, host_levels, domain)
        fields = samples-levels[:, None, None]
        lattice = shared_node_fields(corners, fields, domain)
        tile_fields = lookup_node_fields(tiles, *lattice).reshape(n, n_faces, 4)
        edge_fields = lookup_node_fields(edge_nodes, *lattice).reshape(n, n_edges, 2)
    else:
        all_points = xp.concatenate((xyz.reshape(-1, 3), tile_xyz.reshape(-1, 3), minimal_xyz.reshape(-1, 3)))
        raw, gradients = sample(all_points)
        plane.append(fit_separator_plane(host(all_points), host(raw[separator]), host(gradients[separator]),
                                         precision_tolerance(1e-10, BT.dtype)))
        tile_fields = sample(tile_xyz)[0].reshape(n, n_faces, 4)-levels[:, None, None]
        edge_fields, all_endpoint_gradients = sample(minimal_xyz)
        edge_fields = edge_fields.reshape(n, n_edges, 2)-levels[:, None, None]
        all_endpoint_gradients = all_endpoint_gradients.reshape(n, n_edges, 2, 3)
    # Endpoint gradients orient ring triangles: only edges some surface crosses need them.
    crossed_edges = xp.flatnonzero((edge_fields[:, :, 0]*edge_fields[:, :, 1] < 0).any(axis=0))
    edge_slot = xp.full(n_edges, -1, dtype=np.int64)
    edge_slot[crossed_edges] = xp.arange(len(crossed_edges))
    if all_endpoint_gradients is not None:
        endpoint_gradients = all_endpoint_gradients[:, crossed_edges]
    elif field_query is None and len(crossed_edges):
        endpoint_gradients = sample(minimal_xyz[crossed_edges])[1].reshape(n, len(crossed_edges), 2, 3)
    else:
        endpoint_gradients = xp.zeros((n, len(crossed_edges), 2, 3))
    controllers = ordinary_controllers(identities, truncations)
    if separator is not None:
        controllers = separator_controllers(controllers, faults)
        check_composite_controllers(separator_contacts['controllers'], controllers)
        if ownership is not None:
            raise ValueError('unsupported_ownership: separator contract supplies composite ownership')
    elif isinstance(ownership, dict):
        check_composite_controllers(ownership, controllers)
    elif ownership is not None:
        controllers = controllers_from_ownership(host(ownership), host(fields), truncations)
    pairs = contact_pairs(controllers, allowed)
    # endregion

    # region Crossings, branches and production QEFs
    flags = (fields[:, :, xp.asarray(EDGE_START)]*fields[:, :, xp.asarray(EDGE_END)]) < 0
    merged, multi_fault, source = borrowed_leaves(merge_targets, flags)
    borrowed = source >= 0
    counts, edge_xyz = _surface_crossings(xp, xyz, samples, levels, flags, merged, multi_fault, fields,
                                          strict=separator is not None or fault_merge is not None,
                                          reject_aligned=separator is not None)
    normals = edge_xyz.zeros_like()
    # A surface never solves its own QEF in borrowed leaves; there its Hermite
    # rows only constrain the fault vertex it borrows (pretty's weighted rule).
    solved = flags & ~merged[:, :, None]
    crossed = [flags[s][edge_xyz.rows[s]] for s in range(n)]
    if field_query is None:
        crossing_points = xp.concatenate([edge_xyz.values[s][crossed[s]] for s in range(n)])
        if len(crossing_points):
            sample(crossing_points)
        for s in range(n):
            if bool(crossed[s].any()):
                normals.values[s][crossed[s]] = sample(edge_xyz.values[s][crossed[s]])[1][s]
    else:
        surface_edges = [xp.flatnonzero(edge_fields[s, :, 0]*edge_fields[s, :, 1] < 0) for s in range(n)]
        answers = batch([(host(edge_xyz.values[s][crossed[s]]), [s], 'gradient') for s in range(n)] +
                        [(host(minimal_xyz[surface_edges[s]]).reshape(-1, 3), [s], 'gradient') for s in range(n)])
        for s in range(n):
            normals.values[s][crossed[s]] = xp.asarray(answers[s][0])
            endpoint_gradients[s, edge_slot[surface_edges[s]]] = xp.asarray(answers[n+s][0]).reshape(-1, 2, 3)
    for s in range(n):
        own = solved[s][edge_xyz.rows[s]]
        if bool((xp.norm(normals.values[s][own], axis=1) < 1e-12).any()):
            raise ValueError('degenerate_hermite_normal: actual crossing gradient vanishes')

    # Vertex ids: s*leaves + leaf is surface s's own vertex, joint_start + j the j-th junction.
    joint_start = n*leaves
    regular = xp.full((n, leaves, 3), np.nan)
    weighted_fault_cells = set()
    for s in range(n):
        own = solved[s][edge_xyz.rows[s]]
        solving = own.any(axis=1)
        if not bool(solving.any()):
            continue
        cells = edge_xyz.rows[s][solving]
        extra = None
        if fault_merge is not None and s in fault_merge:
            extra = borrowed_hermite_rows(s, cells, fault_merge[s], source, edge_xyz, normals, flags,
                                          DEFAULT_CROSS_SURFACE_WEIGHT)
            if extra is not None:
                weighted_fault_cells.update((s, c) for c in cells[(extra[2] > 0).any(axis=1)].tolist())
        regular[s, cells] = _solve_qef(xp, edge_xyz.values[s][own], own[solving], xyz[cells], spans[cells], step,
                                       origins[cells], normals.values[s][own], extra)
    # endregion

    # region Junctions, fallback region and edge rows
    region = None
    if bool(borrowed.any()) or bool(multi_fault.any()):
        # One region for every surface: a junction given up there leaves both
        # participants, affected or not, with unshared vertices.
        region = fallback_region(complex_.edge_cells, (merged | multi_fault).any(axis=0))
    junctions, evidence, fallback = _plan_junctions(
        xp, complex_.face_cells, tile_xyz, tile_fields, pairs, identities, origins, spans, counts, edge_xyz,
        normals, flags, controllers, fields, region)
    if region is not None:
        fallback = fallback_junction_leaves(fallback, region, flags.any(axis=2), pairs)
    fallback_leaves = xp.zeros(leaves, dtype=bool) if region is None else xp.copy(region)
    fallback_leaves[xp.asarray(sorted(fallback), dtype=np.int64)] = True

    junction_cells = xp.asarray(sorted(junctions), dtype=np.int64)
    junction_pairs = xp.asarray([sorted(junctions[c]['pair']) for c in sorted(junctions)],
                                dtype=np.int64).reshape(-1, 2)
    junction_ids = xp.full(leaves, -1, dtype=np.int64)
    junction_ids[junction_cells] = joint_start+xp.arange(len(junction_cells))
    vertex_ids = xp.arange(joint_start).reshape(n, leaves)
    borrowers, borrowed_cells = xp.nonzero(borrowed)
    vertex_ids[borrowers, borrowed_cells] = source[borrowers, borrowed_cells]*leaves+borrowed_cells
    for pair_column in range(2):
        vertex_ids[junction_pairs[:, pair_column], junction_cells] = junction_ids[junction_cells]
    # A key's first identity: the surface's own, or the lower one of a junction pair.
    fault_identities = {identities[c] for c in (fault_merge or {})}
    fault_surface = np.array([identities[s] in fault_identities for s in range(n)], dtype=bool)
    lower = np.array([min(pair, key=lambda s: identities[s]) for pair in host(junction_pairs).tolist()],
                     dtype=np.int64)
    fault_vertex = xp.asarray(np.concatenate((np.repeat(fault_surface, leaves), fault_surface[lower])))

    selected, fraction = edge_crossing_decisions(edge_fields, controllers, complex_.edge_cells, counts)
    plan, filtered_triangles = plan_edge_rows(selected, complex_.edge_cells, vertex_ids, borrowed, fault_vertex)
    triangle_normals = crossing_normals(fraction[plan['surface'], plan['edge']][None],
                                        endpoint_gradients[plan['surface'], edge_slot[plan['edge']]][None])[0]
    # endregion

    # region Validation and emission
    seams, coarsefine = validate_adaptive_incidence(plan, junction_ids, junction_pairs, joint_start, evidence, spans)
    junction_positions = np.asarray([junctions[c]['position'] for c in sorted(junctions)], dtype=float)
    positions = xp.concatenate((regular.reshape(-1, 3), xp.asarray(junction_positions.reshape(-1, 3))))
    participants = (junction_pairs*leaves+junction_cells[:, None]).reshape(-1)
    if bool(xp.isnan(positions[participants]).any()):
        raise ValueError('missing_adaptive_vertex: junction participant has no vertex')
    positions[participants] = np.nan
    host_origins, host_spans = host(origins), host(spans)
    host_cells, host_pairs = host(junction_cells), host(junction_pairs)
    vertex_order = _key_order(host(xp.flatnonzero(~xp.isnan(positions[:, 0]))), joint_start, leaves, identities,
                              host_origins, host_spans, host_cells, host_pairs)
    keys = _vertex_keys(vertex_order, joint_start, leaves, identities, host_origins, host_spans, junctions,
                        host_cells)
    vertices = host(positions[xp.asarray(vertex_order)]).reshape(-1, 3)
    host_ids = np.full(len(positions), -1, dtype=np.int64)
    host_ids[vertex_order] = np.arange(len(vertex_order))
    ids = xp.asarray(host_ids)
    triangles = plan_adaptive_triangles(plan, positions, triangle_normals, joint_start, n)
    output_faces, affected = (None, None) if defer_emission else emit_adaptive_triangles(triangles, ids)
    seam_edges = np.sort(host_ids[np.asarray(sorted(seams), dtype=np.int64).reshape(-1, 2)], axis=1)
    seam_edges = seam_edges[np.lexsort(seam_edges.T[::-1])] if len(seam_edges) else seam_edges
    # endregion

    result = dict(vertices=vertices, faces=output_faces, vertex_keys=keys, seam_edges=seam_edges,
                  junction_cells=host_cells.astype(int), leaf_origins=host_origins.copy(),
                  leaf_spans=host_spans.copy(), affected_faces=affected,
                  diagnostics=dict(algorithm='adaptive_joint_hermite_qef_dual_contouring',
                                   coarsefine_seam_edge_count=len(coarsefine), no_uniform_fallback=True,
                                   no_post_reconciliation=True, original_hermite_rows=12,
                                   sampled_point_count=len(query)+batch.sampled, canonical_face_tile_count=n_faces,
                                   minimal_edge_count=n_edges, finite_faults='unsupported',
                                   interior_topology='sampled_boundary_only_not_interior_certified'))
    if defer_emission:
        result.update(_triangle_plan=triangles, _key_to_id=ids)
    if include_reference:
        regular_flat = regular.reshape(-1, 3)
        reference_order = _key_order(host(xp.flatnonzero(~xp.isnan(regular_flat[:, 0]))), joint_start, leaves,
                                     identities, host_origins, host_spans, host_cells, host_pairs)
        reference_ids = np.full(len(regular_flat), -1, dtype=np.int64)
        reference_ids[reference_order] = np.arange(len(reference_order))
        rings = plan['regular'][plan['entry']]
        row, reference_triangles, _ = ring_triangles(rings, (rings >= 0).sum(axis=1))
        reference_plan = dict(surface=plan['surface'][plan['entry']][row], keys=reference_triangles)
        reference_faces, _ = emit_adaptive_triangles(plan_adaptive_triangles(
            reference_plan, regular_flat, triangle_normals[plan['entry']][row], joint_start, n),
            xp.asarray(reference_ids))
        result['reference'] = dict(
            vertices=host(regular_flat)[reference_order].reshape(-1, 3), faces=reference_faces,
            vertex_keys=_vertex_keys(reference_order, joint_start, leaves, identities, host_origins, host_spans,
                                     {}, ()),
            description='Original production QEFs with identical retained minimal edges')
    if fault_merge is not None:
        result['diagnostics'].update(
            contact_type='pretty_fault_vertex_merge', strict_crossings=True,
            crossing_contract='strict_scalar_sides', fault_surfaces=sorted(int(c) for c in fault_merge),
            fault_cell_count={int(c): int(flags[c].any(axis=1).sum()) for c in fault_merge},
            borrowed_vertex_count=int(borrowed.sum()), filtered_fault_triangle_count=filtered_triangles,
            multi_fault_unborrowed_count=int(multi_fault.sum()),
            weighted_fault_vertex_count=len(weighted_fault_cells),
            cross_surface_weight=DEFAULT_CROSS_SURFACE_WEIGHT,
            fault_overridden_junction_cells=sorted(int(c) for c in fallback),
            fallback_region_leaf_count=int(fallback_leaves.sum()))
    if separator is not None:
        result['bank'] = bank
        result['diagnostics'].update(bank=bank, separator_contact=separator,
                                     contact_type='explicit_fault_separator', separator_geometry='planar',
                                     strict_crossings=True, crossing_contract='strict_scalar_sides')
    return result


def _plan_junctions(xp, face_cells, tile_xyz, tile_fields, pairs, identities, origins, spans, counts, edge_xyz,
                    normals, flags, controllers, fields, region):
    """``plan_adaptive_junctions`` on the host, over only the tiles some contact pair straddles.

    Inputs are reduced to those tiles and their leaves (renumbered); results
    use the original tile and leaf indices.
    """
    candidates = xp.zeros(len(face_cells), dtype=bool)
    for controller, target in pairs:
        pair_fields = tile_fields[[controller, target]]
        candidates |= ~((xp.min(pair_fields, axis=2) > 0) | (xp.max(pair_fields, axis=2) < 0)).any(axis=0)
    tiles = xp.flatnonzero(candidates)
    tile_cells = face_cells[tiles]
    cells = xp.unique(tile_cells[tile_cells >= 0])
    local = xp.full(len(origins), -1, dtype=np.int64)
    local[cells] = xp.arange(len(cells))
    n = len(counts)
    junctions, evidence, fallback = plan_adaptive_junctions(
        host(xp.where(tile_cells >= 0, local[xp.maximum(tile_cells, 0)], -1)), host(tile_xyz[tiles]),
        host(tile_fields[:, tiles]), pairs, identities, host(origins[cells]), host(spans[cells]),
        host(counts[:, cells]), host(xp.stack([edge_xyz[s][cells] for s in range(n)])),
        host(xp.stack([normals[s][cells] for s in range(n)])), host(flags[:, cells]),
        controllers=controllers, leaf_fields=host(fields[:, cells]),
        overridden=None if region is None else {s: host(region[cells]) for s in range(n)})
    cells, tiles = host(cells).tolist(), host(tiles).tolist()
    junctions = {cells[c]: junction for c, junction in junctions.items()}
    evidence = [dict(item, tile=tiles[item['tile']], cells=tuple(cells[c] for c in item['cells'])) for item in evidence]
    return junctions, evidence, {cells[c] for c in fallback}


class _LeafRows:
    """Per-surface (N, 12, 3) values stored only on each surface's crossed leaves (zero elsewhere).

    ``rows[s]`` are surface ``s``'s leaves and ``values[s]`` their (R, 12, 3)
    rows; ``self[s][leaves]`` gathers (L, 12, 3) as the dense array would.
    """

    def __init__(self, xp, rows, values, leaves):
        self.rows, self.values, self._xp, self._slots = rows, values, xp, []
        for surface_rows in rows:
            slot = xp.full(leaves, -1, dtype=np.int64)
            slot[surface_rows] = xp.arange(len(surface_rows))
            self._slots.append(slot)

    def zeros_like(self):
        rows = _LeafRows.__new__(_LeafRows)
        rows.rows, rows._xp, rows._slots = self.rows, self._xp, self._slots
        rows.values = [self._xp.zeros(tuple(v.shape)) for v in self.values]
        return rows

    def __getitem__(self, surface):
        slot, values, xp = self._slots[surface], self.values[surface], self._xp

        def gather(leaves):
            index = slot[leaves]
            out = xp.zeros(tuple(index.shape) + (12, 3))
            hit = index >= 0
            out[hit] = values[index[hit]]
            return out
        return _Gather(gather)


class _Gather:
    def __init__(self, gather):
        self._gather = gather

    def __getitem__(self, leaves):
        return self._gather(leaves)


def _key_order(vertex_ids, joint_start, leaves, identities, origins, spans, junction_cells, junction_pairs):
    """``vertex_ids`` sorted by their public keys: joint before regular, then identities, origin, span."""
    vertex_ids = np.asarray(vertex_ids, dtype=np.int64)
    identity = np.asarray(identities, dtype=np.int64).reshape(-1, 2)
    joint = vertex_ids >= joint_start
    j = np.where(joint, vertex_ids-joint_start, 0)
    cells = np.where(joint, junction_cells[j] if len(junction_cells) else 0, vertex_ids % leaves)
    if len(junction_pairs):
        pair = junction_pairs[j]
        # Junction keys hold the pair's identities in sorted (identity) order.
        a, b = identity[pair[:, 0]], identity[pair[:, 1]]
        a_first = (a[:, 0] < b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] <= b[:, 1]))
        first = np.where(a_first[:, None], a, b)
        second = np.where(a_first[:, None], b, a)
    else:
        first = second = np.zeros((len(vertex_ids), 2), dtype=np.int64)
    own = identity[np.minimum(vertex_ids // leaves, len(identity)-1)]
    first = np.where(joint[:, None], first, own)
    second = np.where(joint[:, None], second, 0)
    origin = origins[cells]
    order = np.lexsort((spans[cells], origin[:, 2], origin[:, 1], origin[:, 0], second[:, 1], second[:, 0],
                        first[:, 1], first[:, 0], ~joint))
    return vertex_ids[order]


def _vertex_keys(vertex_ids, joint_start, leaves, identities, origins, spans, junctions, junction_cells):
    """Public symbolic keys ``(kind, identities, leaf origin, leaf span, 0)`` of ``vertex_ids``."""
    keys = []
    joint = vertex_ids >= joint_start
    origin_rows = origins[vertex_ids % leaves].astype(np.int64).tolist()
    span_rows = spans[vertex_ids % leaves].astype(np.int64).tolist()
    surfaces = (vertex_ids // leaves).tolist()
    with _paused_gc():
        for v, is_joint, s, origin, span in zip(vertex_ids.tolist(), joint.tolist(), surfaces, origin_rows,
                                                span_rows):
            if is_joint:
                keys.append(junctions[int(junction_cells[v-joint_start])]['key'])
            else:
                keys.append(('regular', (identities[s],), tuple(origin), span, 0))
    return keys


def _tensor(a, boolean=False):
    """``a`` as an engine-backend array (production primitives' dtype and device)."""
    if BT.engine_backend is AvailableBackends.PYTORCH:
        import torch
        return torch.as_tensor(a, dtype=torch.bool if boolean else BT.dtype_obj,
                               device=BT.device if BT.use_gpu else 'cpu')
    return np.asarray(host(a), dtype=bool if boolean else BT.dtype)


def _surface_crossings(xp, xyz, samples, levels, stricts, merged, multi_fault, fields, *, strict,
                       reject_aligned=False):
    """Production edge crossings per surface on the strict sides, plus branch counts.

    Leaves whose geometry a surface discards (borrowed or multi-fault) are
    neither checked nor classified (count one). ``reject_aligned`` rejects any
    original corner on a surface (bank geometry has no sided rule for it).
    Returns ``(counts, crossings)``: crossings as ``_LeafRows`` on crossed leaves.
    """
    n, leaves = stricts.shape[:2]
    edge_start, edge_end = xp.asarray(EDGE_START), xp.asarray(EDGE_END)
    rows_per_surface, values = [], []
    for s in range(n):
        if reject_aligned and bool((xp.abs(fields[s]) < 1e-10).any()):
            raise ValueError('sample_aligned_interface: original corner lies on surface')
        # Only leaves the surface strictly crosses carry crossings.
        rows = xp.flatnonzero(stricts[s].any(axis=1))
        crossed = stricts[s][rows]
        found = xp.zeros((len(rows), 12, 3))
        if len(rows):
            crossings, production_flags = find_intersection_on_edge(
                _tensor(xyz[rows].reshape(-1, 3)), _tensor(samples[s][rows].reshape(-1)), _tensor(levels[s:s+1]),
                **({'strict_crossings': True} if strict else {}))
            production_flags = xp.asarray(production_flags, dtype=bool).reshape(len(rows), 12)
            # Strict sides decide which edges cross; production supplies the positions,
            # and a strict edge production does not report is interpolated linearly.
            found[production_flags] = xp.asarray(crossings, dtype=np.float64)
            missing = crossed & ~production_flags
            if bool(missing.any()):
                corners, f = xyz[rows], fields[s][rows]
                start, end = corners[:, edge_start][missing], corners[:, edge_end][missing]
                f_start, f_end = f[:, edge_start][missing], f[:, edge_end][missing]
                found[missing] = start + (f_start/(f_start-f_end))[:, None]*(end-start)
            found[~crossed] = 0
        rows_per_surface.append(rows)
        values.append(found)
    # Ordinary dual contouring: one vertex per crossed leaf, whatever its branch topology.
    counts = xp.astype(stricts.any(axis=2) | merged | multi_fault, np.int64)
    return counts, _LeafRows(xp, rows_per_surface, values, leaves)


def _solve_qef(xp, crossing_xyz, own, xyz, spans, spacing, origins, crossing_normals, extra):
    """Production QEF vertices of one surface's crossed leaves, with optional extra weighted rows.

    All inputs are restricted to the solved leaves (the QEF is per leaf).
    """
    dc = DualContouringData(
        _tensor(crossing_xyz), _tensor(own, True), _tensor(xyz.mean(axis=1)),
        _tensor(spans[:, None]*spacing), 1, _tensor(origins), gradients=_tensor(crossing_normals),
        strict_crossings=False)
    if extra is not None:
        dc.extra_edge_xyz, dc.extra_edge_normals, dc.extra_weights = map(_tensor, extra)
    positions = xp.asarray(generate_dual_contouring_vertices(dc), dtype=np.float64)
    if not bool(xp.isfinite(positions).all()):
        raise ValueError('nonfinite_qef_solution: original production solve failed')
    return positions
