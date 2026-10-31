"""Explicit leaf-native joint dual contouring; no production dispatch or repairs.

Orchestration only: lattice, ownership, edge decisions and triangle planning
are module functions; this file samples fields, solves the production QEFs and
passes data between them.
"""

from collections.abc import Sequence
from functools import partial
from itertools import repeat

import numpy as np

from ...config import AvailableBackends
from ...core.backend_tensor import BackendTensor as BT
from ...core.compute_scope import checking
from ...core.data.dual_contouring_data import DualContouringData
from ...core.data.stack_relation_type import StackRelationType as R
from ...modules.dual_contouring._gen_vertices import generate_dual_contouring_vertices, triangles_in_extent_face
from ...modules.dual_contouring.dual_contouring_interface import (find_intersection_on_edge,
                                                                  find_strict_row_intersections)
from ...modules.dual_contouring.joint_cell_branches import CORNERS, EDGE_START, EDGE_END
from ...modules.dual_contouring.joint_cell_complex import (AdaptiveComplex, _ordering_code, _paused_gc,
                                                           adaptive_complex_arrays, partial_complex_arrays)
from ...modules.dual_contouring.joint_contact_relations import build_contact_relations
from ...modules.dual_contouring.joint_edge_decisions import crossing_normals, edge_crossing_decisions
from ...modules.dual_contouring.joint_field_queries import ExactPointCache, TargetedFieldBatch
from ...modules.dual_contouring.joint_lattice import (lattice_frame, leaf_corner_nodes,
                                                      lookup_node_fields, minimal_edge_nodes, node_codes,
                                                      precision_tolerance, shared_node_fields, snap_straddling_nodes,
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
                                                            plan_adaptive_triangles, plan_edge_rows, ring_triangles,
                                                            validate_adaptive_incidence)
from ...modules.dual_contouring.joint_xp import NUMPY, host, namespace
from ...modules.dual_contouring.weighted_qef_setup_multicore import DEFAULT_CROSS_SURFACE_WEIGHT


def extract_adaptive_topology(origins, spans, domain_shape, extent, scalar_samples,
                              surface_to_stack, surface_indices, stack_relations, isovalues,
                              *, sample_fields, ownership=None, include_reference=False,
                              faults_relations=None, separator_contacts=None, defer_emission=False,
                              cell_complex=None, fault_merge=None, field_query=None, snap_extent=False,
                              balanced=False):
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
    ``snap_extent`` puts vertices of leaves crossed at the extent on the extent
    faces (``mesh_extraction_snap_to_extent``); junctions there take their root
    on the face.
    ``cell_complex`` may supply the read-only ``adaptive_complex_arrays`` result
    for these exact origins/spans/domain, so bank partitions share one build.
    Otherwise, outside the banks, only the part of the complex the surfaces use
    is built (``_contact_complex``). ``balanced`` declares the leaves to be
    ``balance_joint_leaves`` output: they are then validated, by building the full
    complex and comparing, only in check mode. The ``canonical_face_tile_count``
    and ``minimal_edge_count`` diagnostics count the tiles and edges examined.
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
    for crossings and selected-edge endpoints), and vertex checks are one
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
    full_complex = None
    if cell_complex is not None:
        full_complex = AdaptiveComplex(*(xp.asarray(a) for a in cell_complex))
    elif separator is not None or not balanced or checking():
        full_complex = adaptive_complex_arrays(origins, spans, domain_shape)
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
    if field_query is not None and (separator_contacts is not None or ownership is not None):
        raise ValueError('invalid_field_query_contract: exclusive with separator contacts and ownership')
    plane = []

    def on_separator_plane(points, raw, gradients):
        if plane:
            check_separator_samples(host(points), host(raw[separator]), host(gradients[separator]), plane[0],
                                    precision_tolerance(1e-10, BT.dtype))

    query = ExactPointCache(sample_fields, n, on_separator_plane if separator is not None else None, xp)

    def sample(points):
        return query(points.reshape(-1, 3))

    batch = TargetedFieldBatch(field_query, xp)
    lattice = None
    if separator is None:
        # Every tile and minimal-edge node is a leaf corner: reuse the supplied
        # corner values, one side per node for every leaf sharing it. A
        # re-queried node (low precision) could sit on the other side of a
        # surface and drop the dual face.
        nodes = node_codes(corners, domain)
        samples = snap_straddling_nodes(corners, samples, host_levels, domain, nodes=nodes)
        fields = samples-levels[:, None, None]
        lattice = shared_node_fields(corners, fields, domain, nodes)
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
    flags = (fields[:, :, xp.asarray(EDGE_START)]*fields[:, :, xp.asarray(EDGE_END)]) < 0
    merged, multi_fault, source = borrowed_leaves(merge_targets, flags)
    borrowed = source >= 0
    counts, edge_xyz = _surface_crossings(xp, xyz, samples, levels, flags, merged, multi_fault, fields,
                                          strict=separator is not None or fault_merge is not None,
                                          reject_aligned=separator is not None)
    # endregion

    # region Cell complex and its fields
    fallback_ring = bool(borrowed.any()) or bool(multi_fault.any())
    if cell_complex is not None or separator is not None:
        complex_ = full_complex
    else:
        complex_ = _contact_complex(xp, origins, spans, domain_shape, flags, pairs, fallback_ring)
    # One region for every surface: a junction given up there leaves both
    # participants, affected or not, with unshared vertices.
    region = fallback_region(complex_.edge_cells, (merged | multi_fault).any(axis=0)) if fallback_ring else None
    if separator is None:
        if full_complex is not None and complex_ is not full_complex:
            _check_contact_complex(xp, complex_, full_complex, counts, pairs, lattice)
        complex_ = _dual_face_edges(xp, complex_, counts)
    n_faces, n_edges = len(complex_.face_axis), len(complex_.edge_axis)
    tiles = tile_nodes(complex_.face_axis, complex_.face_origin, complex_.face_span)
    edge_nodes = minimal_edge_nodes(complex_.edge_axis, complex_.edge_origin, complex_.edge_span)
    tile_xyz, minimal_xyz = low+tiles*step, low+edge_nodes*step
    all_endpoint_gradients = None
    if separator is None:
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
    selected, fraction = edge_crossing_decisions(edge_fields, controllers, complex_.edge_cells, counts)
    # Endpoint gradients orient ring triangles: only selected edges need them.
    used_edges = xp.flatnonzero(selected.any(axis=0))
    edge_slot = xp.full(n_edges, -1, dtype=np.int64)
    edge_slot[used_edges] = xp.arange(len(used_edges))
    if all_endpoint_gradients is not None:
        endpoint_gradients = all_endpoint_gradients[:, used_edges]
    elif field_query is None and len(used_edges):
        endpoint_gradients = sample(minimal_xyz[used_edges])[1].reshape(n, len(used_edges), 2, 3)
    else:
        endpoint_gradients = xp.zeros((n, len(used_edges), 2, 3))
    # endregion

    # region Branches and production QEFs
    normals = edge_xyz.zeros_like()
    # A surface never solves its own QEF in borrowed leaves; there its Hermite
    # rows only constrain the fault vertex it borrows (pretty's weighted rule).
    solved = flags & ~merged[:, :, None]
    # Every surface's crossings at once (surface-major rows); ``point_bounds`` split them per surface.
    crossed = edge_xyz.crossed
    crossing_points = edge_xyz.flat[crossed]
    point_bounds = edge_xyz.segments(crossed.sum(axis=1))
    if field_query is None:
        if len(crossing_points):
            point_surface = xp.repeat(edge_xyz.surface, 12).reshape(-1, 12)[crossed]
            normals.flat[crossed] = sample(crossing_points)[1][point_surface, xp.arange(len(crossing_points))]
    else:
        edge_flat = xp.flatnonzero(selected)
        edge_surface, surface_edges = edge_flat // n_edges, edge_flat % n_edges
        edge_bounds = np.concatenate(([0], np.cumsum(host(selected.sum(axis=1))))).tolist()
        # One request per surface: its crossings, then the endpoints of its selected minimal edges.
        endpoints = minimal_xyz[surface_edges].reshape(-1, 3)
        answers = batch([(xp.concatenate((crossing_points[a:b], endpoints[2*c:2*d])), [s], 'gradient')
                         for s, ((a, b), c, d) in enumerate(zip(point_bounds, edge_bounds[:-1], edge_bounds[1:]))])
        normals.flat[crossed] = xp.concatenate([answers[s][0][:b-a] for s, (a, b) in enumerate(point_bounds)])
        endpoint_gradients[edge_surface, edge_slot[surface_edges]] = xp.concatenate(
            [answers[s][0][b-a:].reshape(-1, 2, 3) for s, (a, b) in enumerate(point_bounds)])
    own = solved[edge_xyz.surface, edge_xyz.flat_rows]
    if bool(((xp.norm(normals.flat, axis=2) < 1e-12) & own).any()):
        raise ValueError('degenerate_hermite_normal: actual crossing gradient vanishes')

    # Vertex ids: s*leaves + leaf is surface s's own vertex, joint_start + j the j-th junction.
    joint_start = n*leaves
    regular = xp.full((n, leaves, 3), np.nan)
    weighted_fault_cells = []  # per fault surface: its leaves whose QEF has weighted borrowed rows (count)
    solving = own.any(axis=1)
    solving_bounds = edge_xyz.segments(solving)
    keep = xp.flatnonzero(solving)
    extras = {}
    for s in (fault_merge or {}):
        a, b = solving_bounds[s]
        if b > a:
            cells = edge_xyz.flat_rows[keep[a:b]]
            extra = borrowed_hermite_rows(s, cells, fault_merge[s], source, edge_xyz, normals, flags,
                                          DEFAULT_CROSS_SURFACE_WEIGHT)
            if extra is not None:
                extras[s] = extra
                weighted_fault_cells.append((extra[2] > 0).any(axis=1).sum())
    for group in _qef_groups(xp, solving_bounds, extras, spans[edge_xyz.flat_rows[keep]], snap_extent):
        index = keep[xp.asarray(np.concatenate([np.arange(*solving_bounds[s]) for s in group]))]
        cells, group_own = edge_xyz.flat_rows[index], own[index]
        extra = (tuple(xp.concatenate([extras[s][k] for s in group]) for k in range(3))
                 if group[0] in extras else None)
        regular[edge_xyz.surface[index], cells] = _solve_qef(
            xp, edge_xyz.flat[index][group_own], group_own, xyz[cells], spans[cells], step, origins[cells],
            normals.flat[index][group_own], extra, extent=host(bounds) if snap_extent else None)
    # endregion

    # region Junctions and edge rows
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

    plan, filtered_triangles = plan_edge_rows(selected, complex_.edge_cells, vertex_ids, borrowed, fault_vertex)
    triangle_normals = crossing_normals(fraction[plan['surface'], plan['edge']][None],
                                        endpoint_gradients[plan['surface'], edge_slot[plan['edge']]][None])[0]
    # endregion

    # region Validation and emission
    seams, coarsefine = validate_adaptive_incidence(plan, junction_ids, junction_pairs, joint_start, evidence, spans)
    junction_positions = np.asarray([junctions[c]['position'] for c in sorted(junctions)], dtype=float)
    if snap_extent and len(junction_positions):
        roots = np.asarray([junctions[c]['roots'] for c in sorted(junctions)], dtype=float)
        junction_positions = _snap_junctions(junction_positions, roots, host(bounds), 1e-3*float(np.min(host(step))))
    positions = xp.concatenate((regular.reshape(-1, 3), xp.asarray(junction_positions.reshape(-1, 3))))
    participants = (junction_pairs*leaves+junction_cells[:, None]).reshape(-1)
    if bool(xp.isnan(positions[participants]).any()):
        raise ValueError('missing_adaptive_vertex: junction participant has no vertex')
    positions[participants] = np.nan
    host_origins, host_spans = host(origins), host(spans)
    host_cells, host_pairs = host(junction_cells), host(junction_pairs)
    vertex_order = _key_order(host(xp.flatnonzero(~xp.isnan(positions[:, 0]))), joint_start, leaves, identities,
                              host_origins, host_spans, host_cells, host_pairs)
    keys = VertexKeys(partial(_vertex_keys, vertex_order, joint_start, leaves, identities, host_origins, host_spans,
                              junctions, host_cells))
    vertices = host(positions[xp.asarray(vertex_order)]).reshape(-1, 3)
    host_ids = np.full(len(positions), -1, dtype=np.int64)
    host_ids[vertex_order] = np.arange(len(vertex_order))
    ids = xp.asarray(host_ids)
    triangles = plan_adaptive_triangles(plan, positions, triangle_normals, joint_start, n)
    if snap_extent:
        in_face = triangles_in_extent_face(positions[triangles['keys']], xp.asarray(bounds[::2]),
                                           xp.asarray(bounds[1::2]), 1e-3*float(np.min(host(step))))
        triangles = dict(triangles, **{k: triangles[k][~in_face] for k in ('surface', 'keys', 'affected')})
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
        faults_merged = sorted(int(c) for c in fault_merge)
        # Every count in one host read.
        tallies = host(xp.stack([flags[c].any(axis=1).sum() for c in faults_merged] + [
            borrowed.sum(), multi_fault.sum(), fallback_leaves.sum(),
            sum(weighted_fault_cells) if weighted_fault_cells else xp.zeros((), dtype=np.int64)])).tolist()
        result['diagnostics'].update(
            contact_type='pretty_fault_vertex_merge', strict_crossings=True,
            crossing_contract='strict_scalar_sides', fault_surfaces=faults_merged,
            fault_cell_count=dict(zip(faults_merged, tallies)),
            borrowed_vertex_count=tallies[-4], filtered_fault_triangle_count=filtered_triangles,
            multi_fault_unborrowed_count=tallies[-3],
            weighted_fault_vertex_count=tallies[-1],
            cross_surface_weight=DEFAULT_CROSS_SURFACE_WEIGHT,
            fault_overridden_junction_cells=sorted(int(c) for c in fallback),
            fallback_region_leaf_count=tallies[-2])
    if separator is not None:
        result['bank'] = bank
        result['diagnostics'].update(bank=bank, separator_contact=separator,
                                     contact_type='explicit_fault_separator', separator_geometry='planar',
                                     strict_crossings=True, crossing_contract='strict_scalar_sides')
    return result


def _contact_complex(xp, origins, spans, domain_shape, flags, pairs, every_ring):
    """The part of the cell complex extraction uses; ``flags`` (S, N, 12) are the surfaces' leaf-edge crossings.

    A tile some contact pair straddles is a whole face of a leaf both cross (the finer one where
    sizes meet), so tiles come from those leaves; without pairs there are none. A dual face needs a
    crossing on a minimal edge, which is a whole edge of the finest leaf around it, and snapped
    corners agree in sign between leaves: that leaf flags the crossing. So edges come from the
    flagged leaf edges, or from every leaf if ``every_ring`` (the fallback region walks all rings
    around its leaves).
    """
    crossed = flags.any(axis=2)
    both = xp.zeros(len(origins), dtype=bool)
    for controller, target in pairs:
        both |= crossed[controller] & crossed[target]
    return partial_complex_arrays(origins, spans, domain_shape, xp.flatnonzero(both),
                                  None if every_ring else xp.nonzero(flags.any(axis=0)))


def _dual_face_edges(xp, complex_, counts):
    """``complex_`` without the edges no surface can emit, in the same order.

    Those are exterior edges and edges without a surface that has a vertex in all four ring
    leaves (``edge_crossing_decisions`` drops them, whatever their fields).
    """
    cells = complex_.edge_cells
    keep = xp.flatnonzero((cells >= 0).all(axis=1) & (counts == 1)[:, xp.maximum(cells, 0)].all(axis=2).any(axis=0))
    return complex_._replace(edge_axis=complex_.edge_axis[keep], edge_origin=complex_.edge_origin[keep],
                             edge_span=complex_.edge_span[keep], edge_cells=cells[keep])


def _straddled_tiles(xp, tile_fields, pairs):
    """(F,) mask of the tiles some contact pair straddles."""
    straddled = xp.zeros(tile_fields.shape[1], dtype=bool)
    for controller, target in pairs:
        pair_fields = tile_fields[[controller, target]]
        straddled |= ~((xp.min(pair_fields, axis=2) > 0) | (xp.max(pair_fields, axis=2) < 0)).any(axis=0)
    return straddled


def _check_contact_complex(xp, partial, full, counts, pairs, lattice):
    """Raise unless extraction reads the same crossed edges and junction tiles from both complexes."""
    def crossed_edges(complex_):
        kept = _dual_face_edges(xp, complex_, counts)
        nodes = minimal_edge_nodes(kept.edge_axis, kept.edge_origin, kept.edge_span)
        fields = lookup_node_fields(nodes, *lattice).reshape(len(counts), -1, 2)
        rows = (fields[:, :, 0]*fields[:, :, 1] < 0).any(axis=0)
        return tuple(column[rows] for column in kept[4:])

    def junction_tiles(complex_):
        nodes = tile_nodes(complex_.face_axis, complex_.face_origin, complex_.face_span)
        rows = _straddled_tiles(xp, lookup_node_fields(nodes, *lattice).reshape(len(counts), -1, 4), pairs)
        return tuple(column[rows] for column in complex_[:4])

    def same(first, second):
        return all(len(a) == len(b) and bool((a == b).all()) for a, b in zip(first, second))

    if not same(crossed_edges(full), crossed_edges(partial)):
        raise AssertionError('partial cell complex: edges differ from the full complex')
    if not same(junction_tiles(full), junction_tiles(partial)):
        raise AssertionError('partial cell complex: junction tiles differ from the full complex')


def _plan_junctions(xp, face_cells, tile_xyz, tile_fields, pairs, identities, origins, spans, counts, edge_xyz,
                    normals, flags, controllers, fields, region):
    """``plan_adaptive_junctions`` on the host, over only the tiles some contact pair straddles.

    Inputs are reduced to those tiles and their leaves (renumbered); results
    use the original tile and leaf indices.
    """
    tiles = xp.flatnonzero(_straddled_tiles(xp, tile_fields, pairs))
    tile_cells = face_cells[tiles]
    cells = xp.unique(tile_cells[tile_cells >= 0])
    local = xp.full(len(origins), -1, dtype=np.int64)
    local[cells] = xp.arange(len(cells))
    n = len(counts)
    junctions, evidence, fallback = plan_adaptive_junctions(
        host(xp.where(tile_cells >= 0, local[xp.maximum(tile_cells, 0)], -1)), host(tile_xyz[tiles]),
        host(tile_fields[:, tiles]), pairs, identities, host(origins[cells]), host(spans[cells]),
        host(counts[:, cells]), host(edge_xyz.gather(slice(None), cells)),
        host(normals.gather(slice(None), cells)), host(flags[:, cells]),
        controllers=controllers, leaf_fields=host(fields[:, cells]),
        overridden=None if region is None else {s: host(region[cells]) for s in range(n)})
    cells, tiles = host(cells).tolist(), host(tiles).tolist()
    junctions = {cells[c]: junction for c, junction in junctions.items()}
    evidence = [dict(item, tile=tiles[item['tile']], cells=tuple(cells[c] for c in item['cells'])) for item in evidence]
    return junctions, evidence, {cells[c] for c in fallback}


class _LeafRows:
    """Per-surface (N, 12, 3) values stored only on each surface's crossed leaves (zero elsewhere).

    Stored flat, surface-major: ``flat`` (R, 12, 3), with each row's ``surface`` and leaf
    (``flat_rows``) and its strictly crossed edges ``crossed`` (R, 12); ``bounds`` are each surface's
    (start, stop) rows (host). ``rows[s]`` are surface ``s``'s leaves and ``values[s]`` their (R_s, 12, 3)
    rows (views of ``flat``); ``self[s][leaves]`` gathers (L, 12, 3) as the dense array would.
    """

    def __init__(self, xp, surface, rows, flat, crossed, bounds, leaves):
        self._xp, self.surface, self.flat_rows, self.flat, self.crossed = xp, surface, rows, flat, crossed
        self.bounds = bounds
        self.rows = [rows[a:b] for a, b in bounds]
        self.values = [flat[a:b] for a, b in bounds]
        self._slot = xp.full((len(bounds), leaves), -1, dtype=np.int64)
        self._slot[surface, rows] = xp.arange(len(rows))

    def zeros_like(self):
        rows = _LeafRows.__new__(_LeafRows)
        rows.__dict__.update(self.__dict__)
        rows.flat = self._xp.zeros(tuple(self.flat.shape))
        rows.values = [rows.flat[a:b] for a, b in self.bounds]
        return rows

    def segments(self, sizes):
        """Each surface's (start, stop) in a surface-major array holding ``sizes[r]`` entries per row (host)."""
        ends = np.concatenate(([0], host(self._xp.cumsum(self._xp.astype(sizes, np.int64)))))
        return [(int(ends[a]), int(ends[b])) for a, b in self.bounds]

    def gather(self, surface, leaves):
        """Rows of ``surface`` (an index, or a slice for every surface) at ``leaves``: zero where not crossed."""
        index = self._slot[surface, leaves]
        if not len(self.flat):
            return self._xp.zeros(tuple(index.shape) + (12, 3))
        return self._xp.where((index >= 0)[..., None, None], self.flat[self._xp.maximum(index, 0)], 0.)

    def __getitem__(self, surface):
        return _Gather(lambda leaves: self.gather(surface, leaves))


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
    columns = (~joint, first[:, 0], first[:, 1], second[:, 0], second[:, 1], origin[:, 0], origin[:, 1], origin[:, 2],
               spans[cells])
    radices = [int(c.max())+2 if len(c) else 1 for c in columns]
    # Identities and lattice coordinates are small: one packed code is usually enough. Negative stack
    # or surface numbers cannot pack.
    code = None if min(int(c.min()) if len(c) else 0 for c in columns) < 0 else _ordering_code(NUMPY, columns, radices)
    return vertex_ids[np.lexsort(columns[::-1]) if code is None else np.argsort(code)]


class VertexKeys(Sequence):
    """Public symbolic vertex keys, built on the first read: extraction never reads them itself."""

    def __init__(self, build):
        self._build, self._keys = build, None

    def _list(self):
        if self._keys is None:
            self._keys, self._build = self._build(), None
        return self._keys

    def __len__(self):
        return len(self._list())

    def __getitem__(self, index):
        return self._list()[index]

    def __iter__(self):
        return iter(self._list())

    def __eq__(self, other):
        return self._list() == list(other)


def _vertex_keys(vertex_ids, joint_start, leaves, identities, origins, spans, junctions, junction_cells):
    """Public symbolic keys ``(kind, identities, leaf origin, leaf span, 0)`` of ``vertex_ids``."""
    joint = vertex_ids >= joint_start
    origin_rows = origins[vertex_ids % leaves].astype(np.int64).tolist()
    span_rows = spans[vertex_ids % leaves].astype(np.int64).tolist()
    # Surface index of regular vertices (junction rows are replaced below).
    surfaces = np.minimum(vertex_ids // leaves, len(identities) - 1).tolist()
    single = [(identity,) for identity in identities]
    with _paused_gc():
        keys = list(zip(repeat('regular'), map(single.__getitem__, surfaces), map(tuple, origin_rows), span_rows,
                        repeat(0)))
        for row in np.flatnonzero(joint).tolist():
            keys[row] = junctions[int(junction_cells[int(vertex_ids[row])-joint_start])]['key']
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
    if reject_aligned and bool((xp.abs(fields) < 1e-10).any()):
        raise ValueError('sample_aligned_interface: original corner lies on surface')
    # Only leaves a surface strictly crosses carry crossings: every surface's, surface-major.
    crossed_leaves = stricts.any(axis=2)
    flat = xp.flatnonzero(crossed_leaves)
    surface, rows = flat // leaves, flat % leaves
    bounds = np.concatenate(([0], np.cumsum(host(crossed_leaves.sum(axis=1)))))
    crossed = stricts[surface, rows]
    found = xp.zeros((len(rows), 12, 3))
    if len(rows):
        corners, scalars = xyz[rows], samples[surface, rows]
        if strict:  # each row against its surface's isovalue, in one production call
            crossings, production_flags = find_strict_row_intersections(
                _tensor(corners.reshape(-1, 3)), _tensor(scalars.reshape(-1)), _tensor(levels[surface]))
            crossings = [crossings]
            production_flags = xp.asarray(production_flags, dtype=bool).reshape(-1, 12)
        else:
            calls = [find_intersection_on_edge(_tensor(corners[a:b].reshape(-1, 3)), _tensor(scalars[a:b].reshape(-1)),
                                               _tensor(levels[s:s+1]))
                     for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])) if b > a]
            crossings = [c for c, _ in calls]
            production_flags = xp.concatenate([xp.asarray(f, dtype=bool).reshape(-1, 12) for _, f in calls])
        # Strict sides decide which edges cross; production supplies the positions,
        # and a strict edge production does not report is interpolated linearly.
        found[production_flags] = xp.concatenate([xp.asarray(c, dtype=np.float64) for c in crossings])
        missing = crossed & ~production_flags
        if bool(missing.any()):
            f = fields[surface, rows]
            start, end = corners[:, edge_start][missing], corners[:, edge_end][missing]
            f_start, f_end = f[:, edge_start][missing], f[:, edge_end][missing]
            found[missing] = start + (f_start/(f_start-f_end))[:, None]*(end-start)
        found[~crossed] = 0
    # Ordinary dual contouring: one vertex per crossed leaf, whatever its branch topology.
    counts = xp.astype(crossed_leaves | merged | multi_fault, np.int64)
    return counts, _LeafRows(xp, surface, rows, found, crossed, list(zip(bounds[:-1].tolist(), bounds[1:].tolist())),
                             leaves)


def _qef_groups(xp, bounds, extras, spans, snap_extent):
    """Surfaces whose QEFs are solved in one production call, in surface order.

    ``bounds`` are each surface's (start, stop) solved rows, ``spans`` those rows' leaf spans. A group
    shares the number of borrowed extra rows (``extras``) and, when vertices snap to the extent, the
    smallest leaf span: the snap tolerance scales with the call's smallest cell spacing.
    """
    present = [s for s, (a, b) in enumerate(bounds) if b > a]
    smallest = (host(xp.stack([spans[bounds[s][0]:bounds[s][1]].min() for s in present])).tolist()
                if snap_extent and present else [None] * len(present))
    groups = {}
    for s, span in zip(present, smallest):
        groups.setdefault((extras[s][0].shape[1] if s in extras else 0, span), []).append(s)
    return list(groups.values())


def _solve_qef(xp, crossing_xyz, own, xyz, spans, spacing, origins, crossing_normals, extra, extent=None):
    """Production QEF vertices of one surface's crossed leaves, with optional extra weighted rows.

    All inputs are restricted to the solved leaves (the QEF is per leaf).
    """
    dc = DualContouringData(
        _tensor(crossing_xyz), _tensor(own, True), _tensor(xyz.mean(axis=1)),
        _tensor(spans[:, None]*spacing), 1, _tensor(origins), gradients=_tensor(crossing_normals),
        strict_crossings=False, extent=extent)
    if extra is not None:
        dc.extra_edge_xyz, dc.extra_edge_normals, dc.extra_weights = map(_tensor, extra)
    positions = xp.asarray(generate_dual_contouring_vertices(dc), dtype=np.float64)
    if not bool(xp.isfinite(positions).all()):
        raise ValueError('nonfinite_qef_solution: original production solve failed')
    return positions


def _snap_junctions(positions, roots, bounds, tolerance):
    """Junctions whose contact line leaves through an extent face take that face root.

    ``roots`` (J, 2, 3) are the two tile roots the junction lies between.
    """
    on_face = (np.abs(roots[:, :, :, None] - bounds.reshape(1, 1, 3, 2)) <= tolerance).any(axis=(2, 3))
    positions = positions.copy()
    snapped = on_face.any(axis=1)
    positions[snapped] = roots[snapped, np.argmax(on_face[snapped], axis=1)]
    return positions
