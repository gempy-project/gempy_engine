"""Explicit leaf-native joint dual contouring; no production dispatch or repairs."""

import numpy as np

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as BT
from gempy_engine.core.data.dual_contouring_data import DualContouringData
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine.modules.dual_contouring._gen_vertices import generate_dual_contouring_vertices
from gempy_engine.modules.dual_contouring.weighted_qef_setup_multicore import DEFAULT_CROSS_SURFACE_WEIGHT
from gempy_engine.modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge
from gempy_engine.modules.dual_contouring.contact_topology import build_contact_relations
from gempy_engine.modules.dual_contouring.topology_extraction import CORNERS, EDGE_START, EDGE_END, classify_cell_branches
from gempy_engine.modules.dual_contouring.adaptive_joint_topology import (
    plan_adaptive_junctions, collapse_ring, ring_triangles, validate_adaptive_incidence, plan_adaptive_triangles,
    validate_adaptive_geometry, emit_adaptive_triangles,
)


def extract_adaptive_topology(origins, spans, domain_shape, extent, scalar_samples,
                              surface_to_stack, surface_indices, stack_relations, isovalues,
                              *, sample_fields, ownership=None, include_reference=False,
                              faults_relations=None, separator_contacts=None, defer_emission=False,
                              cell_complex=None, fault_merge=None, field_query=None, closure=False):
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
    (pretty-style fallback, counted, never silent). Faults are not ownership
    controllers.
    ``field_query([(points, surfaces, 'scalar'|'gradient'), ...])`` optionally
    replaces ``sample_fields`` with batched targeted queries, returning per
    request (S, M) scalars or (S, M, 3) gradients: tile and minimal-edge node
    values come from the supplied leaf corners (shared nodes must agree within
    tolerance), gradients are evaluated only for the crossing surface (one batch
    for crossings and crossed-edge endpoints), and vertex checks are one
    scalar-only batch. Not with separator contacts.
    ``closure=True`` additionally returns ``result['closure']``: the retained
    crossings of minimal edges lying in an extent face (whose dual faces need
    rim vertices), the vertex key of every surface at every boundary leaf, and
    the oriented key-based triangle plan, for the closed-volume builder.
    """
    from gempy_engine.modules.dual_contouring.adaptive_cell_complex import build_adaptive_complex

    if defer_emission and include_reference:
        raise ValueError('unsupported_deferred_reference: defer_emission and include_reference are incompatible')
    # Topology is host NumPy; Torch tensors for crossings/QEFs live on the active device.
    if BT.dtype != 'float64' or BT.engine_backend not in (AvailableBackends.numpy, AvailableBackends.PYTORCH):
        raise ValueError('unsupported_backend: NumPy/Torch float64 only')

    def detached(a):
        return a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a)

    def tensor(a, boolean=False):
        if BT.engine_backend is AvailableBackends.PYTORCH:
            import torch
            return torch.as_tensor(a, dtype=torch.bool if boolean else torch.float64,
                                   device=BT.device if BT.use_gpu else 'cpu')
        return np.asarray(a, dtype=bool if boolean else np.float64)

    allowed, faults, truncations = build_contact_relations(
        surface_to_stack, surface_indices, stack_relations, faults_relations, isovalues)
    if separator_contacts is None and fault_merge is None and (
            faults or any(r is R.FAULT for r in stack_relations)
            or (faults_relations is not None and np.asarray(faults_relations).any())):
        raise ValueError('unsupported_fault_extraction: adaptive bank incidence unavailable')
    merge_targets = {}
    if fault_merge is not None:
        if separator_contacts is not None or ownership is not None:
            raise ValueError('invalid_fault_merge_contract: exclusive with separator contacts and ownership')
        fault_surfaces = {i for i, g in enumerate(surface_to_stack) if stack_relations[g] is R.FAULT}
        expected = {i: set() for i in fault_surfaces}
        for c, t in faults:
            expected.setdefault(c, set()).add(t)
        try:
            supplied = {int(k): set(map(int, v)) for k, v in fault_merge.items()}
        except (AttributeError, TypeError, ValueError):
            supplied = None
        if supplied != expected or set(expected) != fault_surfaces:
            raise ValueError('invalid_fault_merge_contract: must map every fault surface to exactly its affected surfaces')
        for c, targets in supplied.items():
            for t in targets:
                merge_targets.setdefault(t, []).append(c)
    separator, bank = None, None
    if separator_contacts is not None:
        if not isinstance(separator_contacts, dict) or set(separator_contacts) != {
                'separator', 'bank', 'fault_pairs', 'controllers'}:
            raise ValueError('invalid_separator_contract: explicit separator, bank, pairs and controllers required')
        separator, bank = separator_contacts['separator'], separator_contacts['bank']
        if (not isinstance(separator, (int, np.integer)) or isinstance(separator, bool)
                or not 0 <= separator < len(surface_to_stack)
                or not isinstance(bank, (int, np.integer)) or isinstance(bank, bool) or bank not in (0, 1)):
            raise ValueError('invalid_separator_contract: invalid local separator or bank')
        supplied = separator_contacts['fault_pairs']
        if not isinstance(supplied, set) or not supplied:
            raise ValueError('unauthorized_fault_pairs: require exactly the represented directed fault edges')
        if any(not isinstance(pair, tuple) or len(pair) != 2 or any(
                not isinstance(index, (int, np.integer)) or isinstance(index, bool)
                or not 0 <= index < len(surface_to_stack) for index in pair) for pair in supplied):
            raise ValueError('invalid_fault_pairs: integer nonboolean in-bounds local source/target indices required')
        if supplied != faults or any(
                c != separator or c == t or surface_to_stack[c] == surface_to_stack[t] for c, t in supplied):
            raise ValueError('unauthorized_fault_pairs: require exactly the represented directed fault edges')
        if stack_relations[surface_to_stack[separator]] is not R.FAULT:
            raise ValueError('invalid_separator_contract: separator must belong to a fault stack')
        if [g for g, relation in enumerate(stack_relations) if relation is R.FAULT] != [surface_to_stack[separator]]:
            raise ValueError('unsupported_multiple_faults: full metadata must contain only the represented separator fault')
        if not isinstance(separator_contacts['controllers'], dict):
            raise ValueError('unsupported_ownership: explicit composite controller dict required')
        for c, t in supplied:
            allowed[c, t] = allowed[t, c] = True
    # Derive dependencies with every known boundary present so an omitted
    # controller cannot silently become unrestricted ownership of a target.
    full_identities = [(g, s) for g, values in enumerate(isovalues) for s in range(len(values))]
    _, _, full_truncations = build_contact_relations(
        np.array([g for g, _ in full_identities], dtype=int),
        np.array([s for _, s in full_identities], dtype=int), stack_relations, faults_relations, isovalues)
    represented_stacks = set(surface_to_stack)
    represented_boundaries = {(g, float(isovalues[g][s])) for g, s in zip(surface_to_stack, surface_indices)}
    for c, t in full_truncations:
        controller_stack, index = full_identities[c]
        if (full_identities[t][0] in represented_stacks
                and (controller_stack, float(isovalues[controller_stack][index])) not in represented_boundaries):
            raise ValueError('unsupported_missing_controller: represented target requires known ordinary boundary '
                             f'{controller_stack}/{index}')
    complex_ = build_adaptive_complex(origins, spans, domain_shape) if cell_complex is None else cell_complex
    origins, spans = np.asarray(origins), np.asarray(spans)
    domain = np.asarray(domain_shape)
    bounds = np.asarray(extent, dtype=float)
    if bounds.shape != (6,) or not np.isfinite(bounds).all() or np.any(bounds[1::2] <= bounds[::2]):
        raise ValueError('invalid_extent: increasing finite physical bounds required')
    spacing = (bounds[1::2]-bounds[::2])/domain
    corners = origins[:, None, :] + spans[:, None, None]*CORNERS
    xyz = bounds[::2] + corners*spacing
    identities = list(zip(map(int, surface_to_stack), map(int, surface_indices)))
    n, leaves = len(identities), len(origins)
    if not n or len(set(identities)) != n:
        raise ValueError('invalid_surface_ids: unique stack/local identities required')
    samples = detached(scalar_samples).astype(float)
    if samples.shape != (n, leaves, 8) or not np.isfinite(samples).all():
        raise ValueError('invalid_samples: finite (S,N,8) raw corners required')
    levels = np.array([isovalues[g][s] for g, s in identities])
    if separator is not None:
        levels[separator] = 0.
    fields = samples-levels[:, None, None]
    faces, edges = complex_['faces'], complex_['edges']
    # Tile nodes ordered 00, 10, 01, 11 in the two non-normal axes.
    tile_nodes = np.repeat(np.array([f['origin'] for f in faces], dtype=np.int64).reshape(-1, 1, 3), 4, axis=1)
    face_axes = np.array([f['axis'] for f in faces], dtype=np.int64)
    face_spans = np.array([f['span'] for f in faces], dtype=np.int64)
    for axis in range(3):
        rows = face_axes == axis
        other = [a for a in range(3) if a != axis]
        tile_nodes[np.ix_(rows, range(4), other)] += (
            np.array([[0, 0], [1, 0], [0, 1], [1, 1]])*face_spans[rows, None, None])
    tile_xyz = bounds[::2]+tile_nodes*spacing
    edge_nodes = np.repeat(np.array([e['origin'] for e in edges], dtype=np.int64).reshape(-1, 1, 3), 2, axis=1)
    edge_nodes[np.arange(len(edges)), 1, [e['axis'] for e in edges]] += [e['span'] for e in edges]
    minimal_xyz = bounds[::2]+edge_nodes*spacing
    # Exact-coordinate cache: key -> column of the concatenated sample blocks.
    cache, cached_raw, cached_gradients = {}, [], []
    separator_plane = None

    def query(points):
        points = np.asarray(points).reshape(-1, 3)
        keys = list(map(tuple, points))
        missing = sorted(set(keys)-cache.keys())
        if missing:
            raw, gradients = sample_fields(np.asarray(missing))
            raw, gradients = np.array(detached(raw), copy=True), np.array(detached(gradients), copy=True)
            if raw.shape != (n, len(missing)) or gradients.shape != (n, len(missing), 3) or not (
                    np.isfinite(raw).all() and np.isfinite(gradients).all()):
                raise ValueError('invalid_callback_samples: finite all-surface scalar/gradient arrays required')
            if separator_plane is not None:
                gradient, slope, offset = separator_plane
                if not (np.allclose(gradients[separator], gradient, atol=1e-10, rtol=0)
                        and np.allclose(raw[separator], np.asarray(missing) @ slope + offset,
                                        atol=1e-10, rtol=0)):
                    raise ValueError('unsupported_separator_geometry: canonical field must be affine with constant actual gradients')
            cache.update(zip(missing, range(len(cache), len(cache)+len(missing))))
            cached_raw.append(raw)
            cached_gradients.append(gradients)
        columns = np.fromiter((cache[k] for k in keys), dtype=np.int64, count=len(keys))
        if len(cached_raw) > 1:
            cached_raw[:] = [np.concatenate(cached_raw, axis=1)]
            cached_gradients[:] = [np.concatenate(cached_gradients, axis=1)]
        if not cached_raw:
            return np.empty((n, 0)), np.empty((n, 0, 3))
        return cached_raw[0][:, columns], cached_gradients[0][:, columns]

    field_scales = np.ptp(fields, axis=(1, 2))
    # Permit a few representable raw-value ULPs, not a relative raw-offset error.
    roundoff = 4*np.maximum(np.max(np.abs(np.spacing(samples)), axis=(1, 2)), np.abs(np.spacing(levels)))
    field_tolerances = 1e-10*(1+field_scales)+roundoff
    if field_query is not None and (separator_contacts is not None or ownership is not None):
        raise ValueError('invalid_field_query_contract: exclusive with separator contacts and ownership')
    queried = [0]

    def fields_batch(requests):
        """Validated batched targeted queries; empty requests are answered locally."""
        requests = [(np.asarray(points, dtype=float).reshape(-1, 3), list(surfaces), kind)
                    for points, surfaces, kind in requests]
        live = [i for i, (points, _, _) in enumerate(requests) if len(points)]
        answers = field_query([requests[i] for i in live]) if live else []
        if len(answers) != len(live):
            raise ValueError('invalid_callback_samples: one result per request required')
        results = []
        for points, surfaces, kind in requests:
            shape = (len(surfaces), len(points)) + ((3,) if kind == 'gradient' else ())
            results.append(np.empty(shape))
        for i, answer in zip(live, answers):
            points, surfaces, kind = requests[i]
            answer = np.array(detached(answer), dtype=float, copy=True)
            shape = (len(surfaces), len(points)) + ((3,) if kind == 'gradient' else ())
            if answer.shape != shape or not np.isfinite(answer).all():
                raise ValueError(f'invalid_callback_samples: finite requested-surface {kind}s required')
            results[i] = answer
            queried[0] += len(points)
        return results

    if field_query is not None:
        # Every tile and minimal-edge node is a leaf corner: reuse the supplied
        # corner values, after checking that leaves sharing a node agree.
        stride = domain.astype(np.int64)+1
        weights = np.array([stride[1]*stride[2], stride[2], 1])
        codes = (corners.reshape(-1, 3).astype(np.int64)*weights).sum(axis=1)
        unique, first, inverse = np.unique(codes, return_index=True, return_inverse=True)
        flat = fields.reshape(n, -1)
        low = np.full((n, len(unique)), np.inf)
        high = np.full((n, len(unique)), -np.inf)
        for s in range(n):
            np.minimum.at(low[s], inverse.reshape(-1), flat[s])
            np.maximum.at(high[s], inverse.reshape(-1), flat[s])
        if np.any(high-low > field_tolerances[:, None]) or np.any((low < 0) & (high > 0)):
            raise ValueError('inconsistent_corner_samples: leaves sharing a lattice node disagree')
        node_fields = flat[:, first]

        def lattice_fields(nodes):
            node_codes = (nodes.reshape(-1, 3).astype(np.int64)*weights).sum(axis=1)
            found = np.minimum(np.searchsorted(unique, node_codes), len(unique)-1)
            if np.any(unique[found] != node_codes):
                raise ValueError('unsupported_adaptive_node: tile or edge node is not a leaf corner')
            return node_fields[:, found]

        tile_fields = lattice_fields(tile_nodes).reshape(n, len(faces), 4)
        edge_fields = lattice_fields(edge_nodes).reshape(n, len(edges), 2)
        # Endpoint gradients orient ring triangles only on edges a surface
        # crosses; they are queried with the crossing normals in one batch.
        endpoint_gradients = np.zeros((n, len(edges), 2, 3))
    else:
        all_points = np.concatenate((xyz.reshape(-1, 3), tile_xyz.reshape(-1, 3), minimal_xyz.reshape(-1, 3)))
        query(all_points)
    if separator is not None:
        raw, gradients = query(all_points)
        gradient = gradients[separator, 0]
        slope = np.linalg.lstsq(all_points-all_points[0], raw[separator]-raw[separator, 0], rcond=None)[0]
        offset = raw[separator, 0] - all_points[0] @ slope
        if (np.linalg.norm(gradient) < 1e-12 or np.linalg.norm(slope) < 1e-12
                or not np.allclose(gradients[separator], gradient, atol=1e-10, rtol=0)
                or not np.allclose(raw[separator], all_points @ slope + offset, atol=1e-10, rtol=0)):
            raise ValueError('unsupported_separator_geometry: canonical field must be affine with constant actual gradients')
        if gradient @ slope < (1-1e-10)*np.linalg.norm(gradient)*np.linalg.norm(slope):
            raise ValueError('unsupported_separator_orientation: actual gradient must align with normalized affine field')
        separator_plane = gradient.copy(), slope, offset
    if field_query is None:
        canonical, _ = query(xyz)
        canonical_fields = canonical.reshape(samples.shape)-levels[:, None, None]
        if np.any(np.abs(canonical_fields-fields) > field_tolerances[:, None, None]):
            raise ValueError('inconsistent_corner_samples: callback disagrees with raw original corners')
        tile_fields = query(tile_xyz)[0].reshape(n, len(faces), 4)-levels[:, None, None]
        edge_fields, endpoint_gradients = query(minimal_xyz)
        edge_fields = edge_fields.reshape(n, len(edges), 2)-levels[:, None, None]
        endpoint_gradients = endpoint_gradients.reshape(n, len(edges), 2, 3)
    controllers = {s: [] for s in range(n)}
    for c, t in sorted(truncations):
        # Full isovalue extrema and ordinary onlap/erosion roles are unchanged.
        sign = 1 if identities[c][0] > identities[t][0] else -1
        controllers[t].append((c, sign))
    # Banked separators own one side; under fault_merge faults own nothing.
    for c, t in sorted(faults) if fault_merge is None else ():
        controllers[t].append((c, -1))
    explicit = separator_contacts['controllers'] if separator is not None else (
        ownership if isinstance(ownership, dict) else None)
    if explicit is not None:
        if not isinstance(explicit, dict) or any(
                not isinstance(t, (int, np.integer)) or isinstance(t, bool) or t not in controllers for t in explicit):
            raise ValueError('unsupported_ownership: invalid composite controller targets')
        try:
            normalized = {t: [tuple(row) for row in explicit.get(t, [])] for t in controllers}
            valid = all(all(len(row) == 2 and
                            isinstance(row[0], (int, np.integer)) and not isinstance(row[0], bool) and
                            isinstance(row[1], (int, np.integer)) and not isinstance(row[1], bool)
                            for row in normalized[t]) and len(normalized[t]) == len(controllers[t]) and
                        set(normalized[t]) == set(controllers[t]) for t in controllers)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError('unsupported_ownership: composite controllers must equal ordinary and authorized fault union')
        if separator is not None and ownership is not None:
            raise ValueError('unsupported_ownership: separator contract supplies composite ownership')
    elif ownership is not None:
        own = detached(ownership)
        if own.dtype.kind == 'b' or own.shape != samples.shape or not np.isfinite(own).all():
            raise ValueError('unsupported_ownership: finite signed (S,N,8) required')
        for t in range(n):
            if np.all(own[t] > 0):
                controllers[t] = []
                continue
            matches = [(c, sign) for c, target in sorted(truncations) if target == t
                       for sign in (-1, 1) if np.allclose(own[t], sign*fields[c], atol=1e-10, rtol=0)]
            if len(matches) != 1:
                raise ValueError('unsupported_ownership: require exact eligible signed controller')
            controllers[t] = matches
    pairs = {(c, t) for t, cs in controllers.items() for c, _ in cs if allowed[c, t]}
    stricts = (fields[:, :, EDGE_START]*fields[:, :, EDGE_END]) < 0
    # Leaves whose vertex an affected surface borrows from a fault (fault_merge).
    merged = np.zeros((n, leaves), dtype=bool)
    borrowed = {}
    for s, sources in merge_targets.items():
        crossed = stricts[s].any(axis=1)
        hits = np.array([stricts[c].any(axis=1) & crossed for c in sources])
        if np.any(hits.sum(axis=0) > 1):
            raise ValueError('unsupported_multi_fault_cell: one leaf borrows vertices from two faults')
        merged[s] = hits.any(axis=0)
        for c, row in zip(sources, hits):
            borrowed.update(((s, int(cell)), c) for cell in np.flatnonzero(row))
    flags, edge_xyz, counts = [], [], []
    for s in range(n):
        if separator is not None and np.any(np.abs(fields[s]) < 1e-10):
            raise ValueError('sample_aligned_interface: original corner lies on surface')
        crossings, production_flags = find_intersection_on_edge(
            tensor(xyz.reshape(-1, 3)), tensor(samples[s].reshape(-1)), tensor(levels[s:s+1]),
            **({'strict_crossings': True} if separator is not None or fault_merge is not None else {}))
        production_flags = detached(production_flags).reshape(leaves, 12).astype(bool)
        strict = stricts[s]
        kept = ~merged[s]
        if not np.array_equal(strict[kept], production_flags[kept]):
            raise ValueError('unsupported_geometric_flags: production tolerance crosses a non-strict edge')
        if merged[s].any():
            # Production crossings follow production flags; keep only strict edges.
            dense_production = np.zeros((leaves, 12, 3))
            dense_production[production_flags] = detached(crossings)
            crossings = dense_production[strict]
        _, components, _, _, _ = classify_cell_branches(fields[s], strict, skip=merged[s] if merged[s].any() else None)
        if np.any(components > 1):
            raise ValueError('unsupported_adaptive_branches: only original single-branch leaves supported')
        components[merged[s]] = 1
        dense = np.zeros((leaves, 12, 3))
        dense[strict] = detached(crossings)
        flags.append(strict)
        counts.append(components)
        edge_xyz.append(dense)
    flags, counts, edge_xyz = map(np.asarray, (flags, counts, edge_xyz))
    # A surface never solves its own QEF in borrowed leaves; there its Hermite
    # rows only constrain the fault vertex it borrows (pretty's weighted rule).
    solved = flags & ~merged[:, :, None]
    crossing_points = np.concatenate([edge_xyz[s][flags[s]] for s in range(n)])
    if len(crossing_points) and field_query is None:
        query(crossing_points)
    normals = np.zeros_like(edge_xyz)
    regular = {}
    weighted_fault_cells = set()

    leaf_origins = list(map(tuple, origins.astype(np.int64).tolist()))
    leaf_spans = spans.astype(np.int64).tolist()

    def regular_key(s, cell):
        return ('regular', (identities[s],), leaf_origins[cell], leaf_spans[cell], 0)

    if field_query is None:
        for s in range(n):
            if flags[s].any():
                normals[s][flags[s]] = query(edge_xyz[s][flags[s]])[1][s]
    else:
        crossed_edges = [np.flatnonzero(edge_fields[s, :, 0]*edge_fields[s, :, 1] < 0) for s in range(n)]
        answers = fields_batch([(edge_xyz[s][flags[s]], [s], 'gradient') for s in range(n)] +
                               [(minimal_xyz[crossed_edges[s]].reshape(-1, 3), [s], 'gradient') for s in range(n)])
        for s in range(n):
            normals[s][flags[s]] = answers[s][0]
            endpoint_gradients[s, crossed_edges[s]] = answers[n+s][0].reshape(-1, 2, 3)
    for s in range(n):
        if np.any(np.linalg.norm(normals[s][solved[s]], axis=1) < 1e-12):
            raise ValueError('degenerate_hermite_normal: actual crossing gradient vanishes')
    for s in range(n):
        own = solved[s]
        if not own.any():
            continue
        dc = DualContouringData(
            tensor(edge_xyz[s][own]), tensor(own, True), tensor(xyz.mean(axis=1)),
            tensor(spans[:, None]*spacing), 1, tensor(origins), gradients=tensor(normals[s][own]),
            strict_crossings=False)
        if fault_merge is not None and s in fault_merge:
            cells = np.flatnonzero(own.any(axis=1))
            blocks = [t for t in sorted(fault_merge[s]) if any(borrowed.get((t, int(c))) == s for c in cells)]
            if blocks:
                extra_xyz = np.zeros((len(cells), 12*len(blocks), 3))
                extra_normals = np.zeros_like(extra_xyz)
                extra_weights = np.zeros((len(cells), 12*len(blocks)))
                for b, t in enumerate(blocks):
                    rows = np.array([borrowed.get((t, int(c))) == s for c in cells])
                    span = slice(12*b, 12*(b+1))
                    extra_xyz[rows, span] = edge_xyz[t][cells[rows]]
                    extra_normals[rows, span] = normals[t][cells[rows]]
                    # Same rule as pretty: only rows with actual gradient data count.
                    extra_weights[rows, span] = DEFAULT_CROSS_SURFACE_WEIGHT*(
                        flags[t][cells[rows]] & np.any(normals[t][cells[rows]] != 0, axis=-1))
                dc.extra_edge_xyz = tensor(extra_xyz)
                dc.extra_edge_normals = tensor(extra_normals)
                dc.extra_weights = tensor(extra_weights)
                weighted_fault_cells.update((s, int(c)) for c in cells[np.any(extra_weights > 0, axis=1)])
        positions = detached(generate_dual_contouring_vertices(dc))
        if not np.isfinite(positions).all():
            raise ValueError('nonfinite_qef_solution: original production solve failed')
        for cell, position in zip(np.flatnonzero(own.any(axis=1)), positions):
            regular[regular_key(s, cell)] = position
    fallback = set()
    if borrowed:
        # Fallback region: borrowed leaves plus one ring of leaves sharing an
        # edge with them, so no kept seam's controller triangles reach a leaf
        # whose fields jump across the fault.
        neighbors = {}
        for edge in edges:
            ring = [c for c in edge['cells'] if c is not None]
            for c in ring:
                neighbors.setdefault(c, set()).update(ring)
        # One region for every surface: a junction given up there leaves both
        # participants, affected or not, with unshared vertices.
        core_cells = set(np.flatnonzero(merged.any(axis=0)).tolist())
        region = core_cells.union(*(neighbors.get(c, ()) for c in core_cells))
        overridden = {s: region for s in range(n)}
        junctions, evidence, fallback = plan_adaptive_junctions(
            faces, tile_xyz, tile_fields, pairs, identities, origins, spans, counts, edge_xyz, normals, flags,
            controllers=controllers, leaf_fields=fields, overridden=overridden)
    else:
        junctions, evidence = plan_adaptive_junctions(
            faces, tile_xyz, tile_fields, pairs, identities, origins, spans, counts, edge_xyz, normals, flags,
            controllers=controllers, leaf_fields=fields)

    fallback_leaves = ({(tuple(map(int, origins[c])), int(spans[c])) for c in overridden[0]}
                       if borrowed else set())
    # Contact junction candidates given up to the fallback: both participants cross.
    crossed_leaves = flags.any(axis=2)
    fallback = set(fallback).union(*(
        {c for c in overridden.get(a, set()) | overridden.get(b, set())
         if crossed_leaves[a, c] and crossed_leaves[b, c]} for a, b in pairs)) if borrowed else set()

    def vertex_key(s, cell):
        if cell in junctions and s in junctions[cell]['pair']:
            return junctions[cell]['key']
        source = borrowed.get((s, cell))
        return regular_key(s if source is None else source, cell)
    plans = [[] for _ in range(n)]
    fault_identities = {identities[c] for c in (fault_merge or {})}
    filtered_triangles = 0
    # Vectorised per-(edge, surface) decisions; errors keep the edge-major,
    # surface-minor order and per-pair check order of the original sweep.
    start_fields, end_fields = edge_fields[:, :, 0], edge_fields[:, :, 1]
    aligned = np.any(np.abs(edge_fields) < 1e-10, axis=2)
    crossing = ~aligned & (start_fields*end_fields < 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction = np.where(crossing, start_fields/(start_fields-end_fields), 0.)
    junction_aligned = np.zeros_like(crossing)
    retained = crossing.copy()
    for t in range(n):
        for c, sign in controllers[t]:
            decision = sign*((1-fraction[t])*start_fields[c]+fraction[t]*end_fields[c])
            junction_aligned[t] |= crossing[t] & (np.abs(decision) < 1e-10)
            retained[t] &= decision > 0
    edge_cells = np.array([[-1 if c is None else c for c in edge['cells']] for edge in edges],
                          dtype=np.int64).reshape(-1, 4)
    interior = np.all(edge_cells >= 0, axis=1)
    selected = retained & ~junction_aligned & interior[None, :]
    hanging = np.zeros_like(selected)
    for t in range(n):
        hanging[t] = selected[t] & np.any(counts[t][np.maximum(edge_cells, 0)] != 1, axis=1)
    # Minimal edges lying in an extent face (two exterior quadrants): their dual
    # faces close against the rim when a closed-volume build is requested.
    in_face = np.sum(edge_cells < 0, axis=1) == 2
    rim_rows = retained & ~junction_aligned & in_face[None, :] if closure else np.zeros_like(selected)
    for t in range(n):
        hanging[t] |= rim_rows[t] & np.any((counts[t][np.maximum(edge_cells, 0)] != 1) & (edge_cells >= 0), axis=1)
    errors = ((aligned, 'sample_aligned_interface: minimal edge endpoint lies on surface'),
              (junction_aligned, 'grid_edge_junction: geological ownership is aligned'),
              (hanging, 'unsupported_hanging_branch: crossing lacks original leaf branch'))
    failing = np.flatnonzero(np.any([mask.T for mask, _ in errors], axis=0).reshape(-1))
    if len(failing):
        e, t = divmod(int(failing[0]), n)
        raise ValueError(next(message for mask, message in errors if mask[t, e]))
    normals_at_crossing = ((1-fraction)[:, :, None]*endpoint_gradients[:, :, 0]
                           + fraction[:, :, None]*endpoint_gradients[:, :, 1])
    for s in range(n):
        for e in np.flatnonzero(selected[s]).tolist():
            cells = edges[e]['cells']
            regular_keys = collapse_ring([regular_key(s, c) for c in cells])
            keys = collapse_ring([vertex_key(s, c) for c in cells])
            row = dict(keys=keys, regular_keys=regular_keys, normal=normals_at_crossing[s, e], edge=e)
            if any((s, c) in borrowed for c in cells):
                # Production diagonals, minus triangles lying wholly on fault vertices.
                for triangle in ring_triangles(keys):
                    if all(k[1][0] in fault_identities for k in triangle):
                        filtered_triangles += 1
                    else:
                        plans[s].append(dict(row, keys=list(triangle)))
                continue
            plans[s].append(row)
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
    vertex_fields = {}
    joint_keys = [j['key'] for j in junctions.values()]
    if field_query is None:
        if used_regular:
            raw, _ = query([positions[k] for k in used_regular])
            vertex_fields = dict(zip(used_regular, (raw-levels[:, None]).T))
        if junctions:
            raw, _ = query([positions[k] for k in joint_keys])
            vertex_fields.update(zip(joint_keys, (raw-levels[:, None]).T))
    else:
        regular_raw, joint_raw = fields_batch([([positions[k] for k in used_regular], range(n), 'scalar'),
                                               ([positions[k] for k in joint_keys], range(n), 'scalar')])
        vertex_fields = dict(zip(used_regular, (regular_raw-levels[:, None]).T))
        vertex_fields.update(zip(joint_keys, (joint_raw-levels[:, None]).T))
    # Ownership is not certified for borrowed fault vertices nor in fallback
    # leaves, where pretty-style unshared vertices replace a joint junction.
    exempt = ([{k for row in rows for k in row['keys'] if k[0] == 'regular' and (
                   k[1][0] in fault_identities or (k[2], k[3]) in fallback_leaves)}
               for rows in triangles] if borrowed else None)
    validate_adaptive_geometry(triangles, seams, junctions, positions, vertex_fields, field_tolerances, controllers,
                               exempt=exempt)
    output_faces, affected = (None, None) if defer_emission else emit_adaptive_triangles(triangles, ids)
    result = dict(vertices=vertices, faces=output_faces, vertex_keys=keys,
                  seam_edges=np.asarray([[ids[a], ids[b]] for a, b in sorted(seams)], dtype=int).reshape(-1, 2),
                  junction_cells=np.asarray(sorted(junctions), dtype=int), leaf_origins=origins.copy(),
                  leaf_spans=spans.copy(), affected_faces=affected,
                  diagnostics=dict(algorithm='adaptive_joint_hermite_qef_dual_contouring',
                                   coarsefine_seam_edge_count=len(coarsefine), no_uniform_fallback=True,
                                   no_post_reconciliation=True, original_hermite_rows=12,
                                   sampled_point_count=len(cache)+queried[0], canonical_face_tile_count=len(faces),
                                    minimal_edge_count=len(edges), finite_faults='unsupported',
                                    interior_topology='sampled_boundary_only_not_interior_certified'))
    if defer_emission:
        result.update(_triangle_plan=triangles, _key_to_id=ids)
    if closure:
        boundary = np.flatnonzero(np.any(origins == 0, axis=1) | np.any(origins+spans[:, None] == domain, axis=1))
        boundary_keys = {}
        for s in range(n):
            for cell in boundary.tolist():
                key = vertex_key(s, cell)
                if key in positions:
                    boundary_keys[(s, cell)] = key
        rows = []
        for s in range(n):
            for e in np.flatnonzero(rim_rows[s]).tolist():
                cells = [c for c in edges[e]['cells'] if c is not None]
                rows.append(dict(surface=s, edge=e, keys=tuple(vertex_key(s, c) for c in cells)))
        result['closure'] = dict(rows=rows, boundary_keys=boundary_keys, triangles=triangles, positions=positions,
                                 edges=edges, identities=identities, levels=levels, domain=domain,
                                 bounds=bounds, spacing=spacing, origins=origins, spans=spans)
    if include_reference:
        reference_keys = sorted(regular)
        reference_vertices = np.asarray([regular[k] for k in reference_keys]).reshape(-1, 3)
        reference_plans = [[dict(row, keys=row['regular_keys']) for row in rows] for rows in plans]
        reference_faces, _ = emit_adaptive_triangles(
            plan_adaptive_triangles(reference_plans, regular), {k: i for i, k in enumerate(reference_keys)})
        result['reference'] = dict(vertices=reference_vertices, faces=reference_faces, vertex_keys=reference_keys,
                                    description='Original production QEFs with identical retained minimal edges')
    if fault_merge is not None:
        result['diagnostics'].update(
            contact_type='pretty_fault_vertex_merge', strict_crossings=True,
            crossing_contract='strict_scalar_sides', fault_surfaces=sorted(int(c) for c in fault_merge),
            fault_cell_count={int(c): int(flags[c].any(axis=1).sum()) for c in fault_merge},
            borrowed_vertex_count=len(borrowed), filtered_fault_triangle_count=filtered_triangles,
            weighted_fault_vertex_count=len(weighted_fault_cells),
            cross_surface_weight=DEFAULT_CROSS_SURFACE_WEIGHT,
            fault_overridden_junction_cells=sorted(int(c) for c in fallback),
            fallback_region_leaf_count=len(fallback_leaves))
    if separator is not None:
        result['bank'] = int(bank)
        result['diagnostics'].update(bank=int(bank), separator_contact=int(separator),
                                     contact_type='explicit_fault_separator', separator_geometry='planar',
                                     strict_crossings=True, crossing_contract='strict_scalar_sides')
    return result
