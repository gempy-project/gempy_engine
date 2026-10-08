"""Explicit leaf-native joint dual contouring; no production dispatch or repairs."""

import numpy as np

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as BT
from gempy_engine.core.data.dual_contouring_data import DualContouringData
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine.modules.dual_contouring._gen_vertices import generate_dual_contouring_vertices
from gempy_engine.modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge
from gempy_engine.modules.dual_contouring.contact_topology import build_contact_relations
from gempy_engine.modules.dual_contouring.topology_extraction import CORNERS, EDGE_START, EDGE_END, classify_cell_branches
from gempy_engine.modules.dual_contouring.adaptive_joint_topology import (
    plan_adaptive_junctions, collapse_ring, validate_adaptive_incidence, plan_adaptive_triangles,
    validate_adaptive_geometry, emit_adaptive_triangles,
)


def extract_adaptive_topology(origins, spans, domain_shape, extent, scalar_samples,
                              surface_to_stack, surface_indices, stack_relations, isovalues,
                              *, sample_fields, ownership=None, include_reference=False,
                              faults_relations=None, separator_contacts=None, defer_emission=False):
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
    """
    from gempy_engine.modules.dual_contouring.adaptive_cell_complex import build_adaptive_complex

    if defer_emission and include_reference:
        raise ValueError('unsupported_deferred_reference: defer_emission and include_reference are incompatible')
    if BT.use_gpu or BT.dtype != 'float64' or BT.engine_backend not in (
            AvailableBackends.numpy, AvailableBackends.PYTORCH):
        raise ValueError('unsupported_backend: NumPy/Torch CPU float64 only')

    def detached(a):
        return a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a)

    def tensor(a, boolean=False):
        if BT.engine_backend is AvailableBackends.PYTORCH:
            import torch
            return torch.as_tensor(a, dtype=torch.bool if boolean else torch.float64, device='cpu')
        return np.asarray(a, dtype=bool if boolean else np.float64)

    allowed, faults, truncations = build_contact_relations(
        surface_to_stack, surface_indices, stack_relations, faults_relations, isovalues)
    if separator_contacts is None and (faults or any(r is R.FAULT for r in stack_relations)
                                       or (faults_relations is not None and np.asarray(faults_relations).any())):
        raise ValueError('unsupported_fault_extraction: adaptive bank incidence unavailable')
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
    complex_ = build_adaptive_complex(origins, spans, domain_shape)
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
    tile_nodes = []
    for face in faces:
        other = [a for a in range(3) if a != face['axis']]
        nodes = np.tile(face['origin'], (4, 1))
        nodes[:, other] += np.array([[0, 0], [1, 0], [0, 1], [1, 1]])*face['span']
        tile_nodes.append(nodes)
    tile_xyz = bounds[::2]+np.asarray(tile_nodes)*spacing
    edge_nodes = []
    for edge in edges:
        nodes = np.tile(edge['origin'], (2, 1))
        nodes[1, edge['axis']] += edge['span']
        edge_nodes.append(nodes)
    minimal_xyz = bounds[::2]+np.asarray(edge_nodes)*spacing
    cache = {}
    separator_plane = None

    def query(points):
        points = np.asarray(points).reshape(-1, 3)
        keys = [tuple(p) for p in points]
        missing = sorted(set(keys)-cache.keys())
        if missing:
            raw, gradients = sample_fields(np.asarray(missing))
            raw, gradients = detached(raw), detached(gradients)
            if raw.shape != (n, len(missing)) or gradients.shape != (n, len(missing), 3) or not (
                    np.isfinite(raw).all() and np.isfinite(gradients).all()):
                raise ValueError('invalid_callback_samples: finite all-surface scalar/gradient arrays required')
            if separator_plane is not None:
                gradient, slope, offset = separator_plane
                if not (np.allclose(gradients[separator], gradient, atol=1e-10, rtol=0)
                        and np.allclose(raw[separator], np.asarray(missing) @ slope + offset,
                                        atol=1e-10, rtol=0)):
                    raise ValueError('unsupported_separator_geometry: canonical field must be affine with constant actual gradients')
            for i, key in enumerate(missing):
                cache[key] = (raw[:, i].copy(), gradients[:, i].copy())
        return (np.stack([cache[k][0] for k in keys], axis=1),
                np.stack([cache[k][1] for k in keys], axis=1))

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
    canonical, _ = query(xyz)
    canonical_fields = canonical.reshape(samples.shape)-levels[:, None, None]
    field_scales = np.ptp(fields, axis=(1, 2))
    # Permit a few representable raw-value ULPs, not a relative raw-offset error.
    roundoff = 4*np.maximum(np.max(np.abs(np.spacing(samples)), axis=(1, 2)), np.abs(np.spacing(levels)))
    field_tolerances = 1e-10*(1+field_scales)+roundoff
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
    for c, t in sorted(faults):
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
    flags, edge_xyz, counts = [], [], []
    for s in range(n):
        if separator is not None and np.any(np.abs(fields[s]) < 1e-10):
            raise ValueError('sample_aligned_interface: original corner lies on surface')
        crossings, production_flags = find_intersection_on_edge(
            tensor(xyz.reshape(-1, 3)), tensor(samples[s].reshape(-1)), tensor(levels[s:s+1]),
            **({'strict_crossings': True} if separator is not None else {}))
        production_flags = detached(production_flags).reshape(leaves, 12).astype(bool)
        strict = (fields[s][:, EDGE_START]*fields[s][:, EDGE_END]) < 0
        if not np.array_equal(strict, production_flags):
            raise ValueError('unsupported_geometric_flags: production tolerance crosses a non-strict edge')
        _, components, _, _, _ = classify_cell_branches(fields[s], strict)
        if np.any(components > 1):
            raise ValueError('unsupported_adaptive_branches: only original single-branch leaves supported')
        dense = np.zeros((leaves, 12, 3))
        dense[strict] = detached(crossings)
        flags.append(strict)
        counts.append(components)
        edge_xyz.append(dense)
    flags, counts, edge_xyz = map(np.asarray, (flags, counts, edge_xyz))
    crossing_points = np.concatenate([edge_xyz[s][flags[s]] for s in range(n)])
    if len(crossing_points):
        query(crossing_points)
    normals = np.zeros_like(edge_xyz)
    regular = {}

    def regular_key(s, cell):
        return ('regular', (identities[s],), tuple(map(int, origins[cell])), int(spans[cell]), 0)

    for s in range(n):
        if not flags[s].any():
            continue
        normals[s][flags[s]] = query(edge_xyz[s][flags[s]])[1][s]
        if np.any(np.linalg.norm(normals[s][flags[s]], axis=1) < 1e-12):
            raise ValueError('degenerate_hermite_normal: actual crossing gradient vanishes')
        dc = DualContouringData(
            tensor(edge_xyz[s][flags[s]]), tensor(flags[s], True), tensor(xyz.mean(axis=1)),
            tensor(spans[:, None]*spacing), 1, tensor(origins), gradients=tensor(normals[s][flags[s]]),
            strict_crossings=False)
        positions = detached(generate_dual_contouring_vertices(dc))
        if not np.isfinite(positions).all():
            raise ValueError('nonfinite_qef_solution: original production solve failed')
        for cell, position in zip(np.flatnonzero(flags[s].any(axis=1)), positions):
            regular[regular_key(s, cell)] = position
    junctions, evidence = plan_adaptive_junctions(
        faces, tile_xyz, tile_fields, pairs, identities, origins, spans, counts, edge_xyz, normals, flags,
        controllers=controllers, leaf_fields=fields)
    plans = [[] for _ in range(n)]
    for e, edge in enumerate(edges):
        for s in range(n):
            f = edge_fields[s, e]
            if np.any(np.abs(f) < 1e-10):
                raise ValueError('sample_aligned_interface: minimal edge endpoint lies on surface')
            if f[0]*f[1] >= 0:
                continue
            fraction = f[0]/(f[0]-f[1])
            retained = True
            for c, sign in controllers[s]:
                decision = sign*((1-fraction)*edge_fields[c, e, 0]+fraction*edge_fields[c, e, 1])
                if abs(decision) < 1e-10:
                    raise ValueError('grid_edge_junction: geological ownership is aligned')
                retained &= decision > 0
            if not retained or any(c is None for c in edge['cells']):
                continue
            cells = edge['cells']
            if any(counts[s, c] != 1 for c in cells):
                raise ValueError('unsupported_hanging_branch: crossing lacks original leaf branch')
            regular_keys = collapse_ring([regular_key(s, c) for c in cells])
            keys = collapse_ring([junctions[c]['key'] if c in junctions and s in junctions[c]['pair']
                                  else regular_key(s, c) for c in cells])
            normal = ((1-fraction)*endpoint_gradients[s, e, 0]+fraction*endpoint_gradients[s, e, 1])
            plans[s].append(dict(keys=keys, regular_keys=regular_keys, normal=normal, edge=e))
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
    if used_regular:
        raw, _ = query([positions[k] for k in used_regular])
        vertex_fields = dict(zip(used_regular, (raw-levels[:, None]).T))
    if junctions:
        joint_keys = [j['key'] for j in junctions.values()]
        raw, _ = query([positions[k] for k in joint_keys])
        vertex_fields.update(zip(joint_keys, (raw-levels[:, None]).T))
    validate_adaptive_geometry(triangles, seams, junctions, positions, vertex_fields, field_tolerances, controllers)
    output_faces, affected = (None, None) if defer_emission else emit_adaptive_triangles(triangles, ids)
    result = dict(vertices=vertices, faces=output_faces, vertex_keys=keys,
                  seam_edges=np.asarray([[ids[a], ids[b]] for a, b in sorted(seams)], dtype=int).reshape(-1, 2),
                  junction_cells=np.asarray(sorted(junctions), dtype=int), leaf_origins=origins.copy(),
                  leaf_spans=spans.copy(), affected_faces=affected,
                  diagnostics=dict(algorithm='adaptive_joint_hermite_qef_dual_contouring',
                                   coarsefine_seam_edge_count=len(coarsefine), no_uniform_fallback=True,
                                   no_post_reconciliation=True, original_hermite_rows=12,
                                   sampled_point_count=len(cache), canonical_face_tile_count=len(faces),
                                    minimal_edge_count=len(edges), finite_faults='unsupported',
                                    interior_topology='sampled_boundary_only_not_interior_certified'))
    if defer_emission:
        result.update(_triangle_plan=triangles, _key_to_id=ids)
    if include_reference:
        reference_keys = sorted(regular)
        reference_vertices = np.asarray([regular[k] for k in reference_keys]).reshape(-1, 3)
        reference_plans = [[dict(row, keys=row['regular_keys']) for row in rows] for rows in plans]
        reference_faces, _ = emit_adaptive_triangles(
            plan_adaptive_triangles(reference_plans, regular), {k: i for i, k in enumerate(reference_keys)})
        result['reference'] = dict(vertices=reference_vertices, faces=reference_faces, vertex_keys=reference_keys,
                                    description='Original production QEFs with identical retained minimal edges')
    if separator is not None:
        result['bank'] = int(bank)
        result['diagnostics'].update(bank=int(bank), separator_contact=int(separator),
                                     contact_type='explicit_fault_separator', separator_geometry='planar',
                                     strict_crossings=True, crossing_contract='strict_scalar_sides')
    return result
