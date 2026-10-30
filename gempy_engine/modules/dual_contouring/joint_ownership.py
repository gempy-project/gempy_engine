"""Geological ownership for joint extraction: controllers, fault-vertex borrowing, fallback regions.

A controller ``(c, sign)`` of surface ``t`` keeps ``t`` where ``sign * field_c > 0``
(erosion/onlap truncations, from ``build_contact_relations``). Under the
pretty-style fault merge, affected surfaces borrow the fault's vertex in leaves
both cross; contacts there and in the adjacent leaf ring fall back to unshared
vertices. Surfaces are indices into the exported surface order.
"""

import numpy as np

from .joint_xp import host, namespace
from ...core.data.stack_relation_type import StackRelationType as R


# region Controllers

def check_missing_controllers(full_identities, full_truncations, surface_to_stack, surface_indices, isovalues):
    """Reject a represented target whose ordinary controller boundary is not represented.

    ``full_truncations`` come from every known boundary (``full_identities``),
    so an omitted controller cannot silently become unrestricted ownership.
    """
    represented_stacks = set(surface_to_stack)
    represented = {(g, float(isovalues[g][s])) for g, s in zip(surface_to_stack, surface_indices)}
    for c, t in full_truncations:
        controller_stack, index = full_identities[c]
        if (full_identities[t][0] in represented_stacks
                and (controller_stack, float(isovalues[controller_stack][index])) not in represented):
            raise ValueError('unsupported_missing_controller: represented target requires known ordinary boundary '
                             f'{controller_stack}/{index}')


def ordinary_controllers(identities, truncations):
    """Per target, ``(controller, sign)``: younger stacks own their positive side."""
    controllers = {s: [] for s in range(len(identities))}
    for c, t in sorted(truncations):
        # Full isovalue extrema and ordinary onlap/erosion roles are unchanged.
        controllers[t].append((c, 1 if identities[c][0] > identities[t][0] else -1))
    return controllers


def check_composite_controllers(explicit, controllers):
    """A composite controller dict must equal the derived controllers exactly."""
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


def controllers_from_ownership(ownership, fields, truncations):
    """Controllers from signed synthetic ownership (S, N, 8), which must equal ± one eligible field."""
    if ownership.dtype.kind == 'b' or ownership.shape != fields.shape or not np.isfinite(ownership).all():
        raise ValueError('unsupported_ownership: finite signed (S,N,8) required')
    controllers = {}
    for t in range(len(fields)):
        if np.all(ownership[t] > 0):
            controllers[t] = []
            continue
        matches = [(c, sign) for c, target in sorted(truncations) if target == t
                   for sign in (-1, 1) if np.allclose(ownership[t], sign*fields[c], atol=1e-10, rtol=0)]
        if len(matches) != 1:
            raise ValueError('unsupported_ownership: require exact eligible signed controller')
        controllers[t] = matches
    return controllers


def contact_pairs(controllers, allowed):
    """Directed (controller, target) pairs that may share a joint vertex."""
    return {(c, t) for t, cs in controllers.items() for c, _ in cs if allowed[c, t]}

# endregion


# region Fault merge

def fault_merge_targets(fault_merge, surface_to_stack, stack_relations, fault_pairs):
    """Validate ``fault_merge`` (fault surface -> affected surfaces); return target -> faults."""
    fault_surfaces = {i for i, g in enumerate(surface_to_stack) if stack_relations[g] is R.FAULT}
    expected = {i: set() for i in fault_surfaces}
    for c, t in fault_pairs:
        expected.setdefault(c, set()).add(t)
    try:
        supplied = {int(k): set(map(int, v)) for k, v in fault_merge.items()}
    except (AttributeError, TypeError, ValueError):
        supplied = None
    if supplied != expected or set(expected) != fault_surfaces:
        raise ValueError('invalid_fault_merge_contract: must map every fault surface to exactly its affected surfaces')
    targets = {}
    for c, affected in supplied.items():
        for t in affected:
            targets.setdefault(t, []).append(c)
    return targets


def borrowed_leaves(merge_targets, stricts):
    """Leaves where an affected surface borrows a fault's vertex.

    ``stricts`` (S, N, 12) are strict edge crossings. Returns ``merged`` and
    ``multi_fault`` (S, N) masks and ``source`` (S, N), the fault surface whose
    vertex each (surface, leaf) borrows, or -1. A leaf crossed by the surface
    and two or more of its faults has no unambiguous borrow: the surface keeps
    its own vertex there (pretty-style, counted).
    """
    xp = namespace(stricts)
    merged = xp.zeros(tuple(stricts.shape[:2]), dtype=bool)
    multi_fault = xp.zeros(tuple(stricts.shape[:2]), dtype=bool)
    source = xp.full(tuple(stricts.shape[:2]), -1, dtype=np.int64)
    for s, sources in merge_targets.items():
        crossed = stricts[s].any(axis=1)
        hits = xp.stack([stricts[c].any(axis=1) & crossed for c in sources])
        several = hits.sum(axis=0) > 1
        multi_fault[s] = several
        hits &= ~several
        merged[s] = hits.any(axis=0)
        # At most one fault per leaf is left: its index, -1 elsewhere.
        source[s] = xp.where(merged[s], sum(c*xp.astype(row, np.int64) for c, row in zip(sources, hits)), -1)
    return merged, multi_fault, source


def borrowed_hermite_rows(fault, cells, affected, source, edge_xyz, normals, flags, weight):
    """Extra QEF rows for a fault's vertices from the surfaces that borrow them.

    Same rule as pretty: only rows with actual gradient data count, weighted by
    ``weight``. ``source`` (S, N) is ``borrowed_leaves``' borrowed fault per
    leaf; ``edge_xyz[t][leaves]``/``normals[t][leaves]`` give (L, 12, 3) rows.
    Returns ``None`` when no affected surface borrows in ``cells``, else
    ``(xyz, normals, weights)`` of shapes (C, 12B, 3), (C, 12B, 3), (C, 12B).
    """
    xp = namespace(source, cells)
    affected = sorted(affected)
    if not affected:
        return None
    # Which cells each affected surface borrows the fault's vertex in; one host read for the blocks.
    hits = xp.stack([source[t, cells] == fault for t in affected])
    present = host(hits.any(axis=1))
    if not present.any():
        return None
    # Every borrowing surface (block) at once: (B, C, 12[, 3]) rows, zero where the surface does not borrow.
    blocks = np.flatnonzero(present)
    surfaces = xp.asarray(np.asarray(affected, dtype=np.int64)[blocks])[:, None]
    borrows = hits[xp.asarray(blocks)] if len(blocks) < len(affected) else hits
    borrowed_xyz, borrowed_normals = (rows.gather(surfaces, cells) if hasattr(rows, 'gather')
                                      else xp.asarray(rows)[surfaces, cells] for rows in (edge_xyz, normals))
    weights = weight*xp.astype(flags[surfaces, cells] & (borrowed_normals != 0).any(axis=-1), np.float64)

    def columns(values):  # (B, C, 12, ...) -> (C, 12 B, ...): block b in columns 12 b to 12 b + 11
        values = xp.where(borrows.reshape(borrows.shape + (1,) * (values.ndim - 2)), values, 0.)
        return values.swapaxes(0, 1).reshape((len(cells), 12*len(blocks)) + tuple(values.shape[3:]))
    return columns(borrowed_xyz), columns(borrowed_normals), columns(weights)


def fallback_region(edge_cells, core_leaves):
    """(N,) mask: ``core_leaves`` plus one ring of leaves sharing an edge with them.

    ``edge_cells`` (E, 4) are the minimal-edge leaf rings (-1 outside). No kept
    seam's controller triangles then reach a leaf whose fields jump across the fault.
    """
    xp = namespace(edge_cells, core_leaves)
    core = xp.asarray(core_leaves, dtype=bool)
    present = edge_cells >= 0
    touched = (present & core[xp.maximum(edge_cells, 0)]).any(axis=1)
    region = xp.copy(core)
    ring = edge_cells[touched]
    region[ring[ring >= 0]] = True
    return region


def fallback_junction_leaves(planned_fallback, region, crossed_leaves, pairs):
    """Contact junction candidates given up to the fallback: both participants cross.

    ``region`` is the (N,) fallback-region mask; returns a set of leaves.
    """
    xp = namespace(region, crossed_leaves)
    given_up = xp.zeros(tuple(region.shape), dtype=bool)
    for a, b in pairs:
        given_up |= region & crossed_leaves[a] & crossed_leaves[b]
    return set(planned_fallback).union(xp.flatnonzero(given_up).tolist())

# endregion
