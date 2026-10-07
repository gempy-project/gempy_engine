"""Detached, cell-local contact positions and identities for extracted meshes."""

from itertools import combinations

import numpy as np


def reconcile_cell_vertices(
        vertices, cell_coordinates, surface_to_stack, surface_ids, allowed_pairs,
        fault_pairs=(), *, contact_eligible=None
):
    """Return copied vertices, int64 contact IDs, and cell-contact diagnostics.

    ``allowed_pairs`` is a symmetric boolean surface-index graph for nonfault
    contacts; ``fault_pairs`` contains directed controller/target surface indices.
    Each surface has at most one vertex per integer cell. Stable surface IDs are
    (stack_index, surface_index) pairs, independent of input list ordering.
    ``contact_eligible`` optionally supplies boolean per-vertex ownership masks
    for nonfault contacts only; None treats every vertex as eligible.

    Candidates are ordered by original squared distance, then stable IDs. Sets
    never contain two members of one stack. Nonfault sets additionally require
    every cross-set pair to be allowed. Fault sets have one original-position
    controller and cannot participate in nonfault merges or fault chains.

    Contact IDs enumerate shared sets by cell coordinates and stable member IDs.
    Diagnostics count rejected candidate edges (not redundant accepted edges);
    fault_overlap_vertices includes every directed shared-cell target, even if
    its position assignment was rejected. No connectivity is changed here.
    """
    n_surfaces = len(vertices)
    if not (len(cell_coordinates) == len(surface_to_stack) == len(surface_ids) == n_surfaces):
        raise ValueError("Surface metadata lengths must match vertices")
    if contact_eligible is not None and len(contact_eligible) != n_surfaces:
        raise ValueError("contact_eligible must contain one mask per surface")
    integer = lambda value: isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
    if any(not integer(group) for group in surface_to_stack):
        raise ValueError("surface_to_stack must contain integer group identities")
    if any(not isinstance(sid, tuple) or len(sid) != 2 or
           any(not integer(value) for value in sid) for sid in surface_ids):
        raise ValueError("surface_ids must contain integer (stack, surface) tuples")
    if len(set(surface_ids)) != n_surfaces:
        raise ValueError("surface_ids must be unique")
    allowed = np.asarray(allowed_pairs)
    if allowed.shape != (n_surfaces, n_surfaces) or allowed.dtype != np.bool_:
        raise ValueError("allowed_pairs must be an NxN boolean array")
    if not np.array_equal(allowed, allowed.T):
        raise ValueError("allowed_pairs must be symmetric")
    directed = set()
    for pair in fault_pairs:
        if (not isinstance(pair, (tuple, list, np.ndarray)) or len(pair) != 2 or
                any(not integer(index) or not 0 <= index < n_surfaces for index in pair) or
                pair[0] == pair[1]):
            raise ValueError("fault_pairs must contain distinct valid surface indices")
        directed.add(tuple(pair))

    originals = []
    new_vertices = []
    contact_ids = []
    eligibility = []
    buckets = {}
    for surface, (positions, coordinates) in enumerate(zip(vertices, cell_coordinates)):
        positions = np.asarray(positions)
        coordinates = np.asarray(coordinates)
        if (positions.ndim != 2 or positions.shape[1] != 3 or
                not np.issubdtype(positions.dtype, np.floating) or
                not np.isfinite(positions).all()):
            raise ValueError("vertices must be finite floating arrays of shape (Nv, 3)")
        if originals and positions.dtype != originals[0].dtype:
            raise ValueError("Contact vertices must use a common dtype for identical shared positions")
        if coordinates.shape != positions.shape or not np.issubdtype(coordinates.dtype, np.integer):
            raise ValueError("cell_coordinates must be integer arrays matching vertices")
        mask = (np.ones(len(positions), dtype=bool) if contact_eligible is None
                else np.asarray(contact_eligible[surface]))
        if mask.shape != (len(positions),) or mask.dtype != np.bool_:
            raise ValueError("contact_eligible masks must be boolean arrays of length Nv")
        eligibility.append(mask)
        seen = set()
        for row, coordinate in enumerate(coordinates):
            cell = tuple(int(value) for value in coordinate)
            if cell in seen:
                raise ValueError("Duplicate cell coordinates within a surface")
            seen.add(cell)
            buckets.setdefault(cell, []).append((surface, row))
        originals.append(positions)
        new_vertices.append(positions.copy())
        contact_ids.append(np.full(len(positions), -1, dtype=np.int64))

    report = dict(contact_count=0, conflict_count=0, fault_conflict_count=0,
                  nonfault_conflict_count=0, conflicts=[])
    fault_overlap = [set() for _ in vertices]
    shared_sets = []
    for cell, entries in sorted(buckets.items()):
        if len(entries) < 2:
            continue
        rows = dict(entries)
        surfaces = sorted(rows, key=lambda surface: surface_ids[surface])
        owners = {surface: surface for surface in surfaces}
        members = {surface: {surface} for surface in surfaces}
        anchors = {}
        fault_candidates = []
        nonfault_candidates = []
        for left, right in combinations(surfaces, 2):
            # Compute from original positions, never from a previously shared mean.
            with np.errstate(over='ignore'):
                delta = (originals[left][rows[left]].astype(np.float64) -
                         originals[right][rows[right]].astype(np.float64))
                distance = float(np.dot(delta, delta))
            is_fault = False
            for controller, target in ((left, right), (right, left)):
                if (controller, target) in directed:
                    is_fault = True
                    fault_overlap[target].add(rows[target])
                    fault_candidates.append((distance, surface_ids[controller],
                                             surface_ids[target], controller, target))
            if (not is_fault and allowed[left, right] and
                    eligibility[left][rows[left]] and eligibility[right][rows[right]]):
                nonfault_candidates.append((distance, surface_ids[left],
                                            surface_ids[right], left, right))

        for kind, candidates in (("fault", fault_candidates), ("nonfault", nonfault_candidates)):
            for _, _, _, left, right in sorted(candidates):
                left_owner, right_owner = owners[left], owners[right]
                reason = None
                if kind == "fault":
                    if left_owner in anchors and anchors[left_owner] != left:
                        reason = "controller_already_anchored"
                    elif right_owner in anchors and anchors[right_owner] != left:
                        reason = "competing_controller"
                if left_owner == right_owner:
                    if reason is None:
                        continue
                elif kind == "nonfault" and (left_owner in anchors or right_owner in anchors):
                    reason = "fault_anchored"
                if reason is None and left_owner != right_owner:
                    left_groups = {surface_to_stack[s] for s in members[left_owner]}
                    right_groups = {surface_to_stack[s] for s in members[right_owner]}
                    if not left_groups.isdisjoint(right_groups):
                        reason = "same_group"
                    elif kind == "nonfault" and not all(
                            allowed[a, b] for a in members[left_owner] for b in members[right_owner]):
                        reason = "disallowed_cross_pair"
                if reason is not None:
                    report["conflict_count"] += 1
                    report[f"{kind}_conflict_count"] += 1
                    report["conflicts"].append(dict(cell=cell, kind=kind, reason=reason,
                                                    surface_ids=(surface_ids[left], surface_ids[right])))
                    continue
                for surface in members[right_owner]:
                    owners[surface] = left_owner
                members[left_owner].update(members.pop(right_owner))
                if kind == "fault":
                    anchors[left_owner] = left

        for owner, group in members.items():
            if len(group) < 2:
                continue
            ordered = sorted(group, key=lambda surface: surface_ids[surface])
            if owner in anchors:
                controller = anchors[owner]
                position = originals[controller][rows[controller]]
            else:
                positions = np.asarray([originals[s][rows[s]] for s in ordered], dtype=np.float64)
                # Divide first to keep the mean finite even for large finite inputs.
                position = np.sum(positions / len(ordered), axis=0)
            shared_sets.append((cell, tuple(surface_ids[s] for s in ordered),
                                [(s, rows[s]) for s in ordered], position))

    for contact_id, (_, _, entries, position) in enumerate(sorted(shared_sets, key=lambda item: item[:2])):
        for surface, row in entries:
            new_vertices[surface][row] = position
            contact_ids[surface][row] = contact_id
    report["contact_count"] = len(shared_sets)
    report["fault_overlap_vertices"] = [np.array(sorted(rows), dtype=np.int64) for rows in fault_overlap]
    return new_vertices, contact_ids, report


def finalize_cell_vertices(originals, vertices, contact_ids, faces, surface_ids, fault_pairs=()):
    """Prune ordinary contact memberships unsupported by retained triangles.

    Keep provisional groups and surviving IDs: topology has already used them
    to remove redundant patches. Never regroup competing horizons. Fault groups
    retain their directional assignments, including discarded overlap targets.
    """
    positions = [array.copy() for array in vertices]
    ids = [array.copy() for array in contact_ids]
    groups = {}
    active = []
    for surface, (shared, triangles) in enumerate(zip(ids, faces)):
        supported = np.zeros(len(shared), dtype=bool)
        supported[np.asarray(triangles).ravel()] = True
        active.append(supported)
        for row in np.flatnonzero(shared >= 0):
            groups.setdefault(int(shared[row]), []).append((surface, row))
    report = dict(provisional_contact_count=len(groups), contact_count=0,
                  unsupported_contact_member_count=0, dissolved_contact_count=0)
    for members in groups.values():
        surfaces = {surface for surface, _ in members}
        if any(controller in surfaces and target in surfaces for controller, target in fault_pairs):
            report['contact_count'] += 1
            continue
        retained = [(surface, row) for surface, row in members if active[surface][row]]
        report['unsupported_contact_member_count'] += len(members) - len(retained)
        if len(retained) == len(members):
            report['contact_count'] += 1
            continue
        for surface, row in members:
            if not active[surface][row] or len(retained) < 2:
                positions[surface][row] = originals[surface][row]
                ids[surface][row] = -1
        if len(retained) < 2:
            report['dissolved_contact_count'] += 1
            continue
        retained.sort(key=lambda member: surface_ids[member[0]])
        original_positions = np.asarray([originals[surface][row] for surface, row in retained], dtype=np.float64)
        mean = np.sum(original_positions / len(retained), axis=0)
        for surface, row in retained:
            positions[surface][row] = mean
        report['contact_count'] += 1
    return positions, ids, report


def attach_fault_junctions(originals, vertices, contact_ids, faces, cell_coordinates,
                           surface_ids, allowed_pairs, fault_pairs, truncation_pairs):
    """Attach supported ordinary seam endpoints without moving fault anchors.

    Only previously unshared boundary rows can join an anchored contact. A
    matching retained controller edge is required; snapshot IDs prevent new
    attachments from becoming evidence for transitive junction propagation.
    Fault relations, overlap-removal metadata and triangle indices stay intact.
    """
    positions = [array.copy() for array in vertices]
    ids = [array.copy() for array in contact_ids]
    report = dict(fault_junction_attachment_count=0, fault_junction_rejected_count=0,
                  fault_junction_attachments=[], fault_junction_rejections=[])
    if not fault_pairs or not truncation_pairs:
        return positions, ids, report
    fault_controllers = {controller for controller, _ in fault_pairs}
    fault_participants = fault_controllers | {target for _, target in fault_pairs}
    groups, lookups, edges, boundaries, patches = {}, [], [], [], []
    for surface, (shared, triangles, cells) in enumerate(zip(contact_ids, faces, cell_coordinates)):
        lookups.append({tuple(cell): row for row, cell in enumerate(cells)})
        for row in np.flatnonzero(shared >= 0):
            groups.setdefault(int(shared[row]), set()).add(surface)
        unique, counts = np.unique(np.sort(np.concatenate([
            triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]],
        ]), axis=1), axis=0, return_counts=True)
        edges.append({tuple(edge) for edge in unique})
        neighbors = {}
        for a, b in unique[counts == 1]:
            neighbors.setdefault(int(a), set()).add(int(b))
            neighbors.setdefault(int(b), set()).add(int(a))
        boundaries.append(neighbors)
        keys = np.sort(shared[triangles], axis=1)
        patches.append({tuple(key) for key in keys if key[0] >= 0 and key[0] < key[1] < key[2]})
    anchored = {key for key, members in groups.items()
                if any(a in members and b in members for a, b in fault_pairs)}
    candidates = []
    for controller, target in sorted(truncation_pairs, key=lambda pair: (surface_ids[pair[0]], surface_ids[pair[1]])):
        if controller not in fault_participants or controller in fault_controllers:
            continue
        for row, neighbors in boundaries[target].items():
            if contact_ids[target][row] >= 0:
                continue
            cell = tuple(cell_coordinates[target][row])
            partner = lookups[controller].get(cell)
            if partner is None:
                continue
            key = int(contact_ids[controller][partner])
            if key not in anchored:
                continue
            shared_neighbors = [neighbor for neighbor in neighbors if contact_ids[target][neighbor] >= 0]
            if not shared_neighbors:
                continue
            matching = []
            for neighbor in shared_neighbors:
                other = lookups[controller].get(tuple(cell_coordinates[target][neighbor]))
                matching.append(other is not None and contact_ids[controller][other] == contact_ids[target][neighbor]
                                and tuple(sorted((partner, other))) in edges[controller])
            if not all(matching):
                continue
            delta = originals[target][row].astype(np.float64) - originals[controller][partner].astype(np.float64)
            candidates.append((float(np.dot(delta, delta)), surface_ids[controller], surface_ids[target],
                               cell, controller, target, row, partner, key))
    for _, _, _, cell, controller, target, row, partner, key in sorted(candidates):
        if ids[target][row] >= 0:
            continue
        members = groups[key]
        reason = None
        if target in fault_participants:
            reason = 'fault_participant'
        elif any(surface_ids[member][0] == surface_ids[target][0] for member in members):
            reason = 'same_group'
        elif not all(allowed_pairs[member, target] for member in members if member not in fault_controllers):
            reason = 'disallowed_cross_pair'
        incident = faces[target][np.any(faces[target] == row, axis=1)]
        proposed_ids = ids[target][incident].copy()
        proposed_ids[incident == row] = key
        if reason is None and any(tuple(triple) in patches[owner]
                                  for triple in np.sort(proposed_ids, axis=1)
                                  for owner in range(len(patches)) if owner != target):
            reason = 'duplicate_shared_patch'
        if reason is None:
            before = originals[target][incident].astype(np.float64)
            after = positions[target][incident].astype(np.float64)
            after[incident == row] = vertices[controller][partner]
            n0 = np.cross(before[:, 1] - before[:, 0], before[:, 2] - before[:, 0])
            n1 = np.cross(after[:, 1] - after[:, 0], after[:, 2] - after[:, 0])
            dot = np.einsum('ij,ij->i', n0, n1)
            if not np.all(np.isfinite(dot) & (dot > 0)):
                reason = 'collapsed_or_inverted_face'
        detail = dict(cell=cell, controller=surface_ids[controller], target=surface_ids[target])
        if reason is not None:
            report['fault_junction_rejected_count'] += 1
            report['fault_junction_rejections'].append(dict(detail, reason=reason))
            continue
        positions[target][row] = vertices[controller][partner]
        ids[target][row] = key
        members.add(target)
        patches[target].update(tuple(triple) for triple in np.sort(proposed_ids, axis=1)
                               if triple[0] >= 0 and triple[0] < triple[1] < triple[2])
        report['fault_junction_attachment_count'] += 1
        report['fault_junction_attachments'].append(detail)
    return positions, ids, report
