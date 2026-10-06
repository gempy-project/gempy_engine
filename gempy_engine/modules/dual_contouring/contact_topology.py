"""Detached NumPy relation and triangle operations for shared octree contacts."""

import numpy as np

from gempy_engine.core.data.stack_relation_type import StackRelationType


def contact_surface_roles(surface_to_stack, stack_relations, faults_relations):
    """Return fault-interface and ordinary ownership-removal roles per surface."""
    groups = np.asarray(surface_to_stack, dtype=np.int64)
    faults = (np.zeros((len(stack_relations), len(stack_relations)), dtype=bool)
              if faults_relations is None else np.asarray(faults_relations, dtype=bool))
    fault_stacks = np.array([relation is StackRelationType.FAULT for relation in stack_relations], dtype=bool)
    fault_stacks |= faults.any(axis=1)
    ordinary_stacks = ~fault_stacks & np.array(
        [relation is not StackRelationType.NULL_SPACE for relation in stack_relations], dtype=bool,
    )
    return fault_stacks[groups], ordinary_stacks[groups]


def build_contact_relations(surface_to_stack, surface_indices, stack_relations,
                            faults_relations, isovalues):
    """Return a symmetric eligibility matrix and directed surface-pair sets.

    Surface indices are local indices into each stack's full 1-D isovalue
    array, including boundaries not exported as meshes. Pair indices refer to
    the supplied surface order. Eligibility is NOT a connected-component weld
    rule: grouping must independently enforce at most one surface per stack.

    Truncation dependencies mirror the onlap cumulative products and preceding
    mask exclusion in _masking_ops. Erosion uses the minimum boundary; onlap
    uses the maximum boundary of the stack supplying that mask (not necessarily
    a stack whose own relation is ONLAP). Faults may carry an onlap chain but
    never become ordinary contact controllers. No inputs are modified.
    """
    stacks = np.asarray(surface_to_stack)
    indices = np.asarray(surface_indices)
    n = len(stacks)
    relations = tuple(stack_relations)
    ns = len(relations)
    if stacks.shape != (n,) or indices.shape != (n,) or \
            (n and (stacks.dtype.kind not in 'iu' or indices.dtype.kind not in 'iu')):
        raise ValueError('Expected integer surface_to_stack and surface_indices arrays')
    if np.any((stacks < 0) | (stacks >= ns)):
        raise ValueError('Surface stack index out of range')
    if len(isovalues) != ns:
        raise ValueError('Expected one isovalue array per stack')
    levels = [np.asarray(values) for values in isovalues]
    for stack, values in enumerate(levels):
        if values.ndim != 1 or not np.all(np.isfinite(values)) or (not len(values) and np.any(stacks == stack)):
            raise ValueError('Exported stack isovalues must be nonempty finite 1-D arrays')
    for stack, index in zip(stacks, indices):
        if index < 0 or index >= len(levels[stack]):
            raise ValueError('Surface index out of range')
    valid_relations = set(StackRelationType) | {False}
    if any(relation not in valid_relations for relation in relations):
        raise ValueError('Unrecognized stack relation')
    faults = np.zeros((ns, ns), dtype=bool) if faults_relations is None else np.asarray(faults_relations, dtype=bool)
    if faults.shape != (ns, ns):
        raise ValueError('Expected square stack fault relation matrix')
    if not n:
        return np.zeros((0, 0), dtype=bool), set(), set()

    excluded = np.array([relation in (StackRelationType.FAULT, StackRelationType.NULL_SPACE)
                         for relation in relations]) | faults.any(axis=1)
    allowed_pairs = (stacks[:, None] != stacks[None, :]) & \
        ~excluded[stacks, None] & ~excluded[stacks][None, :]
    fault_pairs = {(i, j) for i in range(n) for j in range(n)
                   if stacks[i] != stacks[j] and faults[stacks[i], stacks[j]]}

    # Symbolic dependencies of mask_matrix, before squeezed ownership exclusion.
    masks = [set() for _ in relations]
    chain = 0
    for i, relation in enumerate(relations):
        continues = relations[i - 1] in (StackRelationType.ONLAP, StackRelationType.FAULT)
        began = relations[i - 1 - chain] is StackRelationType.ONLAP
        chain = (chain + 1) * continues * began
        if chain:
            masks[i - 1] = {(i, 'onlap')}
            cumulative = set()
            for j in range(i - 1, i - chain - 1, -1):
                cumulative = cumulative | masks[j]
                masks[j] = cumulative.copy()
        if relation in (StackRelationType.ERODE, StackRelationType.NULL_SPACE):
            masks[i] = {(i, 'erode')}
        elif relation is not StackRelationType.ONLAP:
            masks[i] = set()

    truncation_pairs = set()
    for target_stack, mask in enumerate(masks):
        # A preceding broader mask absorbs narrower masks in their OR. For
        # example !(A & B) & !B is just !B, and a minimum threshold absorbs
        # the same stack's maximum threshold. Without this, onlap chains create
        # spurious backward controller pairs and redundant contact cycles.
        preceding = [entry for entry in masks[:target_stack] if entry]
        expanded = [entry | {(stack, 'erode') for stack, kind in entry if kind == 'onlap'}
                    for entry in preceding]
        dependencies = mask.copy()
        for j, entry in enumerate(preceding):
            if not any(other < expanded[j] for other in expanded):
                dependencies |= entry
        if target_stack == 0 and masks:
            dependencies |= masks[-1]
        for controller_stack, kind in dependencies:
            if (controller_stack == target_stack or excluded[controller_stack] or excluded[target_stack]
                    or not len(levels[controller_stack])):
                continue
            boundary = (np.min if kind == 'erode' else np.max)(levels[controller_stack])
            controllers = [i for i in range(n) if stacks[i] == controller_stack
                           and levels[controller_stack][indices[i]] == boundary]
            targets = np.flatnonzero(stacks == target_stack)
            truncation_pairs.update((i, int(j)) for i in controllers for j in targets)
    return allowed_pairs, fault_pairs, truncation_pairs


def reconcile_cell_faces(faces, contact_ids, corner_ownership,
                         fault_overlap_vertices, truncation_pairs, *, ownership_targets=None):
    """Return copied faces and disjoint per-surface removal counts.

    Each vertex maps to a cell; ownership is bool (Nv, 8), sampled from actual
    squeezed geological masks, NOT extraction/crossing flags. A cell is hidden
    only if all eight corners are unowned. Ownership removal applies ONLY to
    explicit ownership_targets, or targets in truncation_pairs when omitted.
    The API includes ordinary surfaces hidden by non-exported null-space groups
    and excludes fault interfaces, which are not lithological volume owners.
    Any owned corner preserves boundary support. RAW callers deliberately
    supplying full ownership retain all faces
    except directed fault overlaps and redundant controller contact patches.

    Fault overlap indices must already be collected on directed targets. Only
    triangles with ALL three indices in that set are removed. A shared patch
    requires three distinct nonnegative stable contact IDs matching a surviving
    controller triangle, independent of winding. Positions are never consulted.
    No vertices are welded, renumbered, generated or mutated.
    """
    n = len(faces)
    if any(len(items) != n for items in (contact_ids, corner_ownership, fault_overlap_vertices)):
        raise ValueError('Expected metadata for every surface')
    pairs = set(truncation_pairs)
    for controller, target in pairs:
        if not (0 <= controller < n and 0 <= target < n) or controller == target:
            raise ValueError('Invalid directed truncation pair')
    truncation_targets = {target for _, target in pairs}
    if ownership_targets is not None:
        truncation_targets = set(ownership_targets)
        if any(not isinstance(index, (int, np.integer)) or not 0 <= index < n for index in truncation_targets):
            raise ValueError('Invalid ownership target surface index')
    arrays, ids, keep, counts = [], [], [], []
    for surface in range(n):
        triangles = np.asarray(faces[surface])
        shared = np.asarray(contact_ids[surface])
        ownership = np.asarray(corner_ownership[surface])
        overlap = np.asarray(fault_overlap_vertices[surface])
        if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in 'iu':
            raise ValueError('Faces must be integer (M, 3) arrays')
        if shared.ndim != 1 or shared.dtype.kind not in 'iu' or np.any(shared < -1):
            raise ValueError('Contact IDs must be integer (Nv,) arrays with -1 unshared')
        if ownership.shape != (len(shared), 8) or ownership.dtype.kind != 'b':
            raise ValueError('Corner ownership must be bool (Nv, 8)')
        if np.any((triangles < 0) | (triangles >= len(shared))):
            raise ValueError('Triangle vertex index out of range')
        if overlap.ndim != 1 or (overlap.size and overlap.dtype.kind not in 'iu') or \
                np.any((overlap < 0) | (overlap >= len(shared))):
            raise ValueError('Fault overlap vertex index out of range')
        fault_removed = np.all(np.isin(triangles, overlap), axis=1)
        hidden = ~np.any(ownership[triangles], axis=(1, 2)) & ~fault_removed & (surface in truncation_targets)
        arrays.append(triangles)
        ids.append(shared)
        keep.append(~(fault_removed | hidden))
        counts.append(dict(input_count=len(triangles), fault_removed_count=int(fault_removed.sum()),
                           ownership_removed_count=int(hidden.sum()), shared_patch_removed_count=0))

    # Snapshot supported controller patches so pair iteration order cannot change
    # eligibility. A removed intermediate patch can refer to a surviving ancestor.
    patches = []
    for surface in range(n):
        keys = np.sort(ids[surface][arrays[surface][keep[surface]]], axis=1)
        patches.append({tuple(row) for row in keys
                        if row[0] >= 0 and row[0] < row[1] < row[2]})
    removals = [np.zeros(len(array), dtype=bool) for array in arrays]
    for controller, target in pairs:
        keys = np.sort(ids[target][arrays[target]], axis=1)
        removals[target] |= np.array([tuple(row) in patches[controller] for row in keys], dtype=bool)
    supported = []
    removed_patches = []
    for surface in range(n):
        removed_keys = {tuple(row) for row in np.sort(
            ids[surface][arrays[surface][removals[surface] & keep[surface]]], axis=1)}
        removed_patches.append(removed_keys)
        supported.append(patches[surface] - removed_keys)
    for _ in range(n):
        changed = False
        for controller, target in pairs:
            inherited = removed_patches[target] & supported[controller]
            if inherited - supported[target]:
                supported[target] |= inherited
                changed = True
        if not changed:
            break
    if any(removed - support for removed, support in zip(removed_patches, supported)):
        raise ValueError('Truncation relations would remove every copy of a controller patch')
    result = []
    for surface in range(n):
        counts[surface]['shared_patch_removed_count'] = int((removals[surface] & keep[surface]).sum())
        retained = keep[surface] & ~removals[surface]
        result.append(arrays[surface][retained].copy())
        counts[surface]['output_count'] = int(retained.sum())
    return result, {'per_surface': counts,
                    'removed_count': sum(item['input_count'] - item['output_count'] for item in counts)}
