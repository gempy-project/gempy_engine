"""Symbolic contact relations between exported surfaces: eligibility, fault pairs, truncations."""

import numpy as np

from ...core.data.stack_relation_type import StackRelationType


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
