"""Symbolic contact relations between exported surfaces: eligibility, fault pairs, truncations."""

import numpy as np

from ...core.data.stack_relation_type import StackRelationType
from ..masking.stack_rule import StackRule


def build_contact_relations(surface_to_stack, surface_indices, stack_relations,
                            faults_relations, isovalues, null_stacks=None, fault_targets=False):
    """Return a symmetric eligibility matrix and directed surface-pair sets.

    Surface indices are local indices into each stack's full 1-D isovalue
    array, including boundaries not exported as meshes. Pair indices refer to
    the supplied surface order. Eligibility is NOT a connected-component weld
    rule: grouping must independently enforce at most one surface per stack.

    Truncation dependencies are those of the shared ownership rule
    (``StackRule.dependencies``), the same rule ``_masking_ops`` evaluates.
    Erosion uses the minimum boundary; onlap uses the maximum boundary of the
    stack supplying that mask (not necessarily a stack whose own relation is
    ONLAP). Faults are transparent to the rule and never become ordinary contact
    controllers. Null-space stacks (``null_stacks``, or a legacy ``NULL_SPACE``
    relation, which erodes) mask by their relation but take part in no contact.
    With ``fault_targets``, fault surfaces are truncated by the younger non-fault
    stacks (a fault cuts only older rock); their controllers keep the negative side
    (``ordinary_controllers`` with ``fault_stacks``). No inputs are modified.
    """
    stacks = np.asarray(surface_to_stack)
    indices = np.asarray(surface_indices)
    n = len(stacks)
    legacy_null = [relation is StackRelationType.NULL_SPACE for relation in stack_relations]
    relations = tuple(StackRelationType.ERODE if null else relation
                      for relation, null in zip(stack_relations, legacy_null))
    ns = len(relations)
    nulls = np.array(legacy_null, dtype=bool) if null_stacks is None else \
        np.array(legacy_null, dtype=bool) | np.asarray(null_stacks, dtype=bool)
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

    excluded = np.array([relation is StackRelationType.FAULT for relation in relations]) | nulls | faults.any(axis=1)
    allowed_pairs = (stacks[:, None] != stacks[None, :]) & \
        ~excluded[stacks, None] & ~excluded[stacks][None, :]
    fault_pairs = {(i, j) for i in range(n) for j in range(n)
                   if stacks[i] != stacks[j] and faults[stacks[i], stacks[j]]}

    rule = StackRule(relations, nulls)
    # With fault_targets a fault surface is truncated by the younger non-fault stacks, but it never controls a
    # contact; null stacks are neither.
    is_fault = np.array([relation is StackRelationType.FAULT for relation in relations])
    untargeted = excluded & ~is_fault if fault_targets else excluded
    truncation_pairs = set()
    for target_stack in range(ns):
        dependencies = rule.dependencies(target_stack)
        for controller_stack, kind in dependencies:
            if (controller_stack == target_stack or excluded[controller_stack] or untargeted[target_stack]
                    or not len(levels[controller_stack])):
                continue
            boundary = (np.min if kind == 'erode' else np.max)(levels[controller_stack])
            controllers = [i for i in range(n) if stacks[i] == controller_stack
                           and levels[controller_stack][indices[i]] == boundary]
            targets = np.flatnonzero(stacks == target_stack)
            truncation_pairs.update((i, int(j)) for i in controllers for j in targets)
            if is_fault[target_stack]:
                # The fault ends on its controller's contact: they may share a junction vertex there.
                for i in controllers:
                    allowed_pairs[i, targets] = allowed_pairs[targets, i] = True
    return allowed_pairs, fault_pairs, truncation_pairs
