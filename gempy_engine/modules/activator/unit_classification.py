"""Lithology classification of raw stack values, with the production masking rules.

Hard (non-sigmoid) version of ``_lithology_mask`` and the stack segmentation:
erosion/onlap components, onlap chains and preceding-mask exclusion. Values are
raw stack scalars, e.g. from ``evaluate_fields``.
"""

import numpy as np

from ...core.data.stack_relation_type import StackRelationType as R


def classify_units(stack_values, stack_relations, stack_isovalues, null_stacks=None):
    """Active stack and interval per point.

    ``stack_values`` (S, M) are raw scalars of every stack and
    ``stack_isovalues`` each stack's surface levels. Returns ``(stack,
    interval)`` int arrays (M,): the interval is the number of the active
    stack's levels above the value (0 is above its first surface).
    ``null_stacks`` flags stacks whose region is void; they mask by their relation
    like any other, and ``unit_ids`` gives their points the null id. A legacy
    ``NULL_SPACE`` relation is an eroding null stack.
    """
    values = np.asarray(stack_values, dtype=float)
    n, m = values.shape
    if len(stack_relations) != n or len(stack_isovalues) != n:
        raise ValueError('invalid_stack_metadata: one relation and isovalue array per stack required')
    if null_stacks is not None and len(null_stacks) != n:
        raise ValueError('invalid_stack_metadata: one null-space flag per stack required')
    stack_relations = [R.ERODE if relation is R.NULL_SPACE else relation for relation in stack_relations]

    def component(i, kind):
        isovalues = np.asarray(stack_isovalues[i], dtype=float)
        if kind is R.ERODE:
            return values[i] > isovalues.min()
        if kind is R.ONLAP:
            return values[i] > isovalues.max()
        if kind is R.FAULT:
            return np.zeros(m, dtype=bool)
        if kind is False or kind is R.BASEMENT:
            return np.ones(m, dtype=bool)
        raise ValueError(f'unsupported_relation: {kind}')

    mask = np.zeros((n, m), dtype=bool)
    chain = 0
    for i in range(n):
        continues = stack_relations[i - 1] in (R.ONLAP, R.FAULT)
        began = stack_relations[i - 1 - chain] is R.ONLAP
        chain = (chain + 1) * continues * began
        if chain:
            mask[i - 1] = component(i, R.ONLAP)
            block = mask[i - chain:i]
            mask[i - chain:i] = np.flip(np.cumprod(np.flip(block, axis=0), axis=0), axis=0).astype(bool)
        relation = stack_relations[i]
        if relation is R.ONLAP:
            continue
        mask[i] = component(i, relation)
    final = np.zeros_like(mask)
    final[0] = mask[-1]
    final[1:] = np.cumprod(~mask[:-1], axis=0).astype(bool)
    final &= mask
    if not final.any(axis=0).all():
        raise ValueError('unclassified_point: no active stack')
    stack = np.argmax(final, axis=0)
    interval = np.zeros(m, dtype=np.int64)
    for g in np.unique(stack):
        rows = stack == g
        interval[rows] = np.sum(np.asarray(stack_isovalues[g], dtype=float)[:, None] > values[g, rows], axis=0)
    return stack, interval


def unit_ids(stack, interval, number_of_elements_per_stack, unit_values, null_stacks=None, null_space_id=-1):
    """Production unit ids of ``(stack, interval)``: each stack's slice of ``unit_values``.

    Null stacks own no units (``StacksStructure.unit_offsets``); their points get ``null_space_id``. A null last
    stack is the bottom one: below its lowest surface its points are the basement, the unit after all others.
    """
    counts = np.asarray(number_of_elements_per_stack, dtype=np.int64)
    nulls = np.zeros(len(counts), dtype=bool) if null_stacks is None else np.asarray(null_stacks, dtype=bool)
    offsets = np.concatenate(([0], np.cumsum(counts * ~nulls)[:-1]))
    bottom = np.zeros(len(counts), dtype=bool)
    bottom[-1] = nulls[-1]
    unit_values = np.asarray(unit_values)
    ids = unit_values[np.minimum(offsets[stack] + np.where(bottom[stack], 0, interval), len(unit_values) - 1)]
    return np.where(nulls[stack] & ~(bottom[stack] & (interval >= counts[stack])), null_space_id, ids)
