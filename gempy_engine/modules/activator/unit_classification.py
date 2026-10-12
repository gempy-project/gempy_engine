"""Lithology classification of raw stack values, with the production masking rules.

Hard (non-sigmoid) version of ``_lithology_mask`` and the stack segmentation:
erosion/onlap components, onlap chains and preceding-mask exclusion. Values are
raw stack scalars, e.g. from ``evaluate_fields``.
"""

import numpy as np

from ...core.data.stack_relation_type import StackRelationType as R


def classify_units(stack_values, stack_relations, stack_isovalues):
    """Active stack and interval per point.

    ``stack_values`` (S, M) are raw scalars of every stack and
    ``stack_isovalues`` each stack's surface levels. Returns ``(stack,
    interval)`` int arrays (M,): the interval is the number of the active
    stack's levels above the value (0 is above its first surface).
    """
    values = np.asarray(stack_values, dtype=float)
    n, m = values.shape
    if len(stack_relations) != n or len(stack_isovalues) != n:
        raise ValueError('invalid_stack_metadata: one relation and isovalue array per stack required')

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
        if relation is R.NULL_SPACE:
            raise ValueError('unsupported_relation: null space')
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


def unit_ids(stack, interval, number_of_surfaces_per_stack, unit_values):
    """Production unit ids of ``(stack, interval)``: each stack's slice of ``unit_values``."""
    offsets = np.concatenate(([0], np.cumsum(number_of_surfaces_per_stack)[:-1]))
    return np.asarray(unit_values)[offsets[stack] + interval]
