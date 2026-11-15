"""Lithology classification of raw stack values, with the production masking rule (``StackRule``).

Hard (non-sigmoid) version of ``_lithology_mask`` and the stack segmentation. Values are
raw stack scalars, e.g. from ``evaluate_fields``.
"""

import numpy as np

from ..masking.stack_rule import ERODE, StackRule


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
    rule = StackRule(stack_relations, null_stacks)

    def component(i, kind):
        isovalues = np.asarray(stack_isovalues[i], dtype=float)
        return values[i] > (isovalues.min() if kind == ERODE else isovalues.max())

    final = np.stack(rule.owned(component=component, zeros=lambda: np.zeros(m, dtype=bool)))
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
