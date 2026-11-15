"""The stack ownership rule: which stack owns a point, from the stack relations.

One definition for every caller: ``_masking_ops._lithology_mask`` (lith block, squeezed masks), ``classify_units``
(hard classification of raw values) and ``build_contact_relations`` (the joint paths' symbolic truncations).

Each stack has a *row*: a conjunction of *atoms*, ``(stack, 'erode')`` (``raw > min(isovalues)``) or
``(stack, 'onlap')`` (``raw > max(isovalues)``), or FALSE. A stack owns where its row holds and no younger
(earlier) row does. Rows come from the stack sequence:

* An ERODE stack's row is its own erode atom; a BASEMENT (or ``False``) stack's row is TRUE (no atoms).
* An ONLAP stack's row is FALSE until a chain sets it: an onlap chain runs down consecutive ONLAP stacks, each
  onlapping the next older one (its onlap atom), and the stacks above in the chain accumulate the atoms below.
* FAULT stacks are transparent: they own nothing, impose nothing and do not interrupt or end onlap chains. The rule
  runs on the non-fault stacks only. (Before, a chain reaching a fault took the fault's field and gave the fault
  stack rock.)
* The first stack is also confined to the last non-fault stack's row, a no-op when that is the basement.

Null-space flags do not change rows: a null stack claims its region like any other (and callers make it void).
This module is pure Python; numeric evaluation only uses ``&``, ``|`` and ``~`` on the arrays the caller provides,
so the same code serves NumPy and torch.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from ...core.data.stack_relation_type import StackRelationType as R

ERODE, ONLAP = 'erode', 'onlap'


class StackRule:
    """Rows of the ownership rule for a stack sequence (youngest first)."""

    def __init__(self, relations: Sequence, null_flags: Optional[Sequence[bool]] = None):
        legacy_null = [relation is R.NULL_SPACE for relation in relations]
        self.relations = tuple(R.ERODE if null else relation for relation, null in zip(relations, legacy_null))
        flags = [False] * len(relations) if null_flags is None else list(null_flags)
        if len(flags) != len(relations):
            raise ValueError('invalid_stack_metadata: one null-space flag per stack required')
        self.null = tuple(bool(a) or b for a, b in zip(flags, legacy_null))
        self.lithology = tuple(i for i, relation in enumerate(self.relations) if relation is not R.FAULT)
        #: Per stack: ``None`` is FALSE, otherwise the frozenset of atoms of a conjunction (empty: TRUE).
        self.rows: list[Optional[frozenset]] = [None] * len(self.relations)
        self._build_rows()

    def _build_rows(self):
        lithology, relations = self.lithology, self.relations
        rows = [None] * len(lithology)  # Rows by position among the non-fault stacks
        chain = 0
        for p, stack in enumerate(lithology):
            # Indices wrap at the top (p - 1 == -1), exactly as the production loop always has.
            continues = relations[lithology[p - 1]] is R.ONLAP
            began = relations[lithology[p - 1 - chain]] is R.ONLAP
            chain = (chain + 1) * continues * began
            if chain:
                rows[p - 1] = frozenset({(stack, ONLAP)})
                cumulative = frozenset()
                for q in range(p - 1, p - chain - 1, -1):
                    cumulative = cumulative | (rows[q] or frozenset())
                    rows[q] = cumulative
            relation = relations[stack]
            if relation is R.ONLAP:
                continue
            if relation is R.ERODE:
                rows[p] = frozenset({(stack, ERODE)})
            elif relation is False or relation is R.BASEMENT:
                rows[p] = frozenset()
            else:
                raise ValueError(f'unsupported_relation: {relation}')
        for p, stack in enumerate(lithology):
            self.rows[stack] = rows[p]

    @property
    def last_lithology(self) -> Optional[int]:
        return self.lithology[-1] if self.lithology else None

    # region Numeric evaluation

    def owned(self, component: Callable, zeros: Callable, row_hook: Optional[Callable] = None) -> list:
        """Per stack, the boolean array of the points it owns.

        ``component(stack, kind)`` returns that stack's ``'erode'`` or ``'onlap'`` component, ``zeros()`` a FALSE
        array of the same length. ``row_hook(rows)`` may edit the evaluated rows in place before ownership (e.g.
        null stacks not claiming ignored grids).
        """
        cache = {}

        def atom(key):
            if key not in cache:
                cache[key] = component(*key)
            return cache[key]

        def evaluate(row):
            if row is None:
                return zeros()
            value = ~zeros()
            for key in sorted(row):
                value = value & atom(key)
            return value

        rows = [evaluate(row) for row in self.rows]
        if row_hook is not None:
            row_hook(rows)
        claimed = zeros()
        owned = []
        for i, row in enumerate(rows):
            owned.append(row & ~claimed)
            claimed = claimed | row
        if owned and self.lithology:
            owned[0] = owned[0] & rows[self.last_lithology]
        return owned

    # endregion

    # region Symbolic export

    def atoms(self, stack: int) -> frozenset:
        """A stack's row as atoms; FALSE has none."""
        return self.rows[stack] or frozenset()

    def dependencies(self, target: int) -> set:
        """Atoms the target stack's ownership depends on: its row and the younger rows that are not absorbed.

        A younger broader row absorbs narrower ones in their OR: ``!(A & B) & !B`` is just ``!B``, and a minimum
        threshold absorbs the same stack's maximum. Without this, onlap chains give spurious backward controllers.
        The first stack also depends on the last non-fault stack's row.
        """
        preceding = [self.atoms(k) for k in range(target) if self.atoms(k)]
        expanded = [entry | {(stack, ERODE) for stack, kind in entry if kind == ONLAP} for entry in preceding]
        dependencies = set(self.atoms(target))
        for j, entry in enumerate(preceding):
            if not any(other < expanded[j] for other in expanded):
                dependencies |= entry
        if target == 0 and self.lithology:
            dependencies |= self.atoms(self.last_lithology)
        return dependencies

    # endregion
