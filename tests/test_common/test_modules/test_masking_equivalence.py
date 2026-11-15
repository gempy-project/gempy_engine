"""The three evaluations of the stack ownership rule (``StackRule``) agree.

* ``_masking_ops._lithology_mask``: production ownership (lith block, pretty/none meshes through the squeezed masks).
* ``classify_units``: the hard NumPy evaluation used by field evaluation.
* ``build_contact_relations`` + ``ordinary_controllers``: the symbolic export the joint paths evaluate as sign tests,
  ``sign * (raw_c - iso_c) > 0`` for every controller surface ``c`` of a target surface.
* ``StackRule.region``: where none/pretty extract a stack's surfaces (any voxel corner in it).

Ownership at a point depends only on the stack values there, so points are random value vectors. A surface ``t`` of
stack ``s`` is present at a point on it (``raw_s == iso_t``) when the production unit changes across it
(``iso_t ± EPS``): ``s`` need not own a side, e.g. a one-surface stack onlapped from above has no thickness, yet its
surface is the onlap contact. Joint keeps it when all its controllers pass. Faults own no rock and take part in no contact, so
fault surfaces are not targets, but fault stacks stay in the sequences: onlap chains run through them.
"""

import itertools
import zlib

import numpy as np
import pytest

from gempy_engine.API.interp_single._masking_ops import _lithology_mask
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.core.data.scalar_field_output import ScalarFieldOutput
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.modules.activator.unit_classification import classify_units
from gempy_engine.modules.dual_contouring.joint_contact_relations import build_contact_relations
from gempy_engine.modules.dual_contouring.joint_ownership import ordinary_controllers
from gempy_engine.modules.masking.stack_rule import ERODE, StackRule

N_POINTS = 400
EPS = 1e-7


def _configurations():
    """Every sequence of 2 to 4 stacks ending in the basement, with every allowed null-space flag combination."""
    for n in (2, 3, 4):
        for upper in itertools.product((R.ERODE, R.ONLAP, R.FAULT), repeat=n - 1):
            relations = [*upper, R.BASEMENT]
            nullable = [i for i, r in enumerate(relations) if r is not R.FAULT]
            for flags in itertools.product((False, True), repeat=len(nullable)):
                null = [False] * n
                for i, flag in zip(nullable, flags):
                    null[i] = flag
                yield relations, null


CONFIGURATIONS = list(_configurations())


def _label(configuration):
    relations, null = configuration
    return "-".join(r.name.lower() + ("_null" if f else "") for r, f in zip(relations, null))


def _isovalues(rng, n):
    return [np.sort(rng.uniform(-.6, .6, size=rng.integers(1, 3)))[::-1].copy() for _ in range(n)]


def _production_owner(values, relations, null, isovalues):
    """Stack owning each point by ``_lithology_mask``, -1 where none does (the bottom null stack's void)."""
    n, m = values.shape
    outputs = [ScalarFieldOutput(
        weights=None, grid=None, stack_relation=relations[i], values_block=None, is_null_space=null[i],
        exported_fields=ExportedFields(_scalar_field=BackendTensor.t.array(values[i]), _grid_size=m,
                                       _scalar_field_at_surface_points=BackendTensor.t.array(isovalues[i])))
        for i in range(n)]
    stacks = StacksStructure(
        number_of_points_per_stack=np.array([2 * len(v) for v in isovalues]),
        number_of_orientations_per_stack=np.ones(n, dtype=int),
        number_of_surfaces_per_stack=np.array([len(v) for v in isovalues]),
        masking_descriptor=list(relations), null_space_per_stack=list(null))
    final = BackendTensor.t.to_numpy(_lithology_mask(outputs, stacks)).astype(bool)
    assert (final.sum(axis=0) <= 1).all(), "production ownership must be exclusive"
    return np.where(final.any(axis=0), np.argmax(final, axis=0), -1)


def _production_unit(values, relations, null, isovalues):
    """Production unit per point: ``(owner, interval)`` as one code; a null owner's void is one unit, -1 none."""
    owner = _production_owner(values, relations, null, isovalues)
    unit = np.full(owner.shape, -1)
    for g in np.unique(owner[owner >= 0]):
        rows = owner == g
        interval = np.sum(isovalues[g][:, None] > values[g, rows], axis=0)
        bottom_null = null[g] and g == len(relations) - 1
        if null[g]:
            interval = np.where(bottom_null & (interval >= len(isovalues[g])), interval, -1)
        unit[rows] = 100 * g + interval + 1
    return unit


def _surfaces(isovalues):
    return [(s, k) for s, values in enumerate(isovalues) for k in range(len(values))]


def _joint_controllers(relations, isovalues):
    surfaces = _surfaces(isovalues)
    _, _, truncations = build_contact_relations(
        np.array([s for s, _ in surfaces]), np.array([k for _, k in surfaces]), relations, None, isovalues)
    return surfaces, ordinary_controllers(surfaces, truncations)


@pytest.mark.parametrize("configuration", CONFIGURATIONS, ids=[_label(c) for c in CONFIGURATIONS])
def test_classify_units_matches_production(configuration):
    relations, null = configuration
    rng = np.random.default_rng(zlib.crc32(_label(configuration).encode()))
    isovalues = _isovalues(rng, len(relations))
    values = rng.uniform(-1, 1, size=(len(relations), N_POINTS))
    owner = _production_owner(values, relations, null, isovalues)
    stack, _ = classify_units(values, relations, isovalues, null_stacks=null)
    owned = owner >= 0  # classify_units leaves the bottom null void to unit_ids
    np.testing.assert_array_equal(stack[owned], owner[owned])


@pytest.mark.parametrize("configuration", CONFIGURATIONS, ids=[_label(c) for c in CONFIGURATIONS])
def test_faults_own_no_rock(configuration):
    relations, null = configuration
    rng = np.random.default_rng(zlib.crc32(_label(configuration).encode()))
    isovalues = _isovalues(rng, len(relations))
    owner = _production_owner(rng.uniform(-1, 1, size=(len(relations), N_POINTS)), relations, null, isovalues)
    faults = [i for i, r in enumerate(relations) if r is R.FAULT]
    assert not np.isin(owner, faults).any()


@pytest.mark.parametrize("configuration", CONFIGURATIONS, ids=[_label(c) for c in CONFIGURATIONS])
def test_joint_truncations_match_production(configuration):
    relations, null = configuration
    rng = np.random.default_rng(zlib.crc32(_label(configuration).encode()))
    isovalues = _isovalues(rng, len(relations))
    surfaces, controllers = _joint_controllers(relations, isovalues)
    mismatches = []
    for t, (s, k) in enumerate(surfaces):
        if relations[s] is R.FAULT or null[s]:
            continue
        values = rng.uniform(-1, 1, size=(len(relations), N_POINTS))
        values[s] = isovalues[s][k]
        sides = []
        for offset in (EPS, -EPS):
            shifted = values.copy()
            shifted[s] += offset
            sides.append(_production_unit(shifted, relations, null, isovalues))
        present = sides[0] != sides[1]  # The surface is a contact where the unit changes across it
        kept = np.ones(N_POINTS, dtype=bool)
        for c, sign in controllers[t]:
            cs, ck = surfaces[c]
            kept &= sign * (values[cs] - isovalues[cs][ck]) > 0
        if (present != kept).any():
            mismatches.append(f"surface {s}/{k}: production-only {(present & ~kept).sum()}, "
                              f"joint-only {(kept & ~present).sum()} of {N_POINTS}")
    assert not mismatches, "; ".join(mismatches)


@pytest.mark.parametrize("configuration", CONFIGURATIONS, ids=[_label(c) for c in CONFIGURATIONS])
def test_surface_regions_match_joint_truncations(configuration):
    """A surface point has a side in its stack's ``region`` exactly where the joint controllers keep the surface."""
    relations, null = configuration
    rng = np.random.default_rng(zlib.crc32(_label(configuration).encode()))
    isovalues = _isovalues(rng, len(relations))
    surfaces, controllers = _joint_controllers(relations, isovalues)
    rule = StackRule(relations, null)
    mismatches = []
    for t, (s, k) in enumerate(surfaces):
        if relations[s] is R.FAULT or null[s]:
            continue
        values = rng.uniform(-1, 1, size=(len(relations), N_POINTS))
        values[s] = isovalues[s][k]
        in_region = np.zeros(N_POINTS, dtype=bool)
        for offset in (EPS, -EPS):
            shifted = values.copy()
            shifted[s] += offset
            component = lambda g, kind: shifted[g] > (isovalues[g].min() if kind == ERODE else isovalues[g].max())
            in_region |= rule.region(s, component, zeros=lambda: np.zeros(N_POINTS, dtype=bool))
        kept = np.ones(N_POINTS, dtype=bool)
        for c, sign in controllers[t]:
            cs, ck = surfaces[c]
            kept &= sign * (values[cs] - isovalues[cs][ck]) > 0
        if (in_region != kept).any():
            mismatches.append(f"surface {s}/{k}: region-only {(in_region & ~kept).sum()}, "
                              f"joint-only {(kept & ~in_region).sum()} of {N_POINTS}")
    assert not mismatches, "; ".join(mismatches)
