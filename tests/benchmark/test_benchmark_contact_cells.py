"""Prepared cell reconciliation, without extraction or benchmark-owned algorithms.

Opt in with GEMPY_CONTACT_AWARE_BENCHMARKS=1.
GEMPY_CONTACT_BENCHMARK_ROUNDS defaults to 5. Copies are made by pedantic setup,
outside timing; each round has one invocation. Inputs are detached NumPy arrays.
The fault case covers directional vertex snapping only: triangle removal belongs
to the extraction/model integration benchmarks. Coarse cell sticking is accepted.
"""

import copy
import os
from itertools import combinations

import numpy as np
import pytest


CELL_CASES = ("sparse", "dense", "no_overlap", "multiple_horizons", "fault")


def prepare_contact_cells(case, size=256, dtype="float64"):
    """Return the production callable, fresh-input setup, and workload metadata."""
    # Keep production imports out of dataset-independent collection.
    from gempy_engine.modules.dual_contouring.contact_cells import reconcile_cell_vertices

    if case not in CELL_CASES:
        raise ValueError(f"Unknown contact-cell dataset: {case}")
    cells = np.column_stack((np.arange(size), np.zeros(size), np.zeros(size))).astype(np.int64)
    count = 4 if case == "multiple_horizons" else (3 if case == "dense" else 2)
    coordinates = [cells.copy() for _ in range(count)]
    if case == "no_overlap":
        coordinates[1][:, 0] += size
    elif case == "sparse":
        coordinates[1][size // 8:, 0] += size
    stacks = [0, 0, 1, 2] if case == "multiple_horizons" else list(range(count))
    # Stable surface identities are local to their stack, not list offsets.
    surface_indices = [0, 1, 0, 0] if case == "multiple_horizons" else [0] * count
    surface_ids = list(zip(stacks, surface_indices))
    offsets = [0.125, 0.875, 0.375, 0.625][:count]
    vertices = [(c.astype(dtype) + offset) for c, offset in zip(coordinates, offsets)]
    allowed = np.not_equal.outer(stacks, stacks)
    fault_pairs = ((0, 1),) if case == "fault" else ()
    if fault_pairs:
        allowed[:] = False
    args = vertices, coordinates, stacks, surface_ids, allowed, fault_pairs
    pairs = [(i, j) for i, j in combinations(range(count), 2) if allowed[i, j]]
    eligible_pairs = pairs + list(fault_pairs)
    cell_sets = [set(map(tuple, c)) for c in coordinates]
    shared = [cell_sets[i] & cell_sets[j] for i, j in eligible_pairs]
    info = dict(scope="contact-cells", case=case, mode="contact_aware", backend="numpy",
                dtype=dtype, surfaces=count, stacks=len(set(stacks)),
                surface_identities=surface_ids,
                active_cells_per_surface=[len(c) for c in coordinates],
                input_vertices=sum(map(len, vertices)), input_triangles=0,
                candidate_surface_pairs=len(pairs), directed_fault_pairs=len(fault_pairs),
                shared_cell_pair_incidences=sum(map(len, shared)),
                unique_shared_cells=len(set().union(*shared)) if shared else 0,
                actual_contact_cells=None, inserted_seam_vertices=0,
                contact_count_note="Shared integer cells are candidates, not geometric seam contacts.",
                stage_note="Prepared module only; excludes extraction, policy construction and triangle removal.")
    return reconcile_cell_vertices, lambda: (copy.deepcopy(args), {}), info


def add_contact_cell_counts(info, result, inputs):
    vertices, contact_ids, report = result
    info.update(output_vertices=sum(map(len, vertices)), output_triangles=0,
                output_contact_id_rows=sum(map(len, contact_ids)),
                reconciled_contact_sets=report['contact_count'],
                rejected_contact_candidates=report['conflict_count'],
                reconciliation_report=repr(report))
    # Report schema is owned by production; do not relabel candidate cells as contacts.
    _, coordinates, _, _, allowed, fault_pairs = inputs
    pairs = [(i, j) for i, j in combinations(range(len(vertices)), 2) if allowed[i, j]]
    pairs += list(fault_pairs)
    rows = [{tuple(cell): row for row, cell in enumerate(c)} for c in coordinates]
    coincident_cells = set()
    incidences = 0
    for i, j in pairs:
        for cell in rows[i].keys() & rows[j].keys():
            if np.array_equal(vertices[i][rows[i][cell]], vertices[j][rows[j][cell]]):
                incidences += 1
                coincident_cells.add(cell)
    info.update(coincident_output_cell_pair_incidences=incidences,
                unique_coincident_output_cells=len(coincident_cells))


@pytest.mark.parametrize("case", CELL_CASES)
@pytest.mark.parametrize("size", [32, 256], ids=lambda value: f"n{value}")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_benchmark_contact_cells(benchmark, case, size, dtype):
    if os.getenv("GEMPY_CONTACT_AWARE_BENCHMARKS") != "1":
        pytest.skip("Set GEMPY_CONTACT_AWARE_BENCHMARKS=1 after contact_cells is integrated")
    target, setup, info = prepare_contact_cells(case, size, dtype)
    original, _ = setup()
    result = benchmark.pedantic(
        target, setup=setup, rounds=int(os.getenv("GEMPY_CONTACT_BENCHMARK_ROUNDS", "5")),
        iterations=1, warmup_rounds=0,
    )
    vertices, contact_ids, _ = result
    assert len(vertices) == len(original[0]) == len(contact_ids)
    for before, after, ids in zip(original[0], vertices, contact_ids):
        assert after.shape == before.shape
        assert len(ids) == len(after)
        assert np.isfinite(after).all()
    if case == "no_overlap":
        for before, after in zip(original[0], vertices):
            np.testing.assert_array_equal(after, before)
    elif case in ("sparse", "dense"):
        shared_rows = size // 8 if case == "sparse" else size
        expected = np.mean(np.stack([v[:shared_rows] for v in original[0]]), axis=0)
        for after in vertices:
            np.testing.assert_allclose(after[:shared_rows], expected, rtol=1e-6)
        if case == "sparse":
            for before, after in zip(original[0], vertices):
                np.testing.assert_array_equal(after[shared_rows:], before[shared_rows:])
    elif case == "fault":
        np.testing.assert_array_equal(vertices[0], original[0][0])
        np.testing.assert_array_equal(vertices[1], original[0][0])
    else:
        # Either horizon may be selected, but a third group must not bridge them.
        assert np.all(np.any(vertices[0] != vertices[1], axis=1))
    add_contact_cell_counts(info, result, original)
    benchmark.extra_info.update(info)
