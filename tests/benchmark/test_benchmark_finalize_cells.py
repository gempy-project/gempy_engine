"""Opt-in finalization timing; reconciliation and fixture copies are untimed.

Use GEMPY_CONTACT_AWARE_BENCHMARKS=1 and GEMPY_CONTACT_BENCHMARK_ROUNDS.
Isolated total-process RSS is available via contact_benchmark_runner.py with
--scope finalize-cells; it is not a stage-only allocation measurement.
"""

import copy
import os

import numpy as np
import pytest


FINALIZE_CASES = ("compound_supported", "compound_pruned")


def prepare_finalize_cells(case, size=256, dtype="float64"):
    from gempy_engine.modules.dual_contouring.contact_cells import finalize_cell_vertices

    if case not in FINALIZE_CASES:
        raise ValueError(f"Unknown finalization dataset: {case}")
    cells = np.column_stack((np.arange(size), np.arange(size) % 2, np.zeros(size)))
    originals = [(cells + offset).astype(dtype) for offset in (0.125, 0.375, 0.875)]
    mean = np.mean(np.stack(originals).astype(np.float64), axis=0).astype(dtype)
    vertices = [mean.copy() for _ in originals]
    contact_ids = [np.arange(size, dtype=np.int64) for _ in originals]
    # Sliding triangles support every row, including sizes not divisible by three.
    triangles = np.column_stack((np.arange(size - 2), np.arange(1, size - 1),
                                 np.arange(2, size))).astype(np.int64)
    faces = [triangles.copy() for _ in originals]
    if case == "compound_pruned":
        faces[2] = np.empty((0, 3), dtype=np.int64)
    args = originals, vertices, contact_ids, faces, [(0, 0), (1, 0), (2, 0)]
    info = dict(scope="finalize-cells", case=case, mode="contact_aware", backend="numpy",
                dtype=dtype, surfaces=3, stacks=3, input_vertices=3 * size,
                input_triangles=sum(map(len, faces)), provisional_contact_count=size,
                members_per_provisional_contact=3,
                 stage_note="Prepared finalize_cell_vertices only; excludes reconciliation, topology and fixture copies.")
    return finalize_cell_vertices, lambda: (copy.deepcopy(args), {}), info


def add_finalize_counts(info, result):
    vertices, contact_ids, report = result
    info.update(output_vertices=sum(map(len, vertices)),
                output_contact_member_count=sum(np.count_nonzero(ids >= 0).item() for ids in contact_ids),
                finalization_report=report)


@pytest.mark.parametrize("case", FINALIZE_CASES)
@pytest.mark.parametrize("size", [32, 256, 8192], ids=lambda value: f"n{value}")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_benchmark_finalize_cells(benchmark, case, size, dtype):
    if os.getenv("GEMPY_CONTACT_AWARE_BENCHMARKS") != "1":
        pytest.skip("Set GEMPY_CONTACT_AWARE_BENCHMARKS=1")
    target, setup, info = prepare_finalize_cells(case, size, dtype)
    original, _ = setup()
    result = benchmark.pedantic(
        target, setup=setup, rounds=int(os.getenv("GEMPY_CONTACT_BENCHMARK_ROUNDS", "5")),
        iterations=1, warmup_rounds=0,
    )
    vertices, ids, report = result
    assert report["contact_count"] == size
    assert report["dissolved_contact_count"] == 0
    assert report["unsupported_contact_member_count"] == (size if case == "compound_pruned" else 0)
    if case == "compound_supported":
        for before, after, before_ids, after_ids in zip(original[1], vertices, original[2], ids):
            np.testing.assert_array_equal(after, before)
            np.testing.assert_array_equal(after_ids, before_ids)
    else:
        expected = np.mean(np.stack(original[0][:2]).astype(np.float64), axis=0).astype(dtype)
        for after, after_ids in zip(vertices[:2], ids[:2]):
            np.testing.assert_array_equal(after, expected)
            np.testing.assert_array_equal(after_ids, np.arange(size))
        np.testing.assert_array_equal(vertices[2], original[0][2])
        assert np.all(ids[2] == -1)
    add_finalize_counts(info, result)
    benchmark.extra_info.update(info)
