from itertools import product

import numpy as np
import pytest

from gempy_engine.modules.dual_contouring.adaptive_cell_complex import build_adaptive_complex
from gempy_engine.modules.dual_contouring.topology_extraction import INCIDENT_OFFSETS


def _uniform(shape, span=1):
    origins = np.array(list(product(*(range(0, size, span) for size in shape))), dtype=int)
    return origins, np.full(len(origins), span, dtype=int)


def _mixed():
    fine = list(product(range(2, 4), range(2), range(2)))
    return np.array([(0, 0, 0)] + fine), np.array([2] + [1] * len(fine)), (4, 2, 2)


def _assert_face_coverage(complex_, origins, spans):
    # Tiny test-only rasterization checks every leaf face has exactly one tile per pixel.
    for cell, (origin, span) in enumerate(zip(origins, spans)):
        for axis in range(3):
            other = [a for a in range(3) if a != axis]
            for side in (0, 1):
                pixels = np.zeros((span, span), dtype=int)
                for face in complex_['faces']:
                    if (face['axis'] != axis or cell not in face['cells'] or
                            face['origin'][axis] != origin[axis] + side * span):
                        continue
                    u, v = [face['origin'][a] - origin[a] for a in other]
                    size = face['span']
                    assert 0 <= u <= span - size and 0 <= v <= span - size
                    pixels[u:u + size, v:v + size] += 1
                assert np.all(pixels == 1)


@pytest.mark.parametrize('n', [2, 4])
def test_uniform_cartesian_incidence(n):
    origins, spans = _uniform((n, n, n))
    result = build_adaptive_complex(origins, spans, (n, n, n))
    assert set(result) == {'faces', 'edges'}
    assert len(result['faces']) == 3 * (n + 1) * n ** 2
    assert len(result['edges']) == 3 * n * (n + 1) ** 2
    lookup = {tuple(origin): i for i, origin in enumerate(origins)}
    for edge in result['edges']:
        expected = tuple(lookup.get(tuple(np.array(edge['origin']) + offset))
                         for offset in INCIDENT_OFFSETS[edge['axis']])
        assert edge['cells'] == expected
        assert edge['span'] == 1
    for face in result['faces']:
        lower = np.array(face['origin'])
        negative = lower.copy()
        negative[face['axis']] -= 1
        neighbors = (lookup.get(tuple(negative)), lookup.get(tuple(lower)))
        expected = neighbors if None not in neighbors else (next(c for c in neighbors if c is not None), None)
        assert face['cells'] == expected
    _assert_face_coverage(result, origins, spans)


def test_mixed_face_partition_and_hanging_rings():
    origins, spans, domain = _mixed()
    result = build_adaptive_complex(origins, spans, domain)
    shared = [face for face in result['faces'] if face['axis'] == 0 and face['origin'][0] == 2]
    assert len(shared) == 4
    assert {face['origin'] for face in shared} == {(2, y, z) for y, z in product(range(2), repeat=2)}
    assert all(face['span'] == 1 and face['cells'][0] == 0 for face in shared)
    hanging = next(edge for edge in result['edges'] if edge['axis'] == 1 and edge['origin'] == (2, 0, 1))
    assert hanging['span'] == 1
    assert hanging['cells'] == (0, 0, 2, 1)
    _assert_face_coverage(result, origins, spans)


def test_segments_are_exact_original_edge_union():
    origins, spans, domain = _mixed()
    result = build_adaptive_complex(origins, spans, domain)
    lines = {}
    for origin, span in zip(origins, spans):
        for axis in range(3):
            other = [a for a in range(3) if a != axis]
            for offsets in product((0, span), repeat=2):
                transverse = tuple(int(origin[a] + offset) for a, offset in zip(other, offsets))
                lines.setdefault((axis, transverse), []).append((int(origin[axis]), int(origin[axis] + span)))
    expected = set()
    for (axis, transverse), intervals in lines.items():
        endpoints = sorted({value for interval in intervals for value in interval})
        for start, end in zip(endpoints, endpoints[1:]):
            if not any(a <= start and end <= b for a, b in intervals):
                continue
            lower = list(transverse)
            lower.insert(axis, start)
            expected.add((axis, tuple(lower), end - start))
    assert {(edge['axis'], edge['origin'], edge['span']) for edge in result['edges']} == expected
    # Independently resolve quadrants directly against leaf bounding boxes.
    for edge in result['edges']:
        axis = edge['axis']
        midpoint = np.array(edge['origin'], dtype=float)
        midpoint[axis] += edge['span'] / 2
        for offset, cell in zip(INCIDENT_OFFSETS[axis], edge['cells']):
            probe = midpoint.copy()
            for a in range(3):
                if a != axis:
                    probe[a] += 0.25 if offset[a] == 0 else -0.25
            matches = np.flatnonzero(np.all((origins <= probe) & (probe < origins + spans[:, None]), axis=1))
            assert len(matches) <= 1
            assert cell == (int(matches[0]) if len(matches) else None)


def test_deterministic_order_and_input_row_ids():
    origins, spans, domain = _mixed()
    original = build_adaptive_complex(origins, spans, domain)
    permutation = np.arange(len(spans))[::-1]
    shuffled = build_adaptive_complex(origins[permutation], spans[permutation], domain)
    for name in ('faces', 'edges'):
        keys = [(r['axis'], r['origin'], r['span']) for r in original[name]]
        assert keys == sorted(set(keys))
        for before, after in zip(original[name], shuffled[name]):
            assert {k: v for k, v in before.items() if k != 'cells'} == {k: v for k, v in after.items() if k != 'cells'}
            assert before['cells'] == tuple(None if c is None else int(permutation[c]) for c in after['cells'])


@pytest.mark.parametrize('origins,spans,domain,error', [
    ([(0, 0, 0)], [1], (2, 1, 1), 'incomplete_coverage'),
    ([(0, 0, 0), (0, 0, 0)], [1, 1], (2, 1, 1), 'overlapping_leaves'),
    ([(0, 0, 0), (0, 0, 0)], [2, 1], (3, 3, 1), 'outside_domain'),
    ([(0, 0, 0), (0, 0, 0)], [2, 1], (3, 3, 3), 'overlapping_leaves'),
    ([(0, 0, 0)], [3], (3, 3, 3), 'invalid_span'),
    ([(1, 0, 0)], [2], (4, 2, 2), 'unaligned_origin'),
    ([(0, 0, 0)], [1], (0, 1, 1), 'invalid_domain'),
    ([(0., 0., 0.)], [1], (1, 1, 1), 'invalid_integer_input'),
    ([(0, 0, 0)], [1.5], (1, 1, 1), 'invalid_integer_input'),
    ([(0, 0)], [1], (1, 1, 1), 'invalid_shape'),
])
def test_invalid_inputs(origins, spans, domain, error):
    with pytest.raises(ValueError, match=error):
        build_adaptive_complex(origins, spans, domain)


def test_unbalanced_face_rejected():
    fine = list(product(range(4, 8), range(4), range(4)))
    with pytest.raises(ValueError, match='unsupported_unbalanced_octree: face'):
        build_adaptive_complex(np.array([(0, 0, 0)] + fine), np.array([4] + [1] * len(fine)), (8, 4, 4))


def test_edge_only_unbalance_rejected():
    # Coarse and finest blocks meet only at x=y=4; face neighbors are span two.
    origins, spans = [], []
    for x, y in product((0, 4), repeat=2):
        width = 4 if (x, y) == (0, 0) else 1 if (x, y) == (4, 4) else 2
        for origin in product(range(x, x + 4, width), range(y, y + 4, width), range(0, 4, width)):
            origins.append(origin)
            spans.append(width)
    with pytest.raises(ValueError, match='unsupported_unbalanced_octree: edge'):
        build_adaptive_complex(np.array(origins), np.array(spans), (8, 8, 4))


def test_corner_only_depth_jump_is_supported():
    origins, spans = [], []
    for block in product((0, 4), repeat=3):
        width = 4 if block == (0, 0, 0) else 1 if block == (4, 4, 4) else 2
        for origin in product(*(range(value, value + 4, width) for value in block)):
            origins.append(origin)
            spans.append(width)
    origins, spans = np.array(origins), np.array(spans)
    result = build_adaptive_complex(origins, spans, (8, 8, 8))
    _assert_face_coverage(result, origins, spans)


def test_large_sparse_domain_and_exact_integer_probes():
    width = 2 ** 55
    origins, spans = _uniform((3 * width, width, width), width)
    result = build_adaptive_complex(origins, spans, (3 * width, width, width))
    assert len(result['faces']) == 16
    assert len(result['edges']) == 28
    assert all(edge['span'] == width for edge in result['edges'])
