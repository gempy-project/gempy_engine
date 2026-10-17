"""2:1 face/edge balancing of joint leaves before the adaptive cell complex."""

import importlib
from collections import Counter
from itertools import product

import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as BT
from gempy_engine.modules.dual_contouring.joint_balance import balance_joint_leaves, balance_leaves
from gempy_engine.modules.dual_contouring.joint_cell_branches import CORNERS
from gempy_engine.modules.dual_contouring.joint_cell_complex import build_adaptive_complex

from .test_joint_cell_complex import _random_octree

adapter = importlib.import_module("gempy_engine.API.dual_contouring.joint_extraction")


@pytest.fixture(autouse=True)
def cpu_float64():
    saved = dict(engine_backend=BT.engine_backend, use_gpu=BT.use_gpu,
                 use_pykeops=BT.use_pykeops, dtype=BT.dtype, grads=BT.COMPUTE_GRADS)
    BT._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype="float64", grads=False)
    yield
    BT._change_backend(**saved)


def _face_jump():
    """A span-4 leaf next to span-1 leaves (two levels across a face)."""
    fine = list(product(range(4, 8), range(4), range(4)))
    return np.array([(0, 0, 0)] + fine), np.array([4] + [1] * len(fine)), (8, 4, 4)


def _edge_jump():
    """Coarse and finest blocks meet only along an edge; face neighbours are span two."""
    origins, spans = [], []
    for x, y in product((0, 4), repeat=2):
        width = 4 if (x, y) == (0, 0) else 1 if (x, y) == (4, 4) else 2
        for origin in product(range(x, x + 4, width), range(y, y + 4, width), range(0, 4, width)):
            origins.append(origin)
            spans.append(width)
    return np.array(origins), np.array(spans), (8, 8, 4)


def _assert_partition(origins, spans, domain):
    covered = np.zeros(domain, dtype=int)
    for (x, y, z), s in zip(origins.tolist(), spans.tolist()):
        covered[x:x + s, y:y + s, z:z + s] += 1
    assert (covered == 1).all()


@pytest.mark.parametrize("case", [_face_jump, _edge_jump])
def test_balancing_splits_only_the_coarse_leaf(case):
    origins, spans, domain = case()
    with pytest.raises(ValueError, match='unsupported_unbalanced_octree'):
        build_adaptive_complex(origins, spans, domain)
    balanced, widths, source, splits = balance_leaves(origins, spans, domain)
    assert splits == 1 and len(widths) == len(spans) + 7
    _assert_partition(balanced, widths, domain)
    build_adaptive_complex(balanced, widths, domain)
    # Unsplit leaves keep their order and rows; children come last.
    kept = source >= 0
    assert kept[:kept.sum()].all()
    np.testing.assert_array_equal(balanced[kept], origins[source[kept]])


def test_corner_only_jump_is_left_alone():
    origins, spans = [], []
    for block in product((0, 4), repeat=3):
        width = 4 if block == (0, 0, 0) else 1 if block == (4, 4, 4) else 2
        for origin in product(*(range(value, value + 4, width) for value in block)):
            origins.append(origin)
            spans.append(width)
    balanced, widths, source, splits = balance_leaves(np.array(origins), np.array(spans), (8, 8, 8))
    assert splits == 0
    np.testing.assert_array_equal(source, np.arange(len(spans)))


@pytest.mark.parametrize('seed', range(100))
def test_random_octrees_balance_into_a_valid_complex(seed):
    origins, spans, domain = _random_octree(np.random.default_rng(seed))
    balanced, widths, _, _ = balance_leaves(origins, spans, domain)
    _assert_partition(balanced, widths, domain)
    build_adaptive_complex(balanced, widths, domain)


def test_new_corners_reuse_known_nodes_and_query_only_new_ones():
    origins, spans, domain = _face_jump()
    field = lambda nodes: np.stack([nodes @ [1., 2., 3.], nodes @ [-1., 0., .5]])
    corners = origins[:, None, :] + spans[:, None, None] * CORNERS
    samples = np.stack([field(c) for c in corners], axis=1)
    calls = []

    def sample_nodes(nodes):
        calls.append(nodes)
        return field(nodes)

    balanced, widths, balanced_samples, report = balance_joint_leaves(origins, spans, domain, samples, sample_nodes)
    expected = np.stack([field(c) for c in balanced[:, None, :] + widths[:, None, None] * CORNERS], axis=1)
    np.testing.assert_allclose(balanced_samples, expected)
    # The children's 27 nodes: the 8 coarse corners, 5 more x=4 face nodes the fine leaves own, 14 new.
    assert len(calls) == 1 and report == dict(split_leaves=1, added_leaves=7, queried_nodes=14)
    known = {tuple(n) for n in corners.reshape(-1, 3).tolist()}
    assert not any(tuple(n) in known for n in calls[0].tolist())


def test_balanced_input_is_returned_unchanged():
    origins = np.array(list(product(range(0, 4, 2), repeat=3)))
    spans = np.full(len(origins), 2)
    samples = np.zeros((1, len(spans), 8))
    result = balance_joint_leaves(origins, spans, (4, 4, 4), samples, lambda nodes: pytest.fail('no query'))
    assert result[0] is origins and result[1] is spans and result[2] is samples
    assert result[3] == dict(split_leaves=0, added_leaves=0, queried_nodes=0)


@pytest.mark.parametrize("with_null_space", [False, True])
def test_fast_refinement_model7_needs_and_gets_balancing(monkeypatch, with_null_space):
    """Production FAST octrees routinely jump two levels; joint_contacts balances them."""
    from gempy_engine.API.model.model_api import compute_model
    from .test_joint_contacts import _model7, _model7_null_space

    def model():
        return _model7_null_space(4, 3, 'joint_contacts')[0] if with_null_space else _model7(4, 3, 0, 'joint_contacts')

    meshes = compute_model(*model()).dc_meshes
    balance = meshes[0].contact_report['octree_balance']
    assert balance['split_leaves'] > 0 and balance['queried_nodes'] > 0
    for mesh in meshes:
        edges = Counter(tuple(sorted(e)) for tri in mesh.edges.tolist() for e in zip(tri, tri[1:] + tri[:1]))
        assert len(mesh.edges) and max(edges.values()) <= 2

    monkeypatch.setattr(adapter, "balance_joint_leaves",
                        lambda origins, spans, domain, samples, _: (origins, spans, samples, {}))
    with pytest.raises(ValueError, match='unsupported_unbalanced_octree'):
        compute_model(*model())
