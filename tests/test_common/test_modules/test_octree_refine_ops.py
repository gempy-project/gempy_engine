"""The refinement and corner ops agree exactly with the per-axis formulations they replace."""
import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.interp_output import _distinct_count
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.modules.octrees_topology._octree_common import _cell_children, _generate_next_level_centers
from gempy_engine.modules.octrees_topology._octree_internals import (
    _active_index, _host_counts, _mark_voxel, _marked_voxels)


@pytest.fixture(params=["numpy", "pytorch"])
def backend(request, restore_backend):
    if request.param == "pytorch":
        pytest.importorskip("torch")
        BackendTensor._change_backend(AvailableBackends.PYTORCH)
    else:
        BackendTensor._change_backend(AvailableBackends.numpy)
    return request.param


def _reference_children(xyz, dxdydz, level, divisor):
    """The per-axis left/right stacks, repeats and tiles of the former implementation."""
    t = BackendTensor.tfnp

    def stack_left_right(a, d):
        return t.stack([a - d / level / divisor, a + d / level / divisor], axis=1)

    x = t.ravel(t.repeat(stack_left_right(xyz[:, 0], dxdydz[0]), 4, axis=1))
    y = t.ravel(t.tile(t.repeat(stack_left_right(xyz[:, 1], dxdydz[1]), 2, axis=1), (1, 2)))
    z = t.ravel(t.tile(stack_left_right(xyz[:, 2], dxdydz[2]), (1, 4)))
    return t.stack([x, y, z], axis=1)


def _equal(a, b):
    a, b = BackendTensor.t.to_numpy(a), BackendTensor.t.to_numpy(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("divisor", [2, 4])
def test_cell_children_match_per_axis_construction(backend, dtype, divisor):
    rng = np.random.default_rng(divisor)
    grid = RegularGrid(np.array([-0.3, 0.7, 0.1, 0.45, -1.0, 2.0]), np.array([4, 6, 2]))
    xyz = BackendTensor.t.array(rng.uniform(-1, 1, size=(50, 3)), dtype=dtype)
    expected = _reference_children(xyz, grid.dxdydz, 1, divisor)
    assert _equal(_cell_children(xyz, grid.dxdydz, 1, divisor), expected)

    centers, sides = _generate_next_level_centers(xyz, grid.dxdydz)
    assert _equal(centers, _reference_children(xyz, grid.dxdydz, 1, 4))
    left_right = BackendTensor.t.to_numpy(sides)
    assert left_right.dtype == bool
    np.testing.assert_array_equal(left_right[:8], [[k >> 2 & 1, k >> 1 & 1, k & 1] for k in range(8)])
    np.testing.assert_array_equal(left_right, np.tile(left_right[:8], (50, 1)))


def test_marked_voxels_match_edge_marking(backend):
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 3, size=(400, 8)).astype(float)
    ids[::3] = ids[::3, :1]  # uniform cells
    ids[1::7, 5] = np.nan
    uv_8 = BackendTensor.t.array(ids)
    assert _equal(_marked_voxels(uv_8), _mark_voxel(uv_8)[1])


def test_distinct_count_matches_unique(backend):
    rng = np.random.default_rng(1)
    for values in (np.rint(rng.uniform(0, 9, size=2000)), np.zeros(0), np.ones(5)):
        values = BackendTensor.t.array(values, dtype="float32")
        assert int(_distinct_count(values)) == len(BackendTensor.t.unique(values))


def test_host_counts_and_active_index(backend):
    rng = np.random.default_rng(2)
    masks = [BackendTensor.t.array(rng.uniform(size=300) < p) for p in (0.0, 0.3, 1.0)]
    counts = _host_counts(*masks)
    assert counts == [int(BackendTensor.t.to_numpy(mask).sum()) for mask in masks]
    for mask, count in zip(masks, counts):
        expected = np.flatnonzero(BackendTensor.t.to_numpy(mask))
        np.testing.assert_array_equal(BackendTensor.t.to_numpy(_active_index(mask)), expected)
        np.testing.assert_array_equal(BackendTensor.t.to_numpy(_active_index(mask, count)), expected)

