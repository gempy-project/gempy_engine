import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.dual_contouring_data import DualContouringData
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.modules.dual_contouring._gen_vertices import generate_dual_contouring_vertices
from gempy_engine.modules.octrees_topology._octree_common import _generate_next_level_centers


@pytest.fixture(params=['numpy', 'PYTORCH'])
def backend(request):
    if request.param == 'PYTORCH':
        pytest.importorskip('torch')
    old = BackendTensor.engine_backend, BackendTensor.use_gpu, BackendTensor.dtype, BackendTensor.use_pykeops
    BackendTensor._change_backend(engine_backend=AvailableBackends[request.param], use_gpu=False, dtype='float64')
    yield BackendTensor.t
    BackendTensor._change_backend(engine_backend=old[0], use_gpu=old[1], dtype=old[2], use_pykeops=old[3])


def test_grid_extent_and_centers_are_not_translated(backend):
    t = backend
    extent = [-1., 1., -2., 2., -3., 3.]
    dense = RegularGrid(extent, [3, 3, 3])
    np.testing.assert_array_equal(t.to_numpy(dense.values[13]), [0., 0., 0.])
    grid = EngineGrid.from_regular_grid(dense)
    assert grid.dense_grid is dense
    root = grid.octree_grid
    np.testing.assert_array_equal(t.to_numpy(root.orthogonal_extent), extent)
    np.testing.assert_array_equal(t.to_numpy(dense.orthogonal_extent), extent)
    np.testing.assert_array_equal(t.to_numpy(root.values[0]), [-0.5, -1., -1.5])
    for _ in range(3):
        active = root.integer_coordinates[:, 0] == 0
        xyz, bits = _generate_next_level_centers(root.values[active], root.dxdydz)
        root = RegularGrid.from_octree_level(xyz, root, active, bits)
        np.testing.assert_array_equal(t.to_numpy(root.orthogonal_extent), extent)
        expected = root.orthogonal_extent[::2] + (root.integer_coordinates + 0.5) * (
            (root.orthogonal_extent[1::2] - root.orthogonal_extent[::2]) / root.regular_grid_shape
        )
        np.testing.assert_array_equal(t.to_numpy(root.values), t.to_numpy(expected))


@pytest.mark.parametrize('strict_crossings', [False, True])
@pytest.mark.parametrize('points', [
    [[0., 0., 0.], [2., 0., 4.]],
    [[0., 0., 0.], [0., 0., 0.]],
])
def test_mass_points_include_zero_coordinates(backend, strict_crossings, points):
    t = backend
    valid = t.zeros((2, 12), dtype=bool)
    valid[1, 0] = valid[1, 5] = True
    data = DualContouringData(
        xyz_on_edge=t.array(points, dtype=BackendTensor.dtype_obj), valid_edges=valid,
        xyz_on_centers=t.zeros((2, 3)), dxdydz=(1., 1., 1.),
        n_surfaces_to_export=1, left_right_codes=None,
        gradients=t.array([[0., 1., 0.], [0., 1., 0.]], dtype=BackendTensor.dtype_obj),
        strict_crossings=strict_crossings,
    )
    vertices = generate_dual_contouring_vertices(data, debug=True)
    expected = np.mean(points, axis=0)
    np.testing.assert_array_equal(t.to_numpy(data.bias_center_mass), np.tile(expected, (3, 1)))
    # Preserve the existing non-strict PyTorch origin regularization.
    if BackendTensor.engine_backend == AvailableBackends.PYTORCH and not strict_crossings:
        expected = expected / 1.0001
    np.testing.assert_allclose(t.to_numpy(vertices), expected[None, :], atol=1e-15)
