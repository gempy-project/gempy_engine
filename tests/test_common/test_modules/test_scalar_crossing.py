import itertools

import numpy as np
import pytest

from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends
from gempy_engine.core.data.dual_contouring_data import DualContouringData
from gempy_engine.modules.dual_contouring._gen_vertices import generate_dual_contouring_vertices
from gempy_engine.modules.dual_contouring._scalar_crossing import scalar_crossing_parameters


def test_crossing_snaps_near_endpoints():
    crossing, parameters = scalar_crossing_parameters(
        np.array([-1e-14, -1.]), np.array([1., 1e-14]), 0.
    )
    np.testing.assert_array_equal(crossing, [True, True])
    np.testing.assert_array_equal(parameters, [0., 1.])
from gempy_engine.modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge


@pytest.fixture(params=[AvailableBackends.numpy, AvailableBackends.PYTORCH])
def backend(request):
    if request.param == AvailableBackends.PYTORCH:
        pytest.importorskip("torch")
    BackendTensor._change_backend(request.param, use_gpu=False)
    return BackendTensor.t


def as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value


def test_parameters(backend):
    start = backend.array([-1., 1., 0., 1., 0., -1., 0., 1., -1., -1e-14])
    end = backend.array([1., -1., 1., 0., -1., 0., 0., 2., -1., 3e-14])
    iso = backend.array(0.)
    valid, t = scalar_crossing_parameters(start, end, iso, xp=backend)
    np.testing.assert_array_equal(as_numpy(valid), [1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    np.testing.assert_allclose(as_numpy(t), [.5, .5, 0, 1, 0, 0, 0, 0, 0, .25])
    reverse_valid, reverse_t = scalar_crossing_parameters(end, start, iso, xp=backend)
    np.testing.assert_array_equal(as_numpy(reverse_valid), as_numpy(valid))
    np.testing.assert_allclose(as_numpy(reverse_t)[as_numpy(valid)], 1 - as_numpy(t)[as_numpy(valid)])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("argument", [0, 1, 2])
def test_nonfinite(backend, bad, argument):
    args = [backend.array([-1.]), backend.array([1.]), backend.array([0.])]
    args[argument] = backend.array([bad])
    with pytest.raises(ValueError, match="finite"):
        scalar_crossing_parameters(*args, xp=backend)


def test_numpy_utility_independent_of_backend(backend):
    valid, t = scalar_crossing_parameters(np.array([-1.]), np.array([3.]), np.array([0.]))
    assert isinstance(t, np.ndarray)
    np.testing.assert_array_equal(valid, [True])
    np.testing.assert_array_equal(t, [.25])


def test_intersections_masking_and_multiple_surfaces(backend):
    cube = np.array(list(itertools.product([0., 1.], repeat=3)))
    xyz = backend.array(np.concatenate([cube, cube + 2]))
    scalars = xyz[:, 0]
    iso = backend.array([0., .25, 1.])
    points, valid = find_intersection_on_edge(
        xyz, scalars, iso, masking=backend.array([True, False]), strict_crossings=True
    )
    expected_mask = np.zeros((3, 12), dtype=bool)
    expected_mask[:2, :4] = True
    np.testing.assert_array_equal(as_numpy(valid).reshape(3, 12), expected_mask)
    expected = np.concatenate([cube[:4], cube[:4] + [.25, 0, 0]])
    np.testing.assert_allclose(as_numpy(points), expected)
    assert valid.ndim == (1 if BackendTensor.engine_backend == AvailableBackends.PYTORCH else 2)


def test_empty_and_iso_edges(backend):
    xyz = backend.array(np.zeros((8, 3)))
    for masking in (None, backend.array([False])):
        points, valid = find_intersection_on_edge(
            xyz, xyz[:, 0], backend.array([0.]), masking=masking, strict_crossings=True
        )
        assert points.shape == (0, 3)
        assert not as_numpy(valid).any()


def test_default_retains_tolerant_crossings(backend):
    xyz = backend.array(np.array(list(itertools.product([0., 1.], repeat=3))))
    iso = backend.array([1.005])
    default = find_intersection_on_edge(xyz, xyz[:, 0], iso)
    disabled = find_intersection_on_edge(xyz, xyz[:, 0], iso, strict_crossings=False)
    for actual, expected in zip(default, disabled):
        np.testing.assert_array_equal(as_numpy(actual), as_numpy(expected))
    assert as_numpy(default[1]).any()
    strict = find_intersection_on_edge(xyz, xyz[:, 0], iso, strict_crossings=True)
    assert not as_numpy(strict[1]).any()


def test_mass_point_includes_zero_coordinates(backend):
    valid = np.zeros((2, 12), dtype=bool)
    valid[0, :2] = True
    data = DualContouringData(
        xyz_on_edge=backend.array([[0., 0., 0.], [2., 4., 0.]], dtype=BackendTensor.dtype_obj),
        valid_edges=backend.array(valid), xyz_on_centers=None, dxdydz=(1., 1., 1.),
        n_surfaces_to_export=1, left_right_codes=None,
        gradients=backend.array(np.zeros((2, 3)), dtype=BackendTensor.dtype_obj), strict_crossings=True
    )
    vertices = generate_dual_contouring_vertices(data, debug=True)
    np.testing.assert_allclose(as_numpy(data.bias_center_mass), [[1., 2., 0.]] * 3)
    assert np.isfinite(as_numpy(vertices)).all()
    assert DualContouringData.__dataclass_fields__["strict_crossings"].default is False
