from itertools import product

import numpy as np
import pytest

from gempy_engine.modules.dual_contouring.contact_planes import fit_contact_plane


@pytest.fixture
def points():
    return np.array(list(product((-1., 0., 1.), repeat=3)))


@pytest.mark.parametrize("gradient", [(0., 0., 1.), (2., -3., 6.), (-4., 2., -1.)])
@pytest.mark.parametrize("scale", [1e-150, 1., 1e150])
def test_independent_affine_normal_and_offset(points, gradient, scale):
    gradient = np.array(gradient)
    scalars = scale * (points @ gradient + 7.)
    normal, offset = fit_contact_plane(points, scalars, scale * 9.)
    expected_normal = gradient / np.linalg.norm(gradient)
    np.testing.assert_allclose(normal, expected_normal, atol=1e-14)
    assert offset == pytest.approx(2. / np.linalg.norm(gradient))
    assert np.linalg.norm(normal) == pytest.approx(1.)
    assert normal @ gradient > 0


@pytest.mark.parametrize("translation", [(17., -29., 31.), (1e12, -2e12, 3e12)])
def test_translated_small_support(points, translation):
    translation = np.array(translation)
    gradient = np.array([2., -3., 6.])
    normal, offset = fit_contact_plane(points + translation, points @ gradient + 7., 9.)
    np.testing.assert_allclose(normal, gradient / 7., atol=1e-14)
    assert abs(offset - (translation @ (gradient / 7.) + 2. / 7.)) < 0.002


def test_large_scalar_origin_and_coordinate_roundoff(points):
    points = points * 0.3 + np.array([1e12, -2e12, 3e12])
    gradient = np.array([2., -3., 6.])
    scalars = points @ gradient + 7.
    normal, offset = fit_contact_plane(points, scalars, np.mean(scalars))
    np.testing.assert_allclose(normal, gradient / 7., atol=0.002)
    assert np.isfinite(offset)


def test_anisotropic_coordinate_scale(points):
    points = points * [1e-9, 1., 1e9]
    gradient = np.array([2e9, -3., 6e-9])
    normal, offset = fit_contact_plane(points, points @ gradient + 7., 9.)
    np.testing.assert_allclose(normal, gradient / np.linalg.norm(gradient), atol=1e-14)
    assert offset == pytest.approx(2. / np.linalg.norm(gradient))


def test_four_point_full_rank_support():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    normal, offset = fit_contact_plane(points, points @ [2., -3., 6.] + 7., 9.)
    np.testing.assert_allclose(normal, np.array([2., -3., 6.]) / 7.)
    assert offset == pytest.approx(2. / 7.)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("target", ["points", "scalars", "isovalue", "tolerance"])
def test_nonfinite_inputs(points, bad, target):
    scalars = points[:, 2].copy()
    isovalue, tolerance = 0., 1e-8
    if target == "points":
        points[0, 0] = bad
    elif target == "scalars":
        scalars[0] = bad
    elif target == "isovalue":
        isovalue = bad
    else:
        tolerance = bad
    with pytest.raises(ValueError, match="finite"):
        fit_contact_plane(points, scalars, isovalue, tolerance)


@pytest.mark.parametrize("support", [
    np.zeros((4, 3)),
    np.array(list(product((-1., 1.), (-1., 1.), (0.,)))),
    np.array([[t, 2 * t, 3 * t] for t in range(5)]),
    np.array([[x, y, x + y] for x, y in product((-1., 0., 1.), repeat=2)]),
])
def test_degenerate_support(support):
    with pytest.raises(ValueError, match="rank"):
        fit_contact_plane(support, support[:, 0], 0.)


@pytest.mark.parametrize("shape", [(27, 1), (1, 27), (26,), ()])
def test_invalid_scalar_shape(points, shape):
    with pytest.raises(ValueError, match="scalar_values"):
        fit_contact_plane(points, np.zeros(shape), 0.)


@pytest.mark.parametrize("support", [[], np.zeros((3, 3)), np.zeros((4, 2)), np.zeros((4, 3, 1))])
def test_invalid_point_shape(support):
    with pytest.raises(ValueError, match="points"):
        fit_contact_plane(support, [], 0.)


@pytest.mark.parametrize("isovalue,tolerance", [([0.], 1e-8), (0., [1e-8]), (0., -1e-8)])
def test_invalid_scalar_arguments(points, isovalue, tolerance):
    with pytest.raises(ValueError):
        fit_contact_plane(points, points[:, 2], isovalue, tolerance)


def test_zero_gradient(points):
    with pytest.raises(ValueError, match="zero gradient"):
        fit_contact_plane(points, np.full(len(points), 7.), 7.)


@pytest.mark.parametrize("scale", [1e-150, 1., 1e150])
@pytest.mark.parametrize("translation", [0., 1e12])
def test_curved_scalar_fields_are_not_heuristic_planes(points, scale, translation):
    scalars = scale * (points @ [2., -3., 6.] + 0.1 * points[:, 0] ** 2)
    with pytest.raises(ValueError, match="non-affine"):
        fit_contact_plane(points + translation, scalars, 0.)


def test_tolerance_is_relative_to_scalar_range(points):
    scalars = points[:, 2] + 1e-6 * points[:, 0] ** 2
    with pytest.raises(ValueError, match="non-affine"):
        fit_contact_plane(points, scalars, 0.)
    normal, offset = fit_contact_plane(points, scalars, 0., tolerance=1e-5)
    np.testing.assert_allclose(normal, [0., 0., 1.], atol=1e-14)
    assert offset == pytest.approx(-2e-6 / 3)


def test_does_not_mutate_inputs_and_accepts_lists(points):
    scalars = points @ [2., -3., 6.] + 7.
    original_points, original_scalars = points.copy(), scalars.copy()
    fit_contact_plane(points, scalars, 9.)
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(scalars, original_scalars)
    normal, offset = fit_contact_plane(points.tolist(), scalars.tolist(), 9.)
    assert isinstance(normal, np.ndarray)
    assert isinstance(offset, float)
