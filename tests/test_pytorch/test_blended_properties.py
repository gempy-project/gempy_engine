"""Gravity and magnetics with ``GeophysicsInput.blend_properties`` (property mapping without rounded unit ids).

Two stacks: a flat erosion surface (ERODE) cutting a 30 degree dipping surface over basement, 9 gravity receivers
on top. With rounded-id mapping gravity carries no gradient with respect to the geometry. With blended properties
(and the soft erosion mask for the erosion contact) autograd must match a finite difference of the default model.
"""
import numpy as np
import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import SurfacePoints, Orientations, TensorsStructure
from gempy_engine.core.data.centered_grid import CenteredGrid
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.geophysics_input import GeophysicsInput, MagneticsInput
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.modules.geophysics.fw_magnetic import compute_tmi
from gempy_engine.modules.geophysics.gravity_gradient import calculate_gravity_gradient
from gempy_engine.modules.geophysics.property_mapping import interpolate_unit_property

torch = pytest.importorskip("torch")

SCALE = 1 / 1200  # a 1000 m model rescaled to [-5/12, 5/12]
RECEIVERS_X = (np.arange(100, 1000, 100.0) - 500) * SCALE
EROSION_POINTS, OLD_POINTS = slice(0, 4), slice(4, 8)
DENSITIES = np.array([2.0, 2.6, 3.1])
SLOPE = 100.0  # sigmoid_slope and erosion_sigmoid_slope for the differentiable model


@pytest.fixture
def torch_backend():
    previous = BackendTensor.engine_backend
    BackendTensor.change_backend_gempy(engine_backend=AvailableBackends.PYTORCH, use_gpu=False, dtype="float64",
                                       grads=True)
    yield
    BackendTensor.change_backend_gempy(engine_backend=previous, use_gpu=False)


def _model(blend, sigmoid_slope=5_000_000, erosion_sigmoid_slope=None, shift=None):
    sp = np.array([
        [-0.25, 0.00, 0.08333], [0.25, 0.00, 0.08333], [0.00, -0.25, 0.08333], [0.00, 0.25, 0.08333],  # erosion
        [-0.25, 0.00, -0.14417], [0.25, 0.00, 0.14417], [0.00, -0.25, 0.00], [0.00, 0.25, 0.00],  # old surface
    ])
    if shift is not None:
        rows, dz = shift
        sp[rows, 2] += dz
    receivers = np.c_[RECEIVERS_X, np.zeros_like(RECEIVERS_X), np.full_like(RECEIVERS_X, 5 / 12)]
    kernel = CenteredGrid(centers=receivers, resolution=np.array([10, 10, 200]), radius=np.full(3, 1000 * SCALE))
    grid = EngineGrid(octree_grid=RegularGrid([-5 / 12, 5 / 12] * 3, np.array([2, 2, 2])), geophysics_grid=kernel)
    interpolation_input = InterpolationInput(
        surface_points=SurfacePoints(sp),
        orientations=Orientations(np.array([[0.0, 0.0, 0.08333], [0.0, 0.0, 0.0]]),
                                  np.array([[0.0, 0.0, 1.0], [-0.5, 0.0, 0.86603]])),
        grid=grid,
        unit_values=np.array([1, 2, 3])
    )
    descriptor = InputDataDescriptor(
        tensors_structure=TensorsStructure(number_of_points_per_surface=np.array([4, 4])),
        stack_structure=StacksStructure(
            number_of_points_per_stack=np.array([4, 4]),
            number_of_orientations_per_stack=np.array([1, 1]),
            number_of_surfaces_per_stack=np.array([1, 1]),
            masking_descriptor=[StackRelationType.ERODE, StackRelationType.BASEMENT]
        )
    )
    options = InterpolationOptions.from_args(range=1.7, c_o=10., uni_degree=1, mesh_extraction=False,
                                             number_octree_levels=1)
    options.cache_mode = InterpolationOptions.CacheMode.NO_CACHE
    options.sigmoid_slope = sigmoid_slope
    options.erosion_sigmoid_slope = erosion_sigmoid_slope
    geophysics = GeophysicsInput(tz=calculate_gravity_gradient(kernel), densities=DENSITIES, blend_properties=blend)
    return interpolation_input, options, descriptor, geophysics


def _solve(blend, **kwargs):
    interpolation_input, options, descriptor, geophysics = _model(blend, **kwargs)
    solutions = compute_model(interpolation_input, options, descriptor, geophysics_input=geophysics)
    return interpolation_input, geophysics, solutions


def _hard_gravity_derivative(rows, h=30 * SCALE):
    """Central finite difference of the default model (rounded ids, hard masks), numpy backend."""
    previous = BackendTensor.engine_backend
    BackendTensor.change_backend_gempy(engine_backend=AvailableBackends.numpy, use_gpu=False)
    try:
        g = [np.asarray(_solve(False, shift=(rows, dz))[2].gravity) for dz in (h, -h)]
    finally:
        BackendTensor.change_backend_gempy(engine_backend=previous, use_gpu=False, dtype="float64", grads=True)
    return (g[0] - g[1]) / (2 * h)


def _autograd_derivative(rows, **kwargs):
    """d(gravity at each receiver)/d(z of the points in rows, moved together)."""
    interpolation_input, _, solutions = _solve(**kwargs)
    gravity, sp = solutions.gravity, interpolation_input.surface_points.sp_coords
    if not gravity.requires_grad:
        return np.zeros(len(RECEIVERS_X))
    out = []
    for k in range(len(RECEIVERS_X)):
        sp.grad = None
        gravity[k].backward(retain_graph=True)
        out.append(0.0 if sp.grad is None else sp.grad[rows, 2].sum().item())
    return np.array(out)


def _relative_rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2)) / np.abs(b).max()


def test_interpolate_unit_property():
    ids = BackendTensor.t.array(np.array([1.0, 1.5, 2.0, 2.25, 3.0, 0.5, 3.7]))  # inputs are backend tensors
    result = interpolate_unit_property(ids, BackendTensor.t.array(DENSITIES))
    np.testing.assert_allclose(BackendTensor.t.to_numpy(result), [2.0, 2.3, 2.6, 2.725, 3.1, 2.0, 3.1])


def test_rounded_ids_give_no_geometry_gradient(torch_backend):
    np.testing.assert_array_equal(_autograd_derivative(OLD_POINTS, blend=False), 0)


@pytest.mark.parametrize("rows", [EROSION_POINTS, OLD_POINTS], ids=["erosion surface", "surface inside a stack"])
def test_blended_gravity_gradient_matches_finite_difference(torch_backend, rows):
    expected = _hard_gravity_derivative(rows)
    assert np.abs(expected).min() > 0  # every receiver sees the surface
    ad = _autograd_derivative(rows, blend=True, sigmoid_slope=SLOPE, erosion_sigmoid_slope=SLOPE)
    assert _relative_rmse(ad, expected) < 0.1


def test_blend_without_soft_erosion_mask_handles_surfaces_inside_stacks(torch_backend):
    expected = _hard_gravity_derivative(OLD_POINTS)
    ad = _autograd_derivative(OLD_POINTS, blend=True, sigmoid_slope=SLOPE)
    assert _relative_rmse(ad, expected) < 0.1


def test_blended_mapping_equals_rounded_ids_for_a_sharp_model():
    _, _, rounded = _solve(False)
    _, geophysics, blended = _solve(True)
    np.testing.assert_allclose(np.asarray(blended.gravity), np.asarray(rounded.gravity), rtol=1e-6)

    stack_outputs = blended.octrees_output[0].outputs_centers
    magnetics = MagneticsInput(mag_kernel=np.asarray(geophysics.tz), susceptibilities=np.array([0.01, 0.02, 0.05]),
                               igrf_params={})
    tmi = [compute_tmi(magnetics, stack_outputs[-1], stack_outputs=stack_outputs, blend_properties=b)
           for b in (False, True)]
    np.testing.assert_allclose(np.asarray(tmi[1]), np.asarray(tmi[0]), rtol=1e-6)
