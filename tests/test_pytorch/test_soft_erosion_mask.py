"""Gradients across erosion contacts (``InterpolationOptions.erosion_sigmoid_slope``).

Two stacks: a flat erosion surface (ERODE) cutting a tilted surface (BASEMENT). The quantity of interest is the sum of
a density-like value along a vertical line through the erosion contact. Moving the erosion surface up by dz turns
dz worth of line points from the young unit into the old one, so the derivative is the density contrast times the
point density of the line. With the default hard erosion mask autograd sees none of that; with the soft mask it must
match a finite difference of the hard model.
"""
import numpy as np
import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import SurfacePoints, Orientations, TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.generic_grid import GenericGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure

torch = pytest.importorskip("torch")

SPACING = 1 / 1200  # 1 m in a 1000 m model rescaled by 1/1200
LINE_Z = np.arange(-0.4163, 0.4166, SPACING)
EROSION_POINTS = slice(0, 4)
DENSITIES = (2.0, 2.6, 3.1)  # unit ids 1 (young), 2 (old), 3 (basement)


@pytest.fixture
def torch_backend():
    previous = BackendTensor.engine_backend
    BackendTensor.change_backend_gempy(engine_backend=AvailableBackends.PYTORCH, use_gpu=False, dtype="float64",
                                       grads=True)
    yield
    BackendTensor.change_backend_gempy(engine_backend=previous, use_gpu=False)


def _model(erosion_sigmoid_slope, erosion_dz=0.0):
    sp = np.array([
        [-0.25, 0.00, 0.08333],  # erosion surface (flat)
        [0.25, 0.00, 0.08333],
        [0.00, -0.25, 0.08333],
        [0.00, 0.25, 0.08333],
        [-0.25, 0.00, -0.14417],  # old surface (dips 30 degrees)
        [0.25, 0.00, 0.14417],
        [0.00, -0.25, 0.00],
        [0.00, 0.25, 0.00],
    ])
    sp[EROSION_POINTS, 2] += erosion_dz
    dip_positions = np.array([[0.0, 0.0, 0.08333], [0.0, 0.0, 0.0]])
    dip_gradients = np.array([[0.0, 0.0, 1.0], [-0.5, 0.0, 0.86603]])

    line = np.c_[np.full_like(LINE_Z, 0.125), np.zeros_like(LINE_Z), LINE_Z]
    grid = EngineGrid(
        octree_grid=RegularGrid([-0.41666667, 0.41666667] * 3, np.array([2, 2, 2])),
        custom_grid=GenericGrid(values=line)
    )
    interpolation_input = InterpolationInput(
        surface_points=SurfacePoints(sp),
        orientations=Orientations(dip_positions, dip_gradients),
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
    options.erosion_sigmoid_slope = erosion_sigmoid_slope
    return interpolation_input, options, descriptor


def _line_output(erosion_sigmoid_slope, erosion_dz=0.0):
    interpolation_input, options, descriptor = _model(erosion_sigmoid_slope, erosion_dz)
    solutions = compute_model(interpolation_input, options, descriptor)
    output = solutions.octrees_output[0].last_output_center
    return interpolation_input, output


def _mass(ids):
    """Piecewise-linear id -> density, summed along the line (differentiable in soft ids)."""
    d1, d2, d3 = DENSITIES
    one = ids * 0 + 1
    return (d1 * one + (ids - 1).clip(0, 1) * (d2 - d1) + (ids - 2).clip(0, 1) * (d3 - d2)).sum()


def _hard_mass_derivative(h=10 * SPACING):
    """Central finite difference of the default (hard-masked) model, numpy backend."""
    previous = BackendTensor.engine_backend
    BackendTensor.change_backend_gempy(engine_backend=AvailableBackends.numpy, use_gpu=False)
    try:
        masses = [_mass(_line_output(None, dz)[1].final_block[_custom(dz)]) for dz in (h, -h)]
    finally:
        BackendTensor.change_backend_gempy(engine_backend=previous, use_gpu=False, dtype="float64", grads=True)
    return float(masses[0] - masses[1]) / (2 * h)


def _custom(_dz=0.0):
    return slice(8, 8 + LINE_Z.size)  # octree centers (2x2x2) come first, then the custom grid


def _erosion_z_gradient(erosion_sigmoid_slope):
    interpolation_input, output = _line_output(erosion_sigmoid_slope)
    assert output.grid.custom_grid_slice == _custom()
    _mass(output.final_block[_custom()]).backward()
    return interpolation_input.surface_points.sp_coords.grad[EROSION_POINTS, 2].sum().item(), output


def test_hard_mask_gives_no_gradient_at_erosion_contact(torch_backend):
    gradient, output = _erosion_z_gradient(None)
    assert gradient == pytest.approx(0, abs=1e-6)
    assert output.squeezed_mask_array.dtype == torch.bool


# The sigmoid must be wider than the point spacing (1/1200 here) to be resolved: ~4/slope >= spacing.
@pytest.mark.parametrize("erosion_sigmoid_slope", [200.0, 1000.0])
def test_soft_mask_gradient_matches_finite_difference(torch_backend, erosion_sigmoid_slope):
    expected = _hard_mass_derivative()
    assert expected == pytest.approx((DENSITIES[1] - DENSITIES[0]) / SPACING, rel=0.02)  # sanity: contrast x density

    gradient, output = _erosion_z_gradient(erosion_sigmoid_slope)
    assert gradient == pytest.approx(expected, rel=0.02)
    # Meshing and null space read the squeezed masks: they must stay boolean.
    assert output.squeezed_mask_array.dtype == torch.bool


def test_soft_mask_leaves_values_away_from_contact_unchanged(torch_backend):
    _, hard = _line_output(None)
    _, soft = _line_output(1000.0)
    hard_ids = hard.final_block[_custom()].detach().numpy()
    soft_ids = soft.final_block[_custom()].detach().numpy()
    contact = np.argmin(np.abs(LINE_Z - 0.08333))
    far = np.abs(np.arange(LINE_Z.size) - contact) > 30  # 30 m away from the erosion contact
    np.testing.assert_allclose(soft_ids[far], hard_ids[far], atol=1e-3)
