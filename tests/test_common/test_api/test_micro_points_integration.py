import numpy as np
import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import InterpolationOptions, Orientations, SurfacePoints, TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid, RegularGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput, MicroPoints
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.config import AvailableBackends
from gempy_engine.modules.evaluator.micro_anisotropic_evaluator import (
    build_micro_design_matrix, evaluate_micro_correction, evaluate_micro_gradient,
)


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("gradient", [False, True])
def test_authored_micro_contacts_are_stack_local(monkeypatch, flat, gradient):
    if flat:
        BackendTensor._change_backend(AvailableBackends.numpy, use_pykeops=True)
        monkeypatch.setenv("GEMPY_FLAT_STACKS", "True")
    try:
        _assert_authored_micro_contacts_are_stack_local(monkeypatch, gradient)
    finally:
        if flat:
            BackendTensor._change_backend(AvailableBackends.numpy)


def _assert_authored_micro_contacts_are_stack_local(monkeypatch, gradient):
    sp = np.array([[0.2, 0.2, 0.4], [0.8, 0.2, 0.4], [0.2, 0.8, 0.4],
                   [0.8, 0.8, 0.4]])
    points = np.vstack((sp, sp + [0, 0, 0.2]))
    orientations = Orientations(dip_positions=np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.7]]),
                                dip_gradients=np.array([[0, 0, 1], [0, 0, 1]]))
    grid = EngineGrid.from_regular_grid(RegularGrid(
        orthogonal_extent=[0, 1, 0, 1, 0, 1], regular_grid_shape=[3, 3, 3]))
    contact = np.array([[0.5, 0.5, 0.52]])
    micro = MicroPoints(contact, np.array([np.diag([2., 2., 2.])]),
                        np.array([0.0]), np.array([0]))
    ii = InterpolationInput(SurfacePoints(points), orientations, grid, micro_points=micro)
    options = InterpolationOptions.from_args(range=3., c_o=1., uni_degree=0,
                                             mesh_extraction=False)
    options.evaluation_options.compute_scalar_gradient = gradient
    descriptor = InputDataDescriptor(TensorsStructure(np.array([4, 4])), StacksStructure(
        number_of_points_per_stack=np.array([4, 4]),
        number_of_orientations_per_stack=np.array([1, 1]),
        number_of_surfaces_per_stack=np.array([1, 1]),
        masking_descriptor=[StackRelationType.ERODE, StackRelationType.ERODE]))

    def fields():
        result = compute_model(ii, options, descriptor)
        if BackendTensor.engine_backend is AvailableBackends.PYTORCH:
            for output in result.octrees_output[0].outputs:
                field = output.scalar_fields.exported_fields.scalar_field_everywhere
                assert field.device == ii.surface_points.sp_coords.device
                assert field.dtype == ii.surface_points.sp_coords.dtype
        return [BackendTensor.t.to_numpy(output.scalar_fields.exported_fields.scalar_field_everywhere).copy()
                for output in result.octrees_output[0].outputs]

    with pytest.warns(UserWarning, match="Stack 0 contains micro points.*ignored") as recorded:
        baseline = fields()
    assert len(recorded) == 1
    options.micro_options.enabled = True
    monkeypatch.setenv("GEMPY_FLAT_STACKS", "True")
    corrected = fields()
    if BackendTensor.use_pykeops:
        monkeypatch.setenv("GEMPY_FLAT_STACKS", "False")
        sequential = fields()
        for flat_field, sequential_field in zip(corrected, sequential):
            np.testing.assert_allclose(flat_field, sequential_field, atol=1e-5, rtol=1e-5)
    assert np.max(np.abs(corrected[0] - baseline[0])) > 1e-6
    np.testing.assert_allclose(corrected[1], baseline[1], atol=1e-8)
    grid_size = grid.len_all_grids
    np.testing.assert_allclose(corrected[0][grid_size:grid_size + 4],
                               baseline[0][grid_size:grid_size + 4], atol=1e-6)
    assert not hasattr(options.micro_options, "weights")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_authored_fit_preserves_autograd(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    BackendTensor._change_backend(AvailableBackends.PYTORCH, use_gpu=device == "cuda", grads=True)
    try:
        sp = torch.tensor([[0.2, 0.2, 0.4], [0.8, 0.2, 0.4],
                           [0.2, 0.8, 0.4], [0.8, 0.8, 0.4]], dtype=torch.float32, requires_grad=True)
        contact = torch.tensor([[0.5, 0.5, 0.52]], dtype=torch.float32, requires_grad=True)
        metric = torch.diag(torch.tensor([2., 2., 2.], dtype=torch.float32)).unsqueeze(0).requires_grad_()
        nugget = torch.tensor([0.01], dtype=torch.float32, requires_grad=True)
        ii = InterpolationInput(SurfacePoints(sp), Orientations(
            dip_positions=np.array([[0.5, 0.5, 0.5]]), dip_gradients=np.array([[0., 0., 1.]])),
            EngineGrid.from_regular_grid(RegularGrid(
                orthogonal_extent=[0, 1, 0, 1, 0, 1], regular_grid_shape=[3, 3, 3])),
            micro_points=MicroPoints(contact, metric, nugget, np.array([0])))
        options = InterpolationOptions.from_args(range=3., c_o=1., uni_degree=0, mesh_extraction=False)
        options.micro_options.enabled = True
        options.evaluation_options.compute_scalar_gradient = True
        descriptor = InputDataDescriptor(TensorsStructure(np.array([4])), StacksStructure(
            np.array([4]), np.array([1]), np.array([1]), [StackRelationType.ERODE]))
        result = compute_model(ii, options, descriptor)
        field = result.octrees_output[0].outputs[0].scalar_fields.exported_fields.scalar_field_everywhere
        assert field.requires_grad
        grads = torch.autograd.grad(field[:ii.grid.len_all_grids].sum(), (sp, contact, metric, nugget))
        for grad in grads:
            assert torch.isfinite(grad).all() and grad.abs().max() > 1e-9
        assert ii.micro_points.points is contact
        assert not hasattr(options.micro_options, "weights")
    finally:
        torch.set_default_device("cpu")
        BackendTensor._change_backend(AvailableBackends.numpy)


def test_micro_points_reject_invalid_surface_index():
    with pytest.raises(ValueError, match="surface_indices"):
        MicroPoints(np.zeros((1, 3)), np.eye(3)[None], np.zeros(1), np.array([0.5]))


def test_unknown_global_surface_index_with_stack_overrides():
    points = np.array([[0.2, 0.2, 0.4], [0.8, 0.2, 0.4],
                       [0.2, 0.8, 0.4], [0.8, 0.8, 0.4]])
    ii = InterpolationInput(
        SurfacePoints(points),
        Orientations(np.array([[0.5, 0.5, 0.5]]), np.array([[0., 0., 1.]])),
        EngineGrid.from_regular_grid(RegularGrid(
            orthogonal_extent=[0, 1, 0, 1, 0, 1], regular_grid_shape=[2, 2, 2])),
        micro_points=MicroPoints(np.array([[0.5, 0.5, 0.52]]), np.eye(3)[None],
                                 np.zeros(1), np.array([1])),
    )
    options = InterpolationOptions.from_args(range=3., c_o=1., uni_degree=0, mesh_extraction=False)
    overrides = [InterpolationOptions.from_args(range=3., c_o=1., uni_degree=0, mesh_extraction=False)]
    descriptor = InputDataDescriptor(TensorsStructure(np.array([4])), StacksStructure(
        np.array([4]), np.array([1]), np.array([1]), [StackRelationType.ERODE],
        interpolation_options_per_stack=overrides))
    with pytest.raises(ValueError, match="unknown global surface index"):
        compute_model(ii, options, descriptor)


def test_micro_subset_uses_global_surface_indices():
    grid = EngineGrid.from_regular_grid(RegularGrid(
        orthogonal_extent=[0, 1, 0, 1, 0, 1], regular_grid_shape=[2, 2, 2]))
    contacts = MicroPoints(np.array([[0., 0., 0.], [1., 1., 1.]]),
                           np.array([np.eye(3), np.eye(3)]), np.zeros(2), np.array([0, 1]))
    ii = InterpolationInput(SurfacePoints(np.zeros((4, 3))),
                            Orientations(np.zeros((2, 3)), np.ones((2, 3))), grid,
                            micro_points=contacts)
    stacks = StacksStructure(np.array([2, 2]), np.array([1, 1]), np.array([1, 1]),
                             [StackRelationType.ERODE, StackRelationType.ERODE])
    stacks.stack_number = 1
    subset = InterpolationInput.from_interpolation_input_subset(ii, stacks)
    np.testing.assert_array_equal(subset.micro_points.points, contacts.points[1:])
    np.testing.assert_array_equal(subset.micro_points.surface_indices, [0])


def test_micro_design_and_gradient_match_evaluation():
    centers = np.array([[0., 0., 0.], [1., 0., 0.]])
    matrices = np.array([np.diag([1., 2., 3.]), np.diag([3., 1., 2.])])
    weights = np.array([0.3, -0.7])
    xyz = np.array([[0.4, 0.2, 0.1], [0.8, 0.3, 0.2]])
    design = build_micro_design_matrix(centers, centers, matrices, 0.6, "matern_5_2")
    for j in range(len(centers)):
        np.testing.assert_allclose(design[:, j], evaluate_micro_correction(
            centers, centers, np.eye(len(centers))[j], matrices, 0.6, "matern_5_2"))
    grad = evaluate_micro_gradient(xyz, centers, weights, matrices, 0.6, "matern_5_2")
    for axis in range(3):
        shift = np.eye(3)[axis] * 1e-6
        difference = (evaluate_micro_correction(xyz + shift, centers, weights, matrices, 0.6, "matern_5_2")
                      - evaluate_micro_correction(xyz - shift, centers, weights, matrices, 0.6, "matern_5_2")) / 2e-6
        np.testing.assert_allclose(grad[:, axis], difference, atol=1e-8)


@pytest.mark.parametrize("kernel", ["exponential", "matern_3_2", "matern_5_2"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_micro_field_matches_numpy_and_autograd(kernel, dtype, device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    centers = np.array([[0., 0., 0.], [1., 0., 0.]])
    matrices = np.array([np.diag([1., 2., 3.]), np.diag([3., 1., 2.])])
    coords = np.array([[0.4, 0.2, 0.1], [0.8, 0.3, 0.2]])
    with torch.enable_grad():
        xyz = torch.tensor(coords, dtype=getattr(torch, dtype), device=device, requires_grad=True)
        weights = torch.tensor([0.3, -0.7], dtype=xyz.dtype, device=device, requires_grad=True)
        correction = evaluate_micro_correction(xyz, centers, weights, matrices, 0.6, kernel)
        gradient = evaluate_micro_gradient(xyz, centers, weights, matrices, 0.6, kernel)
        xyz_grad, weight_grad = torch.autograd.grad(correction.sum(), (xyz, weights))
    assert correction.dtype == gradient.dtype == xyz.dtype
    assert correction.device == gradient.device == xyz.device
    np.testing.assert_allclose(correction.detach().cpu(), evaluate_micro_correction(
        coords, centers, weights.detach().cpu().numpy(), matrices, 0.6, kernel), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(gradient.detach().cpu(), xyz_grad.detach().cpu(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(weight_grad.detach().cpu(), build_micro_design_matrix(
        coords, centers, matrices, 0.6, kernel).sum(axis=0), rtol=1e-5, atol=1e-6)
