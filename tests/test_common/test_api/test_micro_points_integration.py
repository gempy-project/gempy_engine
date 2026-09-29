import numpy as np
import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import InterpolationOptions, Orientations, SurfacePoints, TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid, RegularGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.micro_points import MicroPoints
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.config import AvailableBackends
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.finite_fault import FiniteFault
from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
from gempy_engine.modules.evaluator.micro_anisotropic_evaluator import (
    build_micro_design_matrix, evaluate_micro_correction, evaluate_micro_gradient,
)
from gempy_engine.modules.evaluator.micro_correction import align_micro_matrices


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
    assert len(corrected[0]) == len(baseline[0]) + 1
    assert np.max(np.abs(corrected[0][:-1] - baseline[0])) > 1e-6
    np.testing.assert_allclose(corrected[1][:-1], baseline[1], atol=1e-8)
    grid_size = grid.len_all_grids
    np.testing.assert_allclose(corrected[0][grid_size:grid_size + 4],
                               baseline[0][grid_size:grid_size + 4], atol=1e-6)
    assert corrected[0].shape == corrected[1].shape
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
    transform = np.diag([2., 3., 4.])
    contacts = MicroPoints(np.array([[0., 0., 0.], [1., 1., 1.]]),
                           np.array([np.eye(3), np.eye(3)]), np.zeros(2), np.array([0, 1]), transform)
    ii = InterpolationInput(SurfacePoints(np.zeros((4, 3))),
                            Orientations(np.zeros((2, 3)), np.ones((2, 3))), grid,
                            micro_points=contacts)
    stacks = StacksStructure(np.array([2, 2]), np.array([1, 1]), np.array([1, 1]),
                             [StackRelationType.ERODE, StackRelationType.ERODE])
    stacks.stack_number = 1
    subset = InterpolationInput.from_interpolation_input_subset(ii, stacks)
    np.testing.assert_array_equal(subset.micro_points.points, contacts.points[1:])
    np.testing.assert_array_equal(subset.micro_points.surface_indices, [0])
    np.testing.assert_array_equal(subset.micro_points.support_to_engine, transform)
    ii._all_micro_points = contacts
    subset = InterpolationInput.from_interpolation_input_subset(ii, stacks)
    np.testing.assert_array_equal(subset.micro_indices, [1])
    assert subset.micro_slice == slice(grid.len_all_grids + 4, grid.len_all_grids + 6)


@pytest.mark.parametrize('flat', [False, True])
def test_aligned_results_map_to_root_contact_rows_across_stacks(monkeypatch, flat):
    sp = np.array([[0.2, 0.2, 0.4], [0.8, 0.2, 0.4], [0.2, 0.8, 0.4], [0.8, 0.8, 0.4]])
    grid = EngineGrid.from_regular_grid(RegularGrid(
        orthogonal_extent=[0, 1, 0, 1, 0, 1], regular_grid_shape=[2, 2, 2]))
    micro = MicroPoints(np.array([[0.5, 0.5, 0.62], [0.5, 0.5, 0.42]]),
                        np.repeat(np.eye(3)[None], 2, axis=0), np.zeros(2), np.array([1, 0]))
    ii = InterpolationInput(SurfacePoints(np.vstack((sp, sp + [0., 0., 0.2]))),
                            Orientations(np.array([[0.5, 0.5, 0.4], [0.5, 0.5, 0.6]]),
                                         np.array([[0.3, 0., 1.], [0., 0.2, 1.]])), grid, micro_points=micro)
    descriptor = InputDataDescriptor(TensorsStructure(np.array([4, 4])), StacksStructure(
        np.array([4, 4]), np.array([1, 1]), np.array([1, 1]),
        [StackRelationType.ERODE, StackRelationType.ERODE]))
    options = InterpolationOptions.from_args(range=3., c_o=1., uni_degree=0, mesh_extraction=False)
    options.micro_options.enabled = True
    monkeypatch.setenv('GEMPY_FLAT_STACKS', str(flat))
    BackendTensor._change_backend(AvailableBackends.numpy, use_pykeops=flat)
    try:
        results = compute_model(ii, options, descriptor).micro_point_results
        np.testing.assert_array_equal(results.source_indices, [1, 0])
        assert results.macro_gradients.shape == (2, 3)
        assert results.anisotropy_matrices.shape == (2, 3, 3)
    finally:
        BackendTensor._change_backend(AvailableBackends.numpy)


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


def _fault_micro_model(finite=False, chain=False):
    fault_sp = np.array([[0.5, 0.2, 0.2], [0.5, 0.8, 0.2],
                         [0.5, 0.2, 0.8], [0.5, 0.8, 0.8]])
    strata_sp = np.array([[0.2, 0.2, 0.4], [0.8, 0.2, 0.4],
                          [0.2, 0.8, 0.4], [0.8, 0.8, 0.4]])
    points = [fault_sp]
    if chain:
        points.append(fault_sp + [0.12, 0, 0])
    points.append(strata_sp)
    count = len(points)
    relations = np.zeros((count, count), dtype=bool)
    relations[0, -1] = True
    if chain:
        relations[0, 1] = relations[1, 2] = True
    contacts = np.array([[0.56, 0.5, 0.54], [0.5, 0.5, 0.5], fault_sp[0]])
    micro = MicroPoints(contacts, np.repeat(np.eye(3)[None] * 3, 3, axis=0),
                        np.zeros(3), np.array([count - 1] * 3))
    grid = EngineGrid.from_regular_grid(RegularGrid(
        orthogonal_extent=[0, 1, 0, 1, 0, 1], regular_grid_shape=[3, 3, 3]))
    ii = InterpolationInput(SurfacePoints(np.vstack(points)), Orientations(
        np.array([[0.5, 0.5, 0.5]] * (count - 1) + [[0.5, 0.5, 0.4]]),
        np.array([[1., 0., 0.]] * (count - 1) + [[0., 0., 1.]])),
        grid, micro_points=micro)
    faults = [None] * count
    if finite:
        faults[0] = FaultsData.from_user_input(None, FiniteFault(
            center=(0.5, 0.5, 0.5), strike_radius=0.8, dip_radius=0.8))
    descriptor = InputDataDescriptor(TensorsStructure(np.array([4] * count)), StacksStructure(
        np.array([4] * count), np.array([1] * count), np.array([1] * count),
        [StackRelationType.FAULT] * (count - 1) + [StackRelationType.ERODE],
        faults_relations=relations, faults_input_data=faults))
    options = InterpolationOptions.from_args(range=3., c_o=1., uni_degree=0, mesh_extraction=False)
    options.micro_options.enabled = True
    options.evaluation_options.number_octree_levels = 1
    return ii, descriptor, options


@pytest.mark.parametrize('backend', ['numpy', 'flat', 'torch'])
@pytest.mark.parametrize('preserve_macro_points', [False, True])
@pytest.mark.parametrize('macro_nugget', [0., 0.1])
def test_micro_contacts_match_final_surface_isovalues(monkeypatch, backend, preserve_macro_points, macro_nugget):
    ii, descriptor, options = _fault_micro_model()
    micro = ii.micro_points
    ii.micro_points = MicroPoints(micro.points[:1], micro.anisotropy_matrices[:1],
                                  micro.nuggets[:1], micro.surface_indices[:1])
    ii.surface_points.sp_coords[-1, 2] += 0.15
    ii.surface_points.nugget_effect_scalar[4:] = macro_nugget
    options.micro_options.preserve_macro_points = preserve_macro_points
    options.micro_options.nugget = 0.
    options.micro_options.strength = 1.
    monkeypatch.setenv('GEMPY_FLAT_STACKS', str(backend == 'flat'))
    if backend == 'torch':
        pytest.importorskip('torch')
    BackendTensor._change_backend(
        AvailableBackends.PYTORCH if backend == 'torch' else AvailableBackends.numpy,
        use_pykeops=backend == 'flat',
    )
    try:
        options.micro_options.enabled = False
        with pytest.warns(UserWarning, match='ignored'):
            baseline = compute_model(ii, options, descriptor)
        baseline_fields = baseline.octrees_output[0].outputs[-1].exported_fields
        target = BackendTensor.t.to_numpy(baseline_fields.scalar_field_at_surface_points).copy()
        if macro_nugget:
            macro_values = BackendTensor.t.to_numpy(baseline_fields.scalar_field_everywhere)[-4:]
            assert abs(macro_values.mean() - target.item()) > 1e-4

        options.micro_options.enabled = True
        result = compute_model(ii, options, descriptor)
        fields = result.octrees_output[0].outputs[-1].exported_fields
        final_isovalues = BackendTensor.t.to_numpy(fields.scalar_field_at_surface_points)
        contact_values = BackendTensor.t.to_numpy(fields.scalar_field_everywhere)[-1:]
        np.testing.assert_allclose(final_isovalues, target, atol=1e-6)
        np.testing.assert_allclose(contact_values, final_isovalues, atol=1e-6)
    finally:
        BackendTensor._change_backend(AvailableBackends.numpy)


@pytest.mark.parametrize('finite,chain', [(False, False), (False, True), (True, False)])
def test_fault_micro_contacts_flow_through_dependencies(monkeypatch, finite, chain):
    ii, descriptor, options = _fault_micro_model(finite, chain)
    from gempy_engine.API.interp_single._aux_faults_ops import _modify_faults_values_output

    normalized = []
    original = _modify_faults_values_output

    def capture(fault_input, output, xyz_to_interpolate):
        values = original(fault_input, output, xyz_to_interpolate)
        normalized.append(np.asarray(values).copy())
        return values

    monkeypatch.setattr('gempy_engine.API.interp_single._multi_scalar_field_manager._modify_faults_values_output', capture)
    monkeypatch.setattr('gempy_engine.API.interp_single._stack_ops._modify_faults_values_output', capture)
    authored_points = ii.micro_points.points.copy()
    surface_points = ii.surface_points.sp_coords.copy()
    def run(flat, enabled=True):
        options.micro_options.enabled = enabled
        monkeypatch.setenv('GEMPY_FLAT_STACKS', str(flat))
        BackendTensor._change_backend(AvailableBackends.numpy, use_pykeops=flat)
        result = compute_model(ii, options, descriptor)
        outputs = result.octrees_output[0].outputs
        return [np.asarray(o.scalar_fields.exported_fields.scalar_field_everywhere).copy() for o in outputs], outputs

    try:
        serial, _ = run(False)
        assert len(normalized) == len(serial) - 1
        assert all(v.shape == (1, len(serial[0])) and np.isfinite(v).all() for v in normalized)
        assert np.any(normalized[0][0, -3:] != normalized[0][0, -4])
        normalized.clear()
        flat, _ = run(True)
        for left, right in zip(serial, flat):
            np.testing.assert_allclose(left, right, rtol=1e-4, atol=1e-4)
        baseline, baseline_outputs = run(False, False)
        np.testing.assert_allclose(serial[0][:-3], baseline[0], atol=1e-8)
        assert np.max(np.abs(serial[-1][:-3] - baseline[-1])) > 1e-5
        grid_index = np.flatnonzero(np.all(np.isclose(ii.grid.values, [0.5, 0.5, 0.5]), axis=1))[0]
        target = np.asarray(baseline_outputs[-1].scalar_fields.scalar_field_at_sp).item()
        assert abs(serial[-1][grid_index] - target) < abs(baseline[-1][grid_index] - target)
        np.testing.assert_array_equal(ii.micro_points.points, authored_points)
        np.testing.assert_array_equal(ii.surface_points.sp_coords, surface_points)
    finally:
        BackendTensor._change_backend(AvailableBackends.numpy)


def test_fault_micro_query_preserves_torch_gradients():
    torch = pytest.importorskip('torch')
    ii, descriptor, options = _fault_micro_model()
    sp = torch.tensor(ii.surface_points.sp_coords, dtype=torch.float32, requires_grad=True)
    contacts = torch.tensor(ii.micro_points.points, dtype=torch.float32, requires_grad=True)
    ii.surface_points = SurfacePoints(sp)
    ii.micro_points = MicroPoints(contacts, ii.micro_points.anisotropy_matrices,
                                  ii.micro_points.nuggets, ii.micro_points.surface_indices)
    BackendTensor._change_backend(AvailableBackends.PYTORCH, grads=True)
    try:
        result = compute_model(ii, options, descriptor)
        field = result.octrees_output[0].outputs[-1].exported_fields.scalar_field
        grads = torch.autograd.grad(field.sum(), (sp, contacts))
        assert all(torch.isfinite(grad).all() and grad.abs().max() > 1e-8 for grad in grads)
    finally:
        BackendTensor._change_backend(AvailableBackends.numpy)


@pytest.mark.parametrize('finite', [False, True])
@pytest.mark.parametrize('flat', [False, True])
def test_fault_reference_prefix_unaffected_by_remote_contact(monkeypatch, finite, flat):
    ii, descriptor, options = _fault_micro_model(finite=finite)
    ii.micro_points.points[0] = [50., -50., 50.]
    from gempy_engine.API.interp_single._aux_faults_ops import _modify_faults_values_output
    rows = []

    def capture(fault_input, output, xyz_to_interpolate):
        values = _modify_faults_values_output(fault_input, output, xyz_to_interpolate)
        rows.append(np.asarray(values).copy())
        return values

    module = ('_stack_ops' if flat else '_multi_scalar_field_manager')
    monkeypatch.setattr(f'gempy_engine.API.interp_single.{module}._modify_faults_values_output', capture)
    monkeypatch.setenv('GEMPY_FLAT_STACKS', str(flat))
    BackendTensor._change_backend(AvailableBackends.numpy, use_pykeops=flat)
    try:
        compute_model(ii, options, descriptor)
        enabled = rows.pop()
        options.micro_options.enabled = False
        with pytest.warns(UserWarning, match='ignored'):
            compute_model(ii, options, descriptor)
        disabled = rows.pop()
        np.testing.assert_allclose(enabled[:, :-3], disabled, rtol=1e-6, atol=1e-6)
        assert enabled.shape[1] == disabled.shape[1] + 3
    finally:
        BackendTensor._change_backend(AvailableBackends.numpy)


def test_micro_fit_uses_existing_macro_evaluation(monkeypatch):
    ii, descriptor, options = _fault_micro_model(chain=True)
    from gempy_engine.API.interp_single import _interp_single_feature
    original = _interp_single_feature._evaluate_sys_eq
    calls = []

    def capture(*args, **kwargs):
        calls.append(args[0].xyz_to_interpolate.shape[0])
        return original(*args, **kwargs)

    monkeypatch.setenv('GEMPY_FLAT_STACKS', 'False')
    monkeypatch.setattr(_interp_single_feature, '_evaluate_sys_eq', capture)
    compute_model(ii, options, descriptor)
    assert calls == [ii.grid.len_all_grids + ii.surface_points.n_points + len(ii.micro_points.points)] * 3


@pytest.mark.parametrize('flat', [False, True])
def test_external_upstream_fault_evaluates_shared_suffix(monkeypatch, flat):
    ii, descriptor, options = _fault_micro_model()
    descriptor.stack_structure.interp_functions_per_stack = [CustomInterpolationFunctions(
        scalar_field_at_surface_points=np.array([0.5]),
        implicit_function=lambda xyz: xyz[:, 0],
    ), None]
    monkeypatch.setenv('GEMPY_FLAT_STACKS', str(flat))
    BackendTensor._change_backend(AvailableBackends.numpy, use_pykeops=flat)
    try:
        result = compute_model(ii, options, descriptor)
        outputs = result.octrees_output[0].outputs
        expected = ii.grid.len_all_grids + ii.surface_points.n_points + len(ii.micro_points.points)
        assert all(len(o.scalar_fields.exported_fields.scalar_field_everywhere) == expected for o in outputs)
        assert np.isfinite(outputs[-1].scalar_fields.exported_fields.scalar_field_everywhere).all()
    finally:
        BackendTensor._change_backend(AvailableBackends.numpy)


def test_flat_faults_use_each_faults_segmentation_function(monkeypatch):
    ii, descriptor, options = _fault_micro_model(chain=True)
    relations = descriptor.stack_structure.faults_relations
    relations[0, 1] = False
    relations[1, 2] = False
    descriptor.stack_structure.segmentation_functions_per_stack = [
        lambda xyz: 0.5, lambda xyz: 30., None,
    ]
    from gempy_engine.API.interp_single._aux_faults_ops import _modify_faults_values_output
    published = []
    def capture(fault_input, output, xyz_to_interpolate):
        values = _modify_faults_values_output(fault_input, output, xyz_to_interpolate)
        published.append(np.asarray(values).copy())
        return values
    monkeypatch.setattr('gempy_engine.API.interp_single._multi_scalar_field_manager._modify_faults_values_output', capture)
    monkeypatch.setattr('gempy_engine.API.interp_single._stack_ops._modify_faults_values_output', capture)
    try:
        BackendTensor._change_backend(AvailableBackends.numpy)
        monkeypatch.setenv('GEMPY_FLAT_STACKS', 'False')
        serial = compute_model(ii, options, descriptor)
        serial_queries = published.copy()
        published.clear()
        BackendTensor._change_backend(AvailableBackends.numpy, use_pykeops=True)
        monkeypatch.setenv('GEMPY_FLAT_STACKS', 'True')
        flat = compute_model(ii, options, descriptor)
        assert len(serial_queries) == len(published) == 2
        for expected, actual in zip(serial_queries, published):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            flat.octrees_output[0].outputs[-1].scalar_fields.exported_fields.scalar_field_everywhere,
            serial.octrees_output[0].outputs[-1].scalar_fields.exported_fields.scalar_field_everywhere,
            rtol=1e-5, atol=1e-5,
        )
    finally:
        BackendTensor._change_backend(AvailableBackends.numpy)


@pytest.mark.parametrize('flat', [False, True])
@pytest.mark.parametrize('chain', [False, True])
def test_fault_surface_micro_rejected_even_with_upstream_faults(monkeypatch, flat, chain):
    ii, descriptor, options = _fault_micro_model(chain=chain)
    ii.micro_points.surface_indices[0] = 1 if chain else 0
    monkeypatch.setenv('GEMPY_FLAT_STACKS', str(flat))
    BackendTensor._change_backend(AvailableBackends.numpy, use_pykeops=flat)
    try:
        with pytest.raises(ValueError, match='fault stack.*micro points on fault surfaces'):
            compute_model(ii, options, descriptor)
    finally:
        BackendTensor._change_backend(AvailableBackends.numpy)


@pytest.mark.parametrize('flat', [False, True])
@pytest.mark.parametrize('root_enabled', [False, True])
def test_disabled_fault_surface_micro_is_ignored(monkeypatch, flat, root_enabled):
    ii, descriptor, options = _fault_micro_model()
    ii.micro_points.surface_indices[0] = 0
    overrides = [options.model_copy(deep=True), options.model_copy(deep=True)]
    overrides[0].micro_options.enabled = False
    descriptor.stack_structure.interpolation_options_per_stack = overrides
    options.micro_options.enabled = root_enabled
    monkeypatch.setenv('GEMPY_FLAT_STACKS', str(flat))
    BackendTensor._change_backend(AvailableBackends.numpy, use_pykeops=flat)
    try:
        with pytest.warns(UserWarning, match='Stack 0 contains micro points.*ignored'):
            result = compute_model(ii, options, descriptor)
        assert len(result.octrees_output[0].outputs) == 2
        expected_size = ii.grid.len_all_grids + ii.surface_points.n_points + len(ii.micro_points.points)
        assert all(len(o.scalar_fields.exported_fields.scalar_field_everywhere) == expected_size
                   for o in result.octrees_output[0].outputs)
    finally:
        BackendTensor._change_backend(AvailableBackends.numpy)


@pytest.mark.parametrize('flat', [False, True])
def test_external_micro_contacts_rejected_before_evaluation(monkeypatch, flat):
    ii, descriptor, options = _fault_micro_model()
    descriptor.stack_structure.interp_functions_per_stack = [None, CustomInterpolationFunctions(
        scalar_field_at_surface_points=np.array([0.4]),
        implicit_function=lambda xyz: xyz[:, 2],
    )]
    monkeypatch.setenv('GEMPY_FLAT_STACKS', str(flat))

    def unexpected_evaluation(*args, **kwargs):
        pytest.fail('Unsupported micro contacts must be rejected before interpolation')

    monkeypatch.setattr('gempy_engine.API.model.model_api.interpolate_n_octree_levels', unexpected_evaluation)
    with pytest.raises(NotImplementedError, match='external-function stacks'):
        compute_model(ii, options, descriptor)


@pytest.mark.parametrize('flat', [False, True])
@pytest.mark.parametrize('deduplicate', [False, True])
@pytest.mark.parametrize('preserve_macro_points', [False, True])
def test_micro_contacts_flow_through_octree_and_mesh(monkeypatch, flat, deduplicate, preserve_macro_points):
    ii, descriptor, options = _fault_micro_model()
    options.micro_options.preserve_macro_points = preserve_macro_points
    ii.set_temp_grid(EngineGrid.from_regular_grid(RegularGrid(
        orthogonal_extent=[0, 1, 0, 1, 0, 1], regular_grid_shape=[2, 2, 2])))
    options.evaluation_options.number_octree_levels = 2
    options.evaluation_options.number_octree_levels_surface = 2
    options.evaluation_options.mesh_extraction = True
    options.evaluation_options.compute_scalar_gradient = True
    options.evaluation_options.deduplicate_octree_corners = deduplicate
    monkeypatch.setenv('GEMPY_FLAT_STACKS', str(flat))
    BackendTensor._change_backend(AvailableBackends.numpy, use_pykeops=flat)
    try:
        result = compute_model(ii, options, descriptor)
        assert len(result.octrees_output) == 2
        for level in result.octrees_output:
            for output in level.outputs:
                fields = output.exported_fields
                assert len(fields.scalar_field) == output.grid.len_all_grids
                assert len(fields.scalar_field_everywhere) == (
                    output.grid.len_all_grids + ii.surface_points.n_points + len(ii.micro_points.points))
                assert np.isfinite(fields.scalar_field_everywhere).all()
            strata_fields = level.outputs[-1].exported_fields
            np.testing.assert_allclose(
                strata_fields.scalar_field_everywhere[-len(ii.micro_points.points):],
                np.repeat(strata_fields.scalar_field_at_surface_points, len(ii.micro_points.points)),
                atol=1e-6,
            )
        assert result.dc_meshes
        for mesh in result.dc_meshes:
            assert mesh is not None
            assert len(mesh.vertices) > 0
            assert len(mesh.edges) > 0
            assert np.isfinite(mesh.vertices).all()
    finally:
        BackendTensor._change_backend(AvailableBackends.numpy)


@pytest.mark.parametrize('backend', ['numpy', 'flat', 'torch'])
@pytest.mark.parametrize('gradient_output', [False, True])
def test_aligned_results_use_uncorrected_contact_gradients(monkeypatch, backend, gradient_output):
    from gempy_engine.modules.evaluator.micro_correction import fit_micro_fields
    ii, descriptor, options = _fault_micro_model()
    ii.orientations.dip_gradients[-1] = [0.7, 0.2, 1.]
    a = np.array([[2., 0.3, 0.], [0., 1.5, 0.2], [0., 0., 0.8]])
    authored_basis = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    ii.micro_points = MicroPoints(ii.micro_points.points, np.repeat(np.linalg.inv(a @ authored_basis)[None], 3, axis=0),
                                  ii.micro_points.nuggets, ii.micro_points.surface_indices, a)
    options.micro_options.preserve_macro_points = True
    options.evaluation_options.compute_scalar_gradient = gradient_output
    monkeypatch.setenv('GEMPY_FLAT_STACKS', str(backend == 'flat'))
    BackendTensor._change_backend(AvailableBackends.PYTORCH if backend == 'torch' else AvailableBackends.numpy,
                                  use_pykeops=backend == 'flat', grads=backend == 'torch')
    try:
        options.micro_options.align_to_macro = False
        unaligned = compute_model(ii, options, descriptor)
        assert unaligned.micro_point_results is None
        suffix = slice(-len(ii.micro_points.points), None)
        raw_gradients = []
        def capture(interpolation_input, fields, *args):
            if interpolation_input.micro_indices is not None and len(interpolation_input.micro_indices):
                raw_gradients.append(np.stack([
                    BackendTensor.t.to_numpy(getattr(fields, f'{axis}_field_everywhere')[suffix]).copy()
                    for axis in ('gx', 'gy', 'gz')], axis=1))
            return fit_micro_fields(interpolation_input, fields, *args)
        monkeypatch.setattr('gempy_engine.API.interp_single._interp_single_feature.fit_micro_fields', capture)
        monkeypatch.setattr('gempy_engine.API.interp_single._stack_ops.fit_micro_fields', capture)
        options.micro_options.align_to_macro = True
        result = compute_model(ii, options, descriptor)
        raw_grad = raw_gradients[-1]
        published = result.micro_point_results
        assert published is not None
        np.testing.assert_array_equal(published.source_indices, [0, 1, 2])
        np.testing.assert_allclose(BackendTensor.t.to_numpy(published.macro_gradients), raw_grad, rtol=1e-5, atol=1e-5)
        fitted = BackendTensor.t.to_numpy(published.anisotropy_matrices)
        assert fitted.shape == (3, 3, 3)
        for matrix, gradient in zip(fitted, raw_grad):
            basis = np.linalg.inv(a) @ np.linalg.inv(matrix)
            np.testing.assert_allclose(np.linalg.norm(basis, axis=0), [1., 2., 3.], atol=1e-5)
            np.testing.assert_allclose(basis[:, 2] / np.linalg.norm(basis[:, 2]),
                                       gradient @ a / np.linalg.norm(gradient @ a), atol=1e-5)
        fields = result.octrees_output[-1].outputs[-1].exported_fields
        assert (fields.gx_field_everywhere is not None) == gradient_output
        assert options.evaluation_options.compute_scalar_gradient == gradient_output
        np.testing.assert_allclose(BackendTensor.t.to_numpy(fields.scalar_field_everywhere[suffix]),
                                   np.repeat(BackendTensor.t.to_numpy(fields.scalar_field_at_surface_points), 3), atol=3e-5)
    finally:
        BackendTensor._change_backend(AvailableBackends.numpy)


@pytest.mark.parametrize('flat', [False, True])
def test_curved_macro_normals_match_independent_contact_differences(monkeypatch, flat):
    sp = np.array([[0.2, 0.2, 0.35], [0.8, 0.2, 0.49],
                   [0.2, 0.8, 0.44], [0.8, 0.8, 0.67]])
    contacts = np.array([[0.32, 0.34, 0.48], [0.7, 0.65, 0.59]])
    grid = EngineGrid.from_regular_grid(RegularGrid(
        orthogonal_extent=[0, 1, 0, 1, 0, 1], regular_grid_shape=[2, 2, 2]))
    ii = InterpolationInput(SurfacePoints(sp), Orientations(
        np.array([[0.5, 0.5, 0.48]]), np.array([[0.4, 0.2, 1.]])), grid,
        micro_points=MicroPoints(contacts, np.repeat(np.diag([1., 2., 0.5])[None], 2, axis=0),
                                 np.zeros(2), np.zeros(2, dtype=int)))
    descriptor = InputDataDescriptor(TensorsStructure(np.array([4])), StacksStructure(
        np.array([4]), np.array([1]), np.array([1]), [StackRelationType.ERODE]))
    options = InterpolationOptions.from_args(range=3., c_o=1., uni_degree=0, mesh_extraction=False)
    options.micro_options.enabled = True
    options.micro_options.align_to_macro = True
    options.evaluation_options.compute_scalar_gradient = False
    step = 1e-3
    shifts = np.eye(3) * step
    queries = np.concatenate([contacts + shift for shift in shifts] +
                             [contacts - shift for shift in shifts])
    baseline_input = InterpolationInput(SurfacePoints(sp), ii.orientations,
                                        EngineGrid.from_regular_grid(RegularGrid(
                                            orthogonal_extent=[0, 1, 0, 1, 0, 1], regular_grid_shape=[2, 2, 2])))
    baseline_input.grid.custom_grid = EngineGrid.from_xyz_coords(queries).custom_grid
    monkeypatch.setenv('GEMPY_FLAT_STACKS', str(flat))
    BackendTensor._change_backend(AvailableBackends.numpy, use_pykeops=flat)
    try:
        baseline = compute_model(baseline_input, options, descriptor)
        values = baseline.octrees_output[0].outputs[0].exported_fields.scalar_field_everywhere[
            baseline_input.grid.custom_grid_slice]
        differences = np.stack([(values[axis * 2:(axis + 1) * 2] -
                                 values[(axis + 3) * 2:(axis + 4) * 2]) / (2 * step)
                                for axis in range(3)], axis=1)
        normals = differences / np.linalg.norm(differences, axis=1, keepdims=True)
        assert np.linalg.norm(normals[0] - normals[1]) > 1e-3

        result = compute_model(ii, options, descriptor)
        published = result.micro_point_results
        np.testing.assert_array_equal(published.source_indices, [0, 1])
        published_normals = published.macro_gradients / np.linalg.norm(
            published.macro_gradients, axis=1, keepdims=True)
        np.testing.assert_allclose(published_normals, normals, atol=3e-3)
        matrices = published.anisotropy_matrices
        for matrix, normal in zip(matrices, normals):
            axis = np.linalg.inv(matrix)[:, 2]
            np.testing.assert_allclose(axis / np.linalg.norm(axis), normal, atol=2e-3)
        assert result.octrees_output[0].outputs[0].exported_fields.gx_field_everywhere is None
    finally:
        BackendTensor._change_backend(AvailableBackends.numpy)


@pytest.mark.parametrize('flat', [False, True])
def test_mixed_stack_overrides_acquire_only_enabled_macro_normals(monkeypatch, flat):
    sp = np.array([[0.2, 0.2, 0.4], [0.8, 0.2, 0.4],
                   [0.2, 0.8, 0.4], [0.8, 0.8, 0.4]])
    grid = EngineGrid.from_regular_grid(RegularGrid(
        orthogonal_extent=[0, 1, 0, 1, 0, 1], regular_grid_shape=[2, 2, 2]))
    micro = MicroPoints(np.array([[0.5, 0.5, 0.62], [0.5, 0.5, 0.42]]),
                        np.repeat(np.diag([1., 2., 0.5])[None], 2, axis=0),
                        np.zeros(2), np.array([1, 0]))
    ii = InterpolationInput(SurfacePoints(np.vstack((sp, sp + [0., 0., 0.2]))),
                            Orientations(np.array([[0.5, 0.5, 0.4], [0.5, 0.5, 0.6]]),
                                         np.array([[0.3, 0., 1.], [0., 0.2, 1.]])), grid, micro_points=micro)
    options = InterpolationOptions.from_args(range=3., c_o=1., uni_degree=0, mesh_extraction=False)
    options.evaluation_options.compute_scalar_gradient = False
    overrides = [options.model_copy(deep=True), options.model_copy(deep=True)]
    overrides[1].micro_options.enabled = True
    overrides[1].micro_options.align_to_macro = True
    descriptor = InputDataDescriptor(TensorsStructure(np.array([4, 4])), StacksStructure(
        np.array([4, 4]), np.array([1, 1]), np.array([1, 1]),
        [StackRelationType.ERODE, StackRelationType.ERODE], interpolation_options_per_stack=overrides))
    monkeypatch.setenv('GEMPY_FLAT_STACKS', str(flat))
    BackendTensor._change_backend(AvailableBackends.numpy, use_pykeops=flat)
    try:
        with pytest.warns(UserWarning, match='Stack 0 contains micro points.*ignored'):
            result = compute_model(ii, options, descriptor)
        published = result.micro_point_results
        np.testing.assert_array_equal(published.source_indices, [0])
        assert published.macro_gradients.shape == (1, 3)
        assert np.isfinite(published.macro_gradients).all()
        assert np.linalg.norm(published.macro_gradients) > 1e-4
        axis = np.linalg.inv(published.anisotropy_matrices[0])[:, 2]
        normal = published.macro_gradients[0] / np.linalg.norm(published.macro_gradients[0])
        np.testing.assert_allclose(axis / np.linalg.norm(axis), normal, atol=1e-5)
        for output in result.octrees_output[0].outputs:
            assert output.exported_fields.gx_field_everywhere is None
        assert not options.micro_options.enabled
        assert not options.evaluation_options.compute_scalar_gradient
        assert not overrides[0].micro_options.enabled
        assert not overrides[1].evaluation_options.compute_scalar_gradient
    finally:
        BackendTensor._change_backend(AvailableBackends.numpy)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_alignment_zero_gradient_and_authored_frame_independence(backend):
    torch = pytest.importorskip('torch') if backend == 'torch' else None
    BackendTensor._change_backend(AvailableBackends.PYTORCH if torch else AvailableBackends.numpy, grads=bool(torch))
    try:
        a = np.diag([2., 3., 4.])
        basis = np.diag([0.5, 1., 2.])
        matrix = np.linalg.inv(a @ basis)
        matrices = np.stack((matrix, matrix @ np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])))
        if torch:
            matrices = torch.tensor(matrices, dtype=torch.float64, requires_grad=True)
            a = torch.tensor(a, dtype=torch.float64, requires_grad=True)
            gradients = torch.tensor([[1., 2., 3.], [0., 0., 0.]], dtype=torch.float64, requires_grad=True)
        else:
            gradients = np.array([[1., 2., 3.], [0., 0., 0.]])
        micro = MicroPoints(np.zeros((2, 3)), matrices, np.zeros(2), np.zeros(2, dtype=int), a)
        fitted = align_micro_matrices(micro, gradients)
        np.testing.assert_allclose(BackendTensor.t.to_numpy(fitted[1]), BackendTensor.t.to_numpy(matrices[1]))
        np.testing.assert_allclose(np.linalg.norm(np.linalg.inv(BackendTensor.t.to_numpy(a)) @
                                                   np.linalg.inv(BackendTensor.t.to_numpy(fitted[0])), axis=0),
                                   [0.5, 1., 2.])
        if torch:
            grads = torch.autograd.grad(fitted[0].square().sum(), (matrices, a, gradients))
            assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in grads)
    finally:
        BackendTensor._change_backend(AvailableBackends.numpy)


@pytest.mark.parametrize('transform', [np.zeros((3, 3)), np.ones((2, 3)),
                                      np.diag([1., 1., np.nan])])
def test_micro_rejects_invalid_support_to_engine(transform):
    with pytest.raises(ValueError, match='support_to_engine'):
        MicroPoints(np.zeros((1, 3)), np.eye(3)[None], np.zeros(1), np.array([0]), transform)
