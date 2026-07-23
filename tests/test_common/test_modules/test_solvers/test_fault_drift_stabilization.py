from copy import deepcopy

import numpy as np
import pytest

from gempy_engine import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.options import KernelOptions
from gempy_engine.modules.solver.fault_drift_stabilization import stabilize_fault_drift_system
from gempy_engine.modules.solver.linear_system_transform import equilibrate_symmetric_system


def _nearly_singular_fault_system():
    covariance = BackendTensor.t.array([
        [2.0, 0.2, 1e-8],
        [0.2, 1.0, 2e-8],
        [1e-8, 2e-8, 0.0],
    ])
    b_vector = BackendTensor.t.array([[1.0], [0.5], [0.0]])
    return covariance, b_vector


def test_fault_drift_equilibration_preserves_unregularized_solution():
    covariance, b_vector = _nearly_singular_fault_system()
    covariance_original = BackendTensor.t.copy(covariance)
    b_vector_original = BackendTensor.t.copy(b_vector)
    expected_weights = BackendTensor.t.linalg.solve(covariance_original, b_vector_original)

    scaling = stabilize_fault_drift_system(
        covariance=covariance,
        b_vector=b_vector,
        n_faults=1,
        relative_regularization=0.0,
    )
    np.testing.assert_array_equal(BackendTensor.t.to_numpy(covariance), BackendTensor.t.to_numpy(covariance_original))
    np.testing.assert_array_equal(BackendTensor.t.to_numpy(b_vector), BackendTensor.t.to_numpy(b_vector_original))
    scaled_weights = BackendTensor.t.linalg.solve(scaling.matrix, scaling.rhs)
    actual_weights = scaling.restore_weights(scaled_weights)

    np.testing.assert_allclose(
        BackendTensor.t.to_numpy(actual_weights),
        BackendTensor.t.to_numpy(expected_weights),
        rtol=1e-6,
    )
    assert BackendTensor.t.linalg.cond(scaling.matrix) < BackendTensor.t.linalg.cond(covariance_original)


def test_fault_drift_regularization_improves_conditioning_without_removing_large_offset():
    covariance, b_vector = _nearly_singular_fault_system()
    original_condition_number = BackendTensor.t.linalg.cond(covariance)
    original_weights = BackendTensor.t.linalg.solve(covariance, b_vector)

    scaling = stabilize_fault_drift_system(
        covariance=covariance,
        b_vector=b_vector,
        n_faults=1,
        relative_regularization=1e-3,
    )
    stabilized_weights = scaling.restore_weights(BackendTensor.t.linalg.solve(scaling.matrix, scaling.rhs))

    assert BackendTensor.t.linalg.cond(scaling.matrix) < original_condition_number
    assert abs(stabilized_weights[-1, 0]) > 1e6
    np.testing.assert_allclose(
        BackendTensor.t.to_numpy(stabilized_weights[-1]),
        BackendTensor.t.to_numpy(original_weights[-1]),
        rtol=1e-2,
    )


def test_fault_drift_regularization_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        KernelOptions(range=1, c_o=1, fault_drift_regularization=-1e-3)


@pytest.mark.parametrize("rhs_columns", [None, 1, 3])
def test_ruiz_equilibration_preserves_solution_and_rhs_shapes(rhs_columns):
    matrix = BackendTensor.t.array([
        [1e-8, 2e-4, 0.0],
        [2e-4, 3.0, 2e2],
        [0.0, 2e2, -4e6],
    ])
    rhs = BackendTensor.t.array([1.0, 2.0, 3.0])
    if rhs_columns is not None:
        rhs = BackendTensor.tfnp.tile(rhs[:, None], (1, rhs_columns))
    expected = BackendTensor.t.linalg.solve(matrix, rhs)

    transform = equilibrate_symmetric_system(matrix, rhs, max_iterations=20)
    actual = transform.restore_weights(BackendTensor.t.linalg.solve(transform.matrix, transform.rhs))

    np.testing.assert_allclose(BackendTensor.t.to_numpy(actual), BackendTensor.t.to_numpy(expected), rtol=1e-6, atol=1e-8)
    assert transform.rhs.shape == rhs.shape
    assert BackendTensor.t.linalg.cond(transform.matrix) < BackendTensor.t.linalg.cond(matrix)


def test_initial_guess_is_transformed_to_solver_coordinates():
    covariance, b_vector = _nearly_singular_fault_system()
    transform = stabilize_fault_drift_system(covariance, b_vector, n_faults=1, relative_regularization=0.0)
    physical_guess = BackendTensor.t.array([1.0, 2.0, 3.0])

    scaled_guess = transform.scale_initial_guess(physical_guess)

    np.testing.assert_allclose(
        BackendTensor.t.to_numpy(transform.restore_weights(scaled_guess)),
        BackendTensor.t.to_numpy(physical_guess),
    )


def test_ruiz_scaling_preserves_pytorch_autograd():
    if BackendTensor.engine_backend is not AvailableBackends.PYTORCH:
        pytest.skip("PyTorch-only autograd test")
    import torch

    with torch.enable_grad():
        matrix = torch.tensor(
            [[2.0, 0.5], [0.5, -3.0]],
            dtype=BackendTensor.dtype_obj,
            requires_grad=True,
        )
        rhs = torch.tensor([[1.0], [2.0]], dtype=BackendTensor.dtype_obj, requires_grad=True)
        transform = equilibrate_symmetric_system(matrix, rhs)
        weights = transform.restore_weights(torch.linalg.solve(transform.matrix, transform.rhs))
        weights.square().sum().backward()

    assert matrix.grad is not None and torch.isfinite(matrix.grad).all()
    assert rhs.grad is not None and torch.isfinite(rhs.grad).all()


def test_fault_drift_stabilization_in_graben_model(graben_fault_model):
    baseline_input, baseline_structure, baseline_options = deepcopy(graben_fault_model)
    baseline_options.evaluation_options.number_octree_levels = 1
    baseline_options.evaluation_options.dual_contouring = False
    baseline_options.kernel_options.compute_condition_number = True
    baseline_options.kernel_options.fault_drift_equilibration = False
    baseline_options.kernel_options.fault_drift_regularization = 0.0
    baseline_solutions = compute_model(baseline_input, baseline_options, baseline_structure)
    interpolation_input, structure, options = deepcopy(graben_fault_model)
    options.evaluation_options.number_octree_levels = 1
    options.evaluation_options.dual_contouring = False
    options.kernel_options.compute_condition_number = True

    solutions = compute_model(interpolation_input, options, structure)

    scalar_field = solutions.octrees_output[-1].outputs[-1].exported_fields.scalar_field
    baseline_scalar_field = baseline_solutions.octrees_output[-1].outputs[-1].exported_fields.scalar_field
    assert np.isfinite(BackendTensor.t.to_numpy(scalar_field)).all()
    assert np.isfinite(options.kernel_options.condition_number)
    if BackendTensor.engine_backend is AvailableBackends.numpy:
        np.testing.assert_allclose(
            BackendTensor.t.to_numpy(scalar_field),
            BackendTensor.t.to_numpy(baseline_scalar_field),
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.parametrize("fixture_name", ["one_fault_model", "one_finite_fault_model", "graben_fault_model"])
def test_whole_system_equilibration_on_fault_models(request, fixture_name):
    interpolation_input, structure, options = deepcopy(request.getfixturevalue(fixture_name))
    options.evaluation_options.number_octree_levels = 1
    options.evaluation_options.dual_contouring = False
    options.kernel_options.compute_condition_number = True
    options.kernel_options.symmetric_equilibration_method = "ruiz"

    solutions = compute_model(interpolation_input, options, structure)

    for output in solutions.octrees_output[-1].outputs:
        scalar_field = output.exported_fields.scalar_field
        assert np.isfinite(BackendTensor.t.to_numpy(scalar_field)).all()
    assert np.isfinite(float(BackendTensor.t.to_numpy(options.kernel_options.condition_number_after)))
    assert np.isfinite(float(BackendTensor.t.to_numpy(options.kernel_options.condition_number_before)))
