from copy import deepcopy

import numpy as np
import pytest

from gempy_engine import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.options import KernelOptions
from gempy_engine.modules.solver.fault_drift_stabilization import stabilize_fault_drift_system


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
    scaled_weights = BackendTensor.t.linalg.solve(covariance, b_vector)
    actual_weights = scaling.restore_weights(scaled_weights)

    np.testing.assert_allclose(
        BackendTensor.t.to_numpy(actual_weights),
        BackendTensor.t.to_numpy(expected_weights),
        rtol=1e-6,
    )
    assert BackendTensor.t.linalg.cond(covariance) < BackendTensor.t.linalg.cond(covariance_original)


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
    stabilized_weights = scaling.restore_weights(BackendTensor.t.linalg.solve(covariance, b_vector))

    assert BackendTensor.t.linalg.cond(covariance) < original_condition_number
    assert abs(stabilized_weights[-1, 0]) > 1e6
    np.testing.assert_allclose(
        BackendTensor.t.to_numpy(stabilized_weights[-1]),
        BackendTensor.t.to_numpy(original_weights[-1]),
        rtol=1e-2,
    )


def test_fault_drift_regularization_must_be_non_negative():
    with pytest.raises(ValueError, match="must be non-negative"):
        KernelOptions(range=1, c_o=1, fault_drift_regularization=-1e-3)


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
