from copy import deepcopy

import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.modules.data_preprocess._input_preparation import orientations_preprocess, surface_points_preprocess
from gempy_engine.modules.kernel_constructor.drift_design import DriftDesign, analyze_drift_design, build_drift_design
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance


def test_universal_design_matches_covariance_block(simple_model_2):
    surface_points, orientations, options, descriptor = deepcopy(simple_model_2)
    options.kernel_options.uni_degree = 1
    solver_input = SolverInput(
        surface_points_preprocess(surface_points, descriptor.tensors_structure),
        orientations_preprocess(orientations),
    )

    design = build_drift_design(solver_input, options.kernel_options)
    covariance = yield_covariance(solver_input, options.kernel_options)
    observation_size = design.combined.shape[0]

    np.testing.assert_allclose(
        BackendTensor.t.to_numpy(design.universal),
        BackendTensor.t.to_numpy(covariance[:observation_size, observation_size:]),
    )


def test_second_degree_design_matches_covariance_block(simple_model):
    surface_points, orientations, options, descriptor = deepcopy(simple_model)
    options.kernel_options.uni_degree = 2
    solver_input = SolverInput(
        surface_points_preprocess(surface_points, descriptor.tensors_structure),
        orientations_preprocess(orientations),
    )

    design = build_drift_design(solver_input, options.kernel_options)
    covariance = yield_covariance(solver_input, options.kernel_options)
    observation_size = design.combined.shape[0]

    np.testing.assert_allclose(
        BackendTensor.t.to_numpy(design.universal),
        BackendTensor.t.to_numpy(covariance[:observation_size, observation_size:]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_combined_diagnostics_detect_dependent_columns():
    universal = BackendTensor.t.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    faults = universal[:, :1]
    combined = BackendTensor.tfnp.concatenate((universal, faults), axis=1)
    design = DriftDesign(
        universal=universal,
        faults=faults,
        combined=combined,
        universal_slice=slice(0, 2),
        fault_slice=slice(2, 3),
        labels=("x", "y", "fault_0"),
    )

    report = analyze_drift_design(design)

    assert report.universal.rank == 2
    assert report.faults.rank == 1
    assert report.combined.rank == 2
    assert np.isinf(report.combined.condition_number)
    assert report.combined.dependent_right_vectors.shape == (1, 3)


def test_zero_fault_column_is_reported():
    universal = BackendTensor.t.eye(3, dtype=BackendTensor.dtype_obj)
    faults = BackendTensor.t.zeros((3, 1), dtype=BackendTensor.dtype_obj)
    design = DriftDesign(
        universal=universal,
        faults=faults,
        combined=BackendTensor.tfnp.concatenate((universal, faults), axis=1),
        universal_slice=slice(0, 3),
        fault_slice=slice(3, 4),
        labels=("x", "y", "z", "fault_0"),
    )

    report = analyze_drift_design(design)

    assert report.faults.zero_columns == (0,)
    assert report.combined.rank == 3


def test_normalized_rank_is_not_affected_by_column_units():
    combined = BackendTensor.t.array([[1.0, 0.0], [0.0, 1e-20]])
    design = DriftDesign(
        universal=combined,
        faults=BackendTensor.t.zeros((2, 0), dtype=BackendTensor.dtype_obj),
        combined=combined,
        universal_slice=slice(0, 2),
        fault_slice=slice(2, 2),
        labels=("x", "y"),
    )

    report = analyze_drift_design(design)

    assert report.combined.rank == 1
    assert report.combined.normalized_rank == 2
