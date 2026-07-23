from copy import deepcopy

import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.modules.data_preprocess._input_preparation import orientations_preprocess, surface_points_preprocess
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance


def test_heterogeneous_nuggets_have_expected_covariance_diagonal(simple_model_2):
    surface_points, orientations, options, descriptor = deepcopy(simple_model_2)
    options.kernel_options.uni_degree = 0
    options.kernel_options.c_o = 7.0
    orientations.nugget_effect_grad = BackendTensor.t.array([0.25, 0.5])
    surface_points.nugget_effect_scalar = BackendTensor.t.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    ori_internal = orientations_preprocess(orientations)
    sp_internal = surface_points_preprocess(surface_points, descriptor.tensors_structure)
    solver_input = SolverInput(sp_internal, ori_internal)
    covariance = yield_covariance(solver_input, options.kernel_options)

    orientations.nugget_effect_grad = BackendTensor.t.zeros(2, dtype=BackendTensor.dtype_obj)
    surface_points.nugget_effect_scalar = BackendTensor.t.zeros(7, dtype=BackendTensor.dtype_obj)
    zero_nugget_covariance = yield_covariance(
        SolverInput(
            surface_points_preprocess(surface_points, descriptor.tensors_structure),
            orientations_preprocess(orientations),
        ),
        options.kernel_options,
    )

    diagonal_delta = BackendTensor.t.diag(covariance - zero_nugget_covariance)
    expected = options.kernel_options.c_o * np.array([
        0.25, 0.5, 0.25, 0.5,
        3.0, 4.0, 5.0, 11.0, 12.0,
    ])
    np.testing.assert_allclose(BackendTensor.t.to_numpy(diagonal_delta), expected, rtol=1e-6, atol=1e-6)


def test_surface_preprocessing_adds_repeated_reference_nuggets(simple_model_2):
    surface_points, _, _, descriptor = deepcopy(simple_model_2)
    surface_points.nugget_effect_scalar = BackendTensor.t.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    internal = surface_points_preprocess(surface_points, descriptor.tensors_structure)

    np.testing.assert_array_equal(
        BackendTensor.t.to_numpy(internal.nugget_effect_ref_rest),
        np.array([3.0, 4.0, 5.0, 11.0, 12.0]),
    )


def test_all_nuggets_receive_pytorch_gradients(simple_model_2):
    if BackendTensor.engine_backend is not AvailableBackends.PYTORCH:
        pytest.skip("PyTorch-only autograd test")
    import torch

    surface_points, orientations, options, descriptor = deepcopy(simple_model_2)
    options.kernel_options.uni_degree = 0
    with torch.enable_grad():
        orientation_nuggets = torch.tensor([0.25, 0.5], dtype=BackendTensor.dtype_obj, requires_grad=True)
        surface_nuggets = torch.arange(1, 8, dtype=BackendTensor.dtype_obj, requires_grad=True)
        orientations.nugget_effect_grad = orientation_nuggets
        surface_points.nugget_effect_scalar = surface_nuggets
        covariance = yield_covariance(
            SolverInput(
                surface_points_preprocess(surface_points, descriptor.tensors_structure),
                orientations_preprocess(orientations),
            ),
            options.kernel_options,
        )
        covariance.sum().backward()

    assert orientation_nuggets.grad is not None and torch.count_nonzero(orientation_nuggets.grad) == 2
    assert surface_nuggets.grad is not None and torch.count_nonzero(surface_nuggets.grad) == 7
