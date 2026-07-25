from copy import deepcopy

import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.options import InterpolationOptions, NuggetImplementation
from gempy_engine.modules.data_preprocess._input_preparation import orientations_preprocess, surface_points_preprocess
from gempy_engine.modules.kernel_constructor._kernels_assembler import create_cov_kernel
from gempy_engine.modules.kernel_constructor._vectors_preparation import cov_vectors_preparation
from gempy_engine.modules.kernel_constructor.execution_mode import KernelExecutionMode
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance


@pytest.mark.parametrize(
    ("mode", "expected_surface_block"),
    [
        (
            NuggetImplementation.LEGACY,
            np.diag([2.0, 3.0, 4.0, 6.0, 7.0]),
        ),
        (
            NuggetImplementation.DIAGONAL_REF_REST,
            np.diag([3.0, 4.0, 5.0, 11.0, 12.0]),
        ),
        (
            NuggetImplementation.FULL_POINT_COVARIANCE,
            np.array([
                [3.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 4.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 11.0, 5.0],
                [0.0, 0.0, 0.0, 5.0, 12.0],
            ]),
        ),
    ],
)
def test_surface_nugget_modes_have_expected_covariance(simple_model_2, mode, expected_surface_block):
    surface_points, orientations, options, descriptor = deepcopy(simple_model_2)
    options.kernel_options.uni_degree = 0
    options.kernel_options.c_o = 7.0
    options.kernel_options.nugget_implementation = mode
    orientations.nugget_effect_grad = BackendTensor.t.array([0.25, 0.5])
    surface_points.nugget_effect_scalar = BackendTensor.t.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        dtype=surface_points.sp_coords.dtype,
    )

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

    covariance_delta = BackendTensor.t.to_numpy(covariance - zero_nugget_covariance)
    expected = np.zeros((9, 9))
    expected_orientation = (
        [0.25, 0.25, 0.25, 0.25]
        if mode is NuggetImplementation.LEGACY
        else [0.25, 0.5, 0.25, 0.5]
    )
    expected[:4, :4] = np.diag(expected_orientation)
    expected[4:, 4:] = expected_surface_block
    np.testing.assert_allclose(covariance_delta, options.kernel_options.c_o * expected, rtol=1e-6, atol=1e-5)


def test_nugget_implementation_defaults_to_legacy():
    from gempy_engine.core.data.options import KernelOptions

    options = KernelOptions(range=1, c_o=1)

    assert options.nugget_implementation is NuggetImplementation.LEGACY


def test_nugget_implementation_accepts_string_values():
    from gempy_engine.core.data.options import KernelOptions

    options = KernelOptions(range=1, c_o=1, nugget_implementation="full_point_covariance")

    assert options.nugget_implementation is NuggetImplementation.FULL_POINT_COVARIANCE


def test_interpolation_options_expose_and_serialize_nugget_implementation():
    options = InterpolationOptions.from_args(
        range=1,
        c_o=1,
        nugget_implementation="full_point_covariance",
    )

    assert options.kernel_options.nugget_implementation is NuggetImplementation.FULL_POINT_COVARIANCE
    assert "full_point_covariance" in options.model_dump_json()


def test_surface_preprocessing_preserves_nugget_components_and_surface_ids(simple_model_2):
    surface_points, _, _, descriptor = deepcopy(simple_model_2)
    surface_points.nugget_effect_scalar = BackendTensor.t.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        dtype=surface_points.sp_coords.dtype,
    )

    internal = surface_points_preprocess(surface_points, descriptor.tensors_structure)

    np.testing.assert_array_equal(
        BackendTensor.t.to_numpy(internal.nugget_effect_rest),
        np.array([2.0, 3.0, 4.0, 6.0, 7.0]),
    )
    np.testing.assert_array_equal(
        BackendTensor.t.to_numpy(internal.nugget_effect_ref),
        np.array([1.0, 1.0, 1.0, 5.0, 5.0]),
    )
    np.testing.assert_array_equal(
        BackendTensor.t.to_numpy(internal.surface_ids),
        np.array([0.0, 0.0, 0.0, 1.0, 1.0]),
    )


def test_surface_preprocessing_uses_backend_device_when_torch_default_differs(simple_model_2):
    if BackendTensor.engine_backend is not AvailableBackends.PYTORCH or not BackendTensor.use_gpu:
        pytest.skip("PyTorch GPU-only device regression test")
    import torch

    previous_default_device = torch.get_default_device()
    try:
        torch.set_default_device("cpu")
        surface_points, _, _, descriptor = deepcopy(simple_model_2)

        internal = surface_points_preprocess(surface_points, descriptor.tensors_structure)

        assert internal.surface_ids.device.type == BackendTensor.device.type
    finally:
        torch.set_default_device(previous_default_device)


def test_pytorch_repeat_moves_existing_tensors_to_backend_device():
    if BackendTensor.engine_backend is not AvailableBackends.PYTORCH or not BackendTensor.use_gpu:
        pytest.skip("PyTorch GPU-only device regression test")
    import torch

    values = torch.tensor([0.0], device="cpu")
    repeats = torch.tensor([1], device=BackendTensor.device)

    result = BackendTensor.t.repeat(values, repeats, 0)

    assert result.device.type == BackendTensor.device.type


def test_all_nuggets_receive_pytorch_gradients(simple_model_2):
    if BackendTensor.engine_backend is not AvailableBackends.PYTORCH:
        pytest.skip("PyTorch-only autograd test")
    import torch

    surface_points, orientations, options, descriptor = deepcopy(simple_model_2)
    options.kernel_options.uni_degree = 0
    options.kernel_options.nugget_implementation = NuggetImplementation.FULL_POINT_COVARIANCE
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
    np.testing.assert_allclose(
        BackendTensor.t.to_numpy(surface_nuggets.grad),
        options.kernel_options.c_o * np.array([9.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0]),
    )


@pytest.mark.parametrize("mode", list(NuggetImplementation))
def test_surface_nugget_modes_match_symbolic_column_reduction(simple_model_2, mode):
    from gempy_engine.config import is_pykeops_installed

    if not is_pykeops_installed:
        pytest.skip("PyKeOps is not installed")

    surface_points, orientations, options, descriptor = deepcopy(simple_model_2)
    options.kernel_options.uni_degree = 0
    options.kernel_options.nugget_implementation = mode
    surface_points.nugget_effect_scalar = BackendTensor.t.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        dtype=surface_points.sp_coords.dtype,
    )
    solver_input = SolverInput(
        surface_points_preprocess(surface_points, descriptor.tensors_structure),
        orientations_preprocess(orientations),
    )

    kernel_data = cov_vectors_preparation(solver_input, options.kernel_options)
    dense = create_cov_kernel(kernel_data, options.kernel_options, execution_mode=KernelExecutionMode.DENSE)
    symbolic = create_cov_kernel(
        kernel_data.upgrade_tensors(),
        options.kernel_options,
        execution_mode=KernelExecutionMode.SYMBOLIC,
    )

    np.testing.assert_allclose(
        np.asarray(BackendTensor.t.to_numpy(dense.sum(0))).reshape(-1, 1),
        np.asarray(BackendTensor.t.to_numpy(symbolic.sum(0))),
        rtol=5e-4,
        atol=2e-4,
    )
