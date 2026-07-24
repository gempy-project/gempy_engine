import concurrent.futures
import threading
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import gempy_engine.API.interp_single._interp_scalar_field as scalar_field_api
import gempy_engine.modules.solver.interpolation_solver as interpolation_solver
from gempy_engine.config import AvailableBackends, is_pykeops_installed
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.options import KernelOptions
from gempy_engine.modules.data_preprocess._input_preparation import orientations_preprocess, surface_points_preprocess
from gempy_engine.modules.kernel_constructor.execution_mode import KernelExecutionMode
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance
from gempy_engine.modules.solver.interpolation_solver import (
    InterpolationSolveRoute,
    select_interpolation_solve_route,
)


@pytest.mark.parametrize(
    ("method", "n_faults", "pykeops_requested", "optimizing", "expected"),
    [
        ("none", 0, False, False, InterpolationSolveRoute.DENSE_UNTRANSFORMED),
        ("none", 1, False, False, InterpolationSolveRoute.DENSE_FAULT_STABILIZED),
        ("ruiz", 0, False, False, InterpolationSolveRoute.DENSE_RUIZ),
        ("ruiz", 2, False, False, InterpolationSolveRoute.DENSE_RUIZ),
        ("none", 0, True, False, InterpolationSolveRoute.PYKEOPS_WITH_DENSE_FALLBACK),
        ("none", 0, True, True, InterpolationSolveRoute.DENSE_UNTRANSFORMED),
    ],
)
def test_select_interpolation_solve_route(
        monkeypatch,
        method,
        n_faults,
        pykeops_requested,
        optimizing,
        expected,
):
    monkeypatch.setattr(BackendTensor, "engine_backend", AvailableBackends.PYTORCH)
    options = KernelOptions(
        range=1,
        c_o=1,
        symmetric_equilibration_method=method,
        optimizing_condition_number=optimizing,
    )

    warning_context = (
        pytest.warns(RuntimeWarning)
        if pykeops_requested and expected is not InterpolationSolveRoute.PYKEOPS_WITH_DENSE_FALLBACK
        else nullcontext()
    )
    with warning_context:
        route = select_interpolation_solve_route(options, n_faults, pykeops_requested)

    assert route is expected


def test_numpy_pykeops_route_is_rejected(monkeypatch):
    monkeypatch.setattr(BackendTensor, "engine_backend", AvailableBackends.numpy)
    options = KernelOptions(range=1, c_o=1)

    with pytest.raises(ValueError, match="requires the PyTorch backend"):
        select_interpolation_solve_route(options, n_faults=0, pykeops_requested=True)


def test_concurrent_mixed_routes_do_not_mutate_global_pykeops_state(monkeypatch):
    monkeypatch.setattr(BackendTensor, "engine_backend", AvailableBackends.PYTORCH)
    monkeypatch.setattr(BackendTensor, "pykeops_enabled", False)
    barrier = threading.Barrier(2)
    observed_routes = []

    def _result(route):
        barrier.wait(timeout=5)
        observed_routes.append((route, BackendTensor.pykeops_enabled))
        return SimpleNamespace(weights=np.array([1.0]))

    monkeypatch.setattr(
        scalar_field_api,
        "solve_dense_fault_stabilized",
        lambda *_: _result(InterpolationSolveRoute.DENSE_FAULT_STABILIZED),
    )
    monkeypatch.setattr(
        scalar_field_api,
        "solve_pykeops_with_dense_fallback",
        lambda *_: _result(InterpolationSolveRoute.PYKEOPS_WITH_DENSE_FALLBACK),
    )

    fault_input = SimpleNamespace(fault_internal=SimpleNamespace(n_faults=1))
    no_fault_input = SimpleNamespace(fault_internal=SimpleNamespace(n_faults=0))
    fault_options = KernelOptions(range=1, c_o=1)
    no_fault_options = KernelOptions(range=1, c_o=1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(scalar_field_api._solve_interpolation, fault_input, fault_options, True),
            executor.submit(scalar_field_api._solve_interpolation, no_fault_input, no_fault_options, True),
        ]
        weights = [future.result(timeout=10) for future in futures]

    assert all(np.array_equal(weight, np.array([1.0])) for weight in weights)
    assert set(observed_routes) == {
        (InterpolationSolveRoute.DENSE_FAULT_STABILIZED, False),
        (InterpolationSolveRoute.PYKEOPS_WITH_DENSE_FALLBACK, False),
    }
    assert BackendTensor.pykeops_enabled is False


def test_dense_covariance_mode_ignores_global_pykeops_flag(monkeypatch, simple_model_2):
    surface_points, orientations, options, descriptor = simple_model_2
    solver_input = SolverInput(
        surface_points_preprocess(surface_points, descriptor.tensors_structure),
        orientations_preprocess(orientations),
    )
    monkeypatch.setattr(BackendTensor, "pykeops_enabled", True)

    covariance = yield_covariance(
        solver_input,
        options.kernel_options,
        execution_mode=KernelExecutionMode.DENSE,
    )

    assert covariance.shape[0] == covariance.shape[1]
    assert BackendTensor.pykeops_enabled is True


@pytest.mark.skipif(not is_pykeops_installed, reason="PyKeOps is not installed")
def test_actual_dense_fault_and_pykeops_solves_can_run_concurrently(monkeypatch, simple_model_2):
    if BackendTensor.engine_backend is not AvailableBackends.PYTORCH:
        pytest.skip("Mixed execution-mode solve requires the PyTorch test backend")

    surface_points, orientations, options, descriptor = simple_model_2
    sp_internal = surface_points_preprocess(surface_points, descriptor.tensors_structure)
    ori_internal = orientations_preprocess(orientations)
    no_fault_input = SolverInput(sp_internal, ori_internal)
    n_surface_equations = sp_internal.n_points
    fault_data = FaultsData(
        fault_values_everywhere=BackendTensor.t.zeros((1, 0), dtype=BackendTensor.dtype_obj),
        fault_values_on_sp=BackendTensor.t.zeros((1, surface_points.n_points), dtype=BackendTensor.dtype_obj),
        fault_values_ref=BackendTensor.t.ones((1, n_surface_equations), dtype=BackendTensor.dtype_obj),
        fault_values_rest=BackendTensor.t.zeros((1, n_surface_equations), dtype=BackendTensor.dtype_obj),
    )
    fault_input = SolverInput(sp_internal, ori_internal, fault_internal=fault_data)
    monkeypatch.setattr(BackendTensor, "pykeops_enabled", False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(
            scalar_field_api._solve_interpolation,
            fault_input,
            deepcopy(options.kernel_options),
            True,
        )
        lazy_future = executor.submit(
            scalar_field_api._solve_interpolation,
            no_fault_input,
            deepcopy(options.kernel_options),
            True,
        )
        dense_weights = dense_future.result(timeout=60)
        lazy_weights = lazy_future.result(timeout=60)

    assert bool(BackendTensor.t.isfinite(dense_weights).all())
    assert bool(BackendTensor.t.isfinite(lazy_weights).all())
    assert BackendTensor.pykeops_enabled is False


def test_pykeops_exception_falls_back_to_dense(monkeypatch):
    kernel_data = SimpleNamespace(upgrade_tensors=lambda: object())
    expected = SimpleNamespace(weights=np.array([2.0]), used_fallback=True)
    monkeypatch.setattr(interpolation_solver, "cov_vectors_preparation", lambda *_: kernel_data)
    monkeypatch.setattr(
        interpolation_solver,
        "_build_physical_system",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("lazy failure")),
    )
    fallback_calls = []

    def _dense_fallback(*args, **kwargs):
        fallback_calls.append(kwargs)
        return expected

    monkeypatch.setattr(interpolation_solver, "solve_dense_untransformed", _dense_fallback)

    with pytest.warns(RuntimeWarning, match="retrying with a dense solve"):
        result = interpolation_solver.solve_pykeops_with_dense_fallback(
            SimpleNamespace(),
            KernelOptions(range=1, c_o=1),
        )

    assert result is expected
    assert fallback_calls == [{"kernel_data": kernel_data, "used_fallback": True}]


def test_fallback_result_is_not_stored_in_pykeops_cache(monkeypatch):
    weights = np.array([3.0])
    decision = SimpleNamespace(
        route=scalar_field_api.WeightCacheRoute.SOLVE_AND_STORE,
        weights=None,
    )
    monkeypatch.setattr(scalar_field_api, "pykeops_solver_requested", lambda: True)
    monkeypatch.setattr(scalar_field_api, "resolve_weight_cache", lambda **kwargs: decision)
    monkeypatch.setattr(
        scalar_field_api,
        "_solve_interpolation_result",
        lambda *args: SimpleNamespace(weights=weights, used_fallback=True),
    )
    stored = []
    monkeypatch.setattr(scalar_field_api, "store_weight_result", lambda *args: stored.append(args))

    actual = scalar_field_api.compute_weights(
        SimpleNamespace(),
        stack_number=0,
        options=SimpleNamespace(kernel_options=KernelOptions(range=1, c_o=1)),
    )

    np.testing.assert_array_equal(actual, weights)
    assert stored == []
