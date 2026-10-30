import numpy as np
from types import SimpleNamespace

from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.options import KernelOptions, NuggetImplementation
from gempy_engine.modules.weights_cache.weight_cache_policy import (
    WeightCacheRoute,
    cacheable_kernel_options,
    resolve_weight_cache,
    store_weight_result,
)
from gempy_engine.modules.weights_cache.weights_cache_interface import (WeightCache, generate_cache_key)

example_weights = np.array([.2, .2, .4, .2])


class _CacheSolverInput:
    def __init__(self, identity: int):
        self.identity = identity
        self.sp_internal = SimpleNamespace(
            ref_surface_points=np.array([[identity, 0.0]]),
            rest_surface_points=np.array([[0.0, identity]]),
            nugget_effect_rest=np.array([identity]),
            nugget_effect_ref=np.array([identity]),
            nugget_effect_ref_unique=np.array([identity]),
            surface_ids=np.array([0.0]),
        )
        self.ori_internal = SimpleNamespace(
            dip_positions_tiled=np.array([[identity, 0.0]]),
            gradients_tiled=np.array([[0.0, identity]]),
            nugget_effect_grad=np.array([identity]),
        )
        self.fault_internal = SimpleNamespace(
            fault_values_ref=np.array([[identity]]),
            fault_values_rest=np.array([[0.0]]),
        )

    def __hash__(self):
        return self.identity


def test_save_weights():
    WeightCache.initialize_cache_dir()
    WeightCache.store_weights(
        file_name=f"Sandstone.1",
        hash=(generate_cache_key(
            name="",
            parameters={
                    "shape": 1,
                    "sum"  : np.arange(10),
            }
        )),
        weights=example_weights
    )


def test_load_weights():
    # Load weights
    WeightCache.initialize_cache_dir()
    weights_key = generate_cache_key(
        name="sandstone",
        parameters={
            "shape": 1,
            "sum": np.arange(10),
        }
    )

    retrieved_weights = WeightCache.load_weights(weights_key, look_in_disk=True)
    print(retrieved_weights)


def test_kernel_stabilization_options_change_cache_fingerprint():
    options = KernelOptions(range=1, c_o=1)
    initial = generate_cache_key("", cacheable_kernel_options(options))

    options.fault_drift_regularization = 2e-3
    regularized = generate_cache_key("", cacheable_kernel_options(options))
    options.symmetric_equilibration_method = "ruiz"
    equilibrated = generate_cache_key("", cacheable_kernel_options(options))

    assert len({initial, regularized, equilibrated}) == 3


def test_nugget_implementation_changes_cache_fingerprint():
    options = KernelOptions(range=1, c_o=1)
    legacy = generate_cache_key("", cacheable_kernel_options(options))

    options.nugget_implementation = NuggetImplementation.DIAGONAL_REF_REST
    diagonal = generate_cache_key("", cacheable_kernel_options(options))
    options.nugget_implementation = NuggetImplementation.FULL_POINT_COVARIANCE
    full = generate_cache_key("", cacheable_kernel_options(options))

    assert len({legacy, diagonal, full}) == 3


def test_no_cache_mode_routes_directly_to_solve():
    options = InterpolationOptions.from_args(range=1, c_o=1)
    options.cache_mode = InterpolationOptions.CacheMode.NO_CACHE

    decision = resolve_weight_cache(
        options,
        stack_number=2,
        pykeops_requested=False,
        solver_input=_CacheSolverInput(1),
    )

    assert decision.route is WeightCacheRoute.SOLVE
    assert decision.key == ".2"


def test_matching_cache_fingerprint_returns_cached_weights(monkeypatch):
    options = InterpolationOptions.from_args(range=1, c_o=1)
    options.cache_mode = InterpolationOptions.CacheMode.IN_MEMORY_CACHE
    options.temp_interpolation_values.start_computation_ts = 123
    expected_weights = np.array([1.0, 2.0])

    solver_input = _CacheSolverInput(1)
    first_decision = resolve_weight_cache(
        options,
        stack_number=0,
        pykeops_requested=False,
        solver_input=solver_input,
    )
    assert first_decision.route is WeightCacheRoute.SOLVE_AND_STORE
    monkeypatch.setattr(
        WeightCache,
        "load_weights",
        lambda key, look_in_disk: {
            "hash": first_decision.fingerprint,
            "weights": expected_weights,
        },
    )

    decision = resolve_weight_cache(
        options,
        stack_number=0,
        pykeops_requested=False,
        solver_input=solver_input,
    )

    assert decision.route is WeightCacheRoute.CACHED
    np.testing.assert_array_equal(decision.weights, expected_weights)


def test_solver_input_identity_changes_cache_fingerprint(monkeypatch):
    options = InterpolationOptions.from_args(range=1, c_o=1)
    options.cache_mode = InterpolationOptions.CacheMode.CACHE
    options.temp_interpolation_values.start_computation_ts = 123
    monkeypatch.setattr(WeightCache, "load_weights", lambda **kwargs: None)

    first = resolve_weight_cache(options, 0, False, _CacheSolverInput(1))
    second = resolve_weight_cache(options, 0, False, _CacheSolverInput(2))

    assert first.fingerprint != second.fingerprint


def test_in_memory_cache_does_not_write_to_disk(monkeypatch):
    options = InterpolationOptions.from_args(range=1, c_o=1)
    options.cache_mode = InterpolationOptions.CacheMode.IN_MEMORY_CACHE
    options.temp_interpolation_values.start_computation_ts = 123
    monkeypatch.setattr(WeightCache, "load_weights", lambda **kwargs: None)
    decision = resolve_weight_cache(options, 0, False, _CacheSolverInput(1))
    stored = []
    monkeypatch.setattr(WeightCache, "store_weights", lambda **kwargs: stored.append(kwargs))

    store_weight_result(decision, example_weights)

    assert stored[0]["write_to_disk"] is False


def _reference_fingerprint(fields):
    """The fingerprint as it was hashed before the reads were batched: one host copy per array."""
    import hashlib
    from gempy_engine.core.backend_tensor import BackendTensor
    hasher = hashlib.sha256()
    for name, values in fields:
        hasher.update(name.encode())
        if values is None:
            hasher.update(b"none")
            continue
        array = np.ascontiguousarray(BackendTensor.t.to_numpy(values))
        hasher.update(str(array.shape).encode())
        hasher.update(str(array.dtype).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def test_batched_fingerprint_reads_hash_the_same_bytes(restore_backend):
    import pytest
    torch = pytest.importorskip("torch")
    from gempy_engine.config import AvailableBackends
    from gempy_engine.core.backend_tensor import BackendTensor
    from gempy_engine.core.compute_scope import StackCache
    from gempy_engine.modules.weights_cache import weight_cache_policy as policy
    BackendTensor._change_backend(AvailableBackends.PYTORCH)

    rng = np.random.default_rng(2)

    def solver_input(n):
        sp = SimpleNamespace(ref_surface_points=torch.as_tensor(rng.random((n, 3)), dtype=torch.float32),
                             rest_surface_points=torch.as_tensor(rng.random((n, 3)), dtype=torch.float32).T.T,
                             nugget_effect_rest=torch.as_tensor(rng.random(n), dtype=torch.float32),
                             nugget_effect_ref_unique=torch.as_tensor(rng.random(2), dtype=torch.float64),
                             surface_ids=None)
        ori = SimpleNamespace(dip_positions_tiled=torch.as_tensor(rng.random((3, 6)), dtype=torch.float32).T,  # a view
                              gradients_tiled=torch.as_tensor(rng.random((6, 3)), dtype=torch.float32),
                              nugget_effect_grad=torch.as_tensor(rng.random(6), dtype=torch.float32))
        faults = SimpleNamespace(fault_values_ref=np.zeros(0), fault_values_rest=torch.zeros((0, n)))
        return SimpleNamespace(sp_internal=sp, ori_internal=ori, fault_internal=faults, stack_cache=StackCache())

    inputs = [solver_input(n) for n in (4, 7, 1)]
    for item in inputs:
        assert policy._solver_input_fingerprint(item) == _reference_fingerprint(policy._fingerprint_fields(item))
    policy.prefetch_fingerprints(inputs)
    for item in inputs:
        assert item.stack_cache.fingerprint == _reference_fingerprint(policy._fingerprint_fields(item))
        assert policy.solver_input_fingerprint(item) == item.stack_cache.fingerprint
