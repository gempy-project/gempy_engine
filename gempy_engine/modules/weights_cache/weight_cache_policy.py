import hashlib
import warnings
from dataclasses import asdict, dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.options import InterpolationOptions, KernelOptions
from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache, generate_cache_key


class WeightCacheRoute(Enum):
    CACHED = auto()
    SOLVE = auto()
    SOLVE_AND_STORE = auto()


@dataclass(frozen=True)
class WeightCacheDecision:
    route: WeightCacheRoute
    key: str
    fingerprint: str | None = None
    weights: Any | None = None
    write_to_disk: bool = True


def resolve_weight_cache(
        options: InterpolationOptions,
        stack_number: int,
        pykeops_requested: bool,
        solver_input,
) -> WeightCacheDecision:
    key = f"{options.cache_model_name}.{stack_number}"
    cache_enabled = (
        options.cache_mode in (InterpolationOptions.CacheMode.CACHE, InterpolationOptions.CacheMode.IN_MEMORY_CACHE)
        and not BackendTensor.COMPUTE_GRADS
    )

    match options.cache_mode:
        case InterpolationOptions.CacheMode.CLEAR_CACHE:
            WeightCache.initialize_cache_dir()
            return WeightCacheDecision(route=WeightCacheRoute.SOLVE, key=key)
        case _ if not cache_enabled:
            return WeightCacheDecision(route=WeightCacheRoute.SOLVE, key=key)
        case InterpolationOptions.CacheMode.CACHE | InterpolationOptions.CacheMode.IN_MEMORY_CACHE:
            return _resolve_enabled_cache(options, key, pykeops_requested, solver_input)
        case InterpolationOptions.CacheMode.NO_CACHE:
            return WeightCacheDecision(route=WeightCacheRoute.SOLVE, key=key)
        case _:
            raise ValueError("Cache mode not recognized")


def store_weight_result(decision: WeightCacheDecision, weights) -> None:
    if decision.route is not WeightCacheRoute.SOLVE_AND_STORE:
        return
    WeightCache.store_weights(
        file_name=decision.key,
        hash=decision.fingerprint,
        weights=weights,
        write_to_disk=decision.write_to_disk,
    )


def cacheable_kernel_options(kernel_options: KernelOptions) -> dict[str, object]:
    parameters = asdict(kernel_options)
    for runtime_field in ("condition_number", "condition_number_before", "condition_number_after"):
        parameters.pop(runtime_field, None)
    return parameters


def solver_input_fingerprint(solver_input) -> str:
    hasher = hashlib.sha256()
    fields = (
        ("surface_points_ref", solver_input.sp_internal.ref_surface_points),
        ("surface_points_rest", solver_input.sp_internal.rest_surface_points),
        ("surface_points_nugget_rest", solver_input.sp_internal.nugget_effect_rest),
        ("surface_points_nugget_ref", solver_input.sp_internal.nugget_effect_ref_unique),
        ("surface_points_surface_ids", solver_input.sp_internal.surface_ids),
        ("orientation_positions", solver_input.ori_internal.dip_positions_tiled),
        ("orientation_gradients", solver_input.ori_internal.gradients_tiled),
        ("orientation_nugget", solver_input.ori_internal.nugget_effect_grad),
        ("fault_values_ref", solver_input.fault_internal.fault_values_ref),
        ("fault_values_rest", solver_input.fault_internal.fault_values_rest),
    )
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


def _resolve_enabled_cache(
        options: InterpolationOptions,
        key: str,
        pykeops_requested: bool,
        solver_input,
) -> WeightCacheDecision:
    timestamp = options.temp_interpolation_values.start_computation_ts
    if timestamp == -1:
        warnings.warn("No start computation timestamp found. No caching.")
        return WeightCacheDecision(route=WeightCacheRoute.SOLVE, key=key)

    fingerprint = generate_cache_key(
        name="",
        parameters={
            "schema": 3,
            "ts": timestamp,
            "kernel_options": cacheable_kernel_options(options.kernel_options),
            "solver": repr(options.kernel_options.kernel_solver),
            "backend": repr(BackendTensor.engine_backend),
            "dtype": str(BackendTensor.dtype),
            "pykeops": pykeops_requested,
            "solver_input": solver_input_fingerprint(solver_input),
        },
    )
    cached = WeightCache.load_weights(
        key=key,
        look_in_disk=options.cache_mode is not InterpolationOptions.CacheMode.IN_MEMORY_CACHE,
    )
    if cached is not None and cached["hash"] == fingerprint:
        return WeightCacheDecision(
            route=WeightCacheRoute.CACHED,
            key=key,
            fingerprint=fingerprint,
            weights=cached["weights"],
            write_to_disk=options.cache_mode is InterpolationOptions.CacheMode.CACHE,
        )
    return WeightCacheDecision(
        route=WeightCacheRoute.SOLVE_AND_STORE,
        key=key,
        fingerprint=fingerprint,
        write_to_disk=options.cache_mode is InterpolationOptions.CacheMode.CACHE,
    )
