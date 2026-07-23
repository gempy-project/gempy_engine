import numpy as np

from gempy_engine.API.interp_single._interp_scalar_field import _cacheable_kernel_options
from gempy_engine.core.data.options import KernelOptions
from gempy_engine.modules.weights_cache.weights_cache_interface import (WeightCache, generate_cache_key)

example_weights = np.array([.2, .2, .4, .2])


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
    initial = generate_cache_key("", _cacheable_kernel_options(options))

    options.fault_drift_regularization = 2e-3
    regularized = generate_cache_key("", _cacheable_kernel_options(options))
    options.symmetric_equilibration_method = "ruiz"
    equilibrated = generate_cache_key("", _cacheable_kernel_options(options))

    assert len({initial, regularized, equilibrated}) == 3
