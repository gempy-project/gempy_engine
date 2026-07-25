from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from gempy_engine.API.model import model_api
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.options.temp_interpolation_values import TempInterpolationValues
from gempy_engine.modules.evaluator.symbolic_evaluator import _validate_stacked_dimensions


def test_computations_do_not_share_volatile_option_state(monkeypatch):
    options = InterpolationOptions.from_args(
        range=1.0,
        c_o=1.0,
        number_octree_levels=2,
        mesh_extraction=False,
    )
    barrier = Barrier(2)

    monkeypatch.setattr(model_api.WeightCache, "initialize_cache_dir", lambda: None)
    monkeypatch.setattr(model_api.WeightCache, "clear_cache", lambda: None)
    monkeypatch.setattr(model_api.BackendTensor, "clear_gpu_memory", lambda: None)
    monkeypatch.setattr(model_api, "_check_input_validity", lambda *_: None)

    def observe_options(interpolation_input, options, data_descriptor):
        level = interpolation_input
        options.temp_interpolation_values.current_octree_level = level
        barrier.wait()
        return options.temp_interpolation_values.current_octree_level

    monkeypatch.setattr(model_api, "interpolate_n_octree_levels", observe_options)

    class FakeSolutions:
        def __init__(self, octrees_output, **_):
            self.observed_level = octrees_output

    monkeypatch.setattr(model_api, "Solutions", FakeSolutions)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(model_api.compute_model, level, options, None)
            for level in (1, 3)
        ]

    assert [future.result().observed_level for future in futures] == [1, 3]
    assert options.temp_interpolation_values == TempInterpolationValues()


def test_stacked_dimension_validation_reports_range_mismatch():
    class FakeKernel:
        shape = (4, 8, 1)

    class FakeWeights:
        shape = (4,)

    with pytest.raises(ValueError, match="range dimensions=\\(4, 16\\)") as error:
        _validate_stacked_dimensions(FakeKernel(), FakeWeights(), M_sizes=[8, 8], N_sizes=[2, 2])

    assert "kernel=(4, 8, 1)" in str(error.value)
