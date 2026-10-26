import numpy as np
import pytest

from gempy_engine.API.interp_single import _interp_scalar_field
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.output.blocks_value_type import ValueType
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level
from tests.fixtures.simple_models import unconformity_complex_factory

N_LEVELS = 3


def _unconformity_model():
    interpolation_input, options, structure = unconformity_complex_factory()
    options.evaluation_options.number_octree_levels = N_LEVELS
    options.evaluation_options.number_octree_levels_surface = N_LEVELS
    options.evaluation_options.mesh_extraction = False
    return interpolation_input, options, structure


def _to_numpy(values):
    return np.asarray(BackendTensor.t.to_numpy(values))


def test_callback_is_called_once_per_completed_level():
    calls = []

    def on_octree_level(level, octree_list):
        calls.append((level, len(octree_list), id(octree_list)))

    solutions = compute_model(*_unconformity_model(), on_octree_level=on_octree_level)

    assert [(level, n) for level, n, _ in calls] == [(i, i + 1) for i in range(N_LEVELS)]
    assert {list_id for *_, list_id in calls} == {id(solutions.octrees_output)}


def test_callback_does_not_change_the_result():
    preview = {}

    def on_octree_level(level, octree_list):
        if level == 1:
            preview["scalar"] = _to_numpy(get_regular_grid_value_for_level(octree_list, value_type=ValueType.scalar))

    with_callback = compute_model(*_unconformity_model(), on_octree_level=on_octree_level)
    without_callback = compute_model(*_unconformity_model())

    for value_type in (ValueType.ids, ValueType.scalar):
        np.testing.assert_allclose(
            _to_numpy(get_regular_grid_value_for_level(with_callback.octrees_output, value_type=value_type)),
            _to_numpy(get_regular_grid_value_for_level(without_callback.octrees_output, value_type=value_type)),
        )
    # A snapshot read inside the callback equals the same level read after the run
    np.testing.assert_allclose(
        preview["scalar"],
        _to_numpy(get_regular_grid_value_for_level(with_callback.octrees_output, level=1, value_type=ValueType.scalar)),
    )


def test_weights_are_solved_once_per_stack_across_levels(monkeypatch):
    solve = _interp_scalar_field._solve_interpolation_result
    n_solves = []

    def counting_solve(*args, **kwargs):
        n_solves.append(1)
        return solve(*args, **kwargs)

    monkeypatch.setattr(_interp_scalar_field, "_solve_interpolation_result", counting_solve)
    interpolation_input, options, structure = _unconformity_model()

    compute_model(interpolation_input, options, structure, on_octree_level=lambda level, octree_list: None)

    assert len(n_solves) == structure.stack_structure.n_stacks


def test_raising_in_callback_aborts_the_computation():
    class Stop(Exception):
        pass

    levels = []

    def on_octree_level(level, octree_list):
        levels.append(level)
        if level == 1:
            raise Stop

    with pytest.raises(Stop):
        compute_model(*_unconformity_model(), on_octree_level=on_octree_level)
    assert levels == [0, 1]
