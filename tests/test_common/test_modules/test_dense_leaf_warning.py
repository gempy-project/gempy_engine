import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from gempy_engine.API.interp_single.interp_features import _warn_dense_leaf_level


def _inputs(min_level):
    interpolation_input = SimpleNamespace(grid=SimpleNamespace(octree_grid=SimpleNamespace(
        regular_grid_shape=np.array([38, 32, 2]))))
    options = SimpleNamespace(evaluation_options=SimpleNamespace(octree_min_level=min_level))
    return interpolation_input, options


def test_warns_when_min_level_reaches_leaf_level():
    with pytest.warns(UserWarning, match=r"dense 304x256x16 grid \(1,245,184 cells\).*octree_min_level <= 2"):
        _warn_dense_leaf_level(*_inputs(3), n_levels=4)


@pytest.mark.parametrize("min_level, n_levels", [(2, 4), (0, 1), (3, 1)])
def test_silent_when_leaf_level_stays_adaptive(min_level, n_levels):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_dense_leaf_level(*_inputs(min_level), n_levels=n_levels)
