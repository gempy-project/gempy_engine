import pytest

from gempy_engine.API.model.model_api import compute_model
import gempy_engine.config
from gempy_engine.core.data import InterpolationOptions
from tests.conftest import plot_pyvista
from tests.fixtures.simple_models import simple_model_interpolation_input_factory


