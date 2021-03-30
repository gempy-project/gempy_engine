import pytest

from gempy_engine.modules.kernel_constructor._input_preparation import surface_points_preprocess, \
    orientations_preprocess
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_kriging_eq
from gempy_engine.modules.solver.solver_interface import kernel_reduction


@pytest.fixture(scope='module')
def kriging_eq(simple_model_2):
    surface_points, orientations, options, tensors_structure = simple_model_2

    a_matrix, b_vector = yield_kriging_eq(surface_points, orientations, options, tensors_structure)
    return a_matrix, b_vector

def test_solver(kriging_eq):
    
    weights = kernel_reduction(*kriging_eq)
    print(weights)
