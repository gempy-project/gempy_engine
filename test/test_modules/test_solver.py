import pytest

from gempy_engine.config import AvailableBackends, BackendTensor
from gempy_engine.modules.kernel_constructor._input_preparation import surface_points_preprocess, \
    orientations_preprocess
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_kriging_eq
from gempy_engine.modules.solver.solver_interface import kernel_reduction
import numpy as np


@pytest.fixture(scope='module')
def kriging_eq(simple_model_2):
    surface_points, orientations, options, tensors_structure = simple_model_2

    a_matrix, b_vector = yield_kriging_eq(surface_points, orientations, options, tensors_structure)
    return a_matrix, b_vector


weights_sol = np.array(
    [[-1.50020818],
     [0.09578431],
     [4.61644987],
     [-0.05253686],
     [0.54084665],
     [1.55710522],
     [-4.80773576],
     [-0.14403434],
     [-1.18340672]]
)


def test_solver(kriging_eq):
    weights = kernel_reduction(*kriging_eq)
    np.testing.assert_array_almost_equal(weights.reshape(-1,1), weights_sol, decimal=3)
    print(weights)


# TODO: Obsolete once we change backend in conftest randomly
def test_solver_tf(kriging_eq):
    # BackendTensor.change_backend(AvailableBackends.tensorflow, use_gpu=True,
    #                             pykeops_enabled=False)

    A_matrix, b_vector = kriging_eq
    weights = kernel_reduction(A_matrix, b_vector)
    np.testing.assert_array_almost_equal(weights, weights_sol, decimal=3)
    print(weights)
