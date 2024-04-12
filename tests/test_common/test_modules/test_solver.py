import numpy as np
import pytest

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.modules.kernel_constructor._vectors_preparation import \
    evaluation_vectors_preparations
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance, \
    yield_b_vector
from gempy_engine.modules.solver.solver_interface import kernel_reduction


@pytest.fixture(scope='module')
def kriging_eq(simple_model_2_internals):
    sp_internal, ori_internal, options = simple_model_2_internals

    # noinspection PyTypeChecker
    solver_input = SolverInput(sp_internal, ori_internal, xyz_to_interpolate=None, fault_internal=None)
    A_matrix = yield_covariance(solver_input, options.kernel_options)
    b_vector = yield_b_vector(ori_internal, A_matrix.shape[0])
    return A_matrix, b_vector, options.kernel_options


weights_sol = np.array(
    [-1.50021, 0.09578, 4.61645, -0.05254, 0.27042, 0.77855,
     -2.40387, -0.07202, -0.59170]
).reshape((-1, 1))


def test_solver(kriging_eq):
    weights = kernel_reduction(*kriging_eq)
    np.testing.assert_array_almost_equal(np.asarray(weights).reshape(-1,1), weights_sol, decimal=0.5)
    print(weights)


def test_scalar_field_export(simple_model_2_internals, simple_grid_2d):
    sp_internal, ori_internal, options = simple_model_2_internals

    # noinspection PyTypeChecker
    simple_grid_2d = BackendTensor.t.array(simple_grid_2d)
    solver_input = SolverInput(sp_internal, ori_internal, xyz_to_interpolate=simple_grid_2d, fault_internal=None)
    evp = evaluation_vectors_preparations(solver_input, options.kernel_options)
    print(evp)














