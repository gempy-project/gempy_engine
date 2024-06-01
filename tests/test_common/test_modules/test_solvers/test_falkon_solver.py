import numpy as np
import pytest
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, orientations_preprocess

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.modules.kernel_constructor._vectors_preparation import \
    evaluation_vectors_preparations
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance, \
    yield_b_vector
from gempy_engine.modules.solver.solver_interface import kernel_reduction


@pytest.fixture(scope='module')
def kriging_eq(moureze_model):
    interpolation_input, options, input_data_descriptor = moureze_model

    sp_internal = surface_points_preprocess(interpolation_input.surface_points, input_data_descriptor.tensors_structure)
    ori_internal = orientations_preprocess(interpolation_input.orientations)
    # noinspection PyTypeChecker
    solver_input = SolverInput(sp_internal, ori_internal, xyz_to_interpolate=None, fault_internal=None)
    A_matrix = yield_covariance(solver_input, options.kernel_options)
    b_vector = yield_b_vector(ori_internal, A_matrix.shape[0])
    return A_matrix, b_vector, options.kernel_options


def test_falkon_1(kriging_eq):
    cov, b, kernel_options = kriging_eq
    kernel_options.compute_condition_number = True
    weights = kernel_reduction(cov, b, kernel_options)
    print(weights)
