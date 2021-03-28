import numpy as np

from gempy_engine.modules.covariance._input_preparation import surface_points_preprocess
from gempy_engine.modules.covariance.covariance_interface import yield_covariance
from gempy_engine.config import BackendTensor, AvailableBackends


def test_surface_points_preprocess(simple_model_2):
    surface_points = simple_model_2[0]
    tensors_structure = simple_model_2[3]
    s = surface_points_preprocess(surface_points, tensors_structure.number_of_points_per_surface)
    print(s)

    BackendTensor.change_backend(AvailableBackends.tensorflow)
    s = surface_points_preprocess(surface_points, tensors_structure.number_of_points_per_surface)
    print(s)

def test_covariance(simple_model_2):
    l = np.load('test_kernel_numeric2.npy')
    surface_points = simple_model_2[0]
    orientations = simple_model_2[1]
    options = simple_model_2[2]
    tensors_structure = simple_model_2[3]
    cov = yield_covariance(surface_points, orientations, options, tensors_structure)
    print(cov)
    print(l)

    np.testing.assert_array_almost_equal(np.asarray(cov), l, decimal=3)


def test_covariance_tf(simple_model_2):


    BackendTensor.change_backend(AvailableBackends.tensorflow, use_gpu=False)


    l = np.load('test_kernel_numeric2.npy')
    surface_points = simple_model_2[0]
    orientations = simple_model_2[1]
    options = simple_model_2[2]
    tensors_structure = simple_model_2[3]
    cov = yield_covariance(surface_points, orientations, options, tensors_structure)
    print(cov)
    print(l)

    np.testing.assert_array_almost_equal(np.asarray(cov), l, decimal=3)

    print(BackendTensor.tfnp.debugging.get_log_device_placement())
