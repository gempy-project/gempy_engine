import numpy as np

from gempy_engine.modules.covariance.covariance_interface import yield_covariance


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

