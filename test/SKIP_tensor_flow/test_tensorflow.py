import pytest

from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends
tf = pytest.importorskip("tensorflow")
import numpy as np

from ..fixtures.simple_models import simple_model_2


def test_xla_surface_points_preprocessing(simple_model_2):
    surface_points = simple_model_2[0]
    tensors_structure = simple_model_2[3]
    BackendTensor.change_backend(AvailableBackends.tensorflow)
    from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess

    @tf.function
    def tf_f(surface_points, tensors_structure):
        static = surface_points_preprocess(surface_points, tensors_structure.number_of_points_per_surface)
        return static.ref_surface_points

    s = tf_f(surface_points, tensors_structure.tensors_structure)
    print(s)


def test_tf_solver_in_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    A_matrix = np.array(
        [[3.33333333e-01 , - 1.94673332e-01,    0.00000000e+00, - 3.01533853e-01,  - 3.92331581e-01, - 3.54864422e-01, - 2.65353320e-01,  7.73119610e-03 ,   5.92207474e-03],
         [-1.94673332e-01,  3.33333333e-01, - 3.01533853e-01,  0.00000000e+00,        - 3.78884022e+01, - 1.89442011e+01,- 9.96376726e+00,- 8.42708750e+00,- 1.64019424e+00],
         [0.00000000e+00 , - 3.01533853e-01,    3.33333333e-01, - 1.16388929e+00,    5.42257372e-01,   4.76095895e-01,   3.34540850e-01,  1.50954706e-02 , - 4.11295338e-03],
         [-3.01533853e-01,   0.00000000e+00,  - 1.16388929e+00,  3.33333333e-01 ,    0.00000000e+00,   8.47661381e+00,   6.39166728e+00, - 2.09259705e+01, - 1.71154976e+01],
         [-3.92331581e-01, - 3.78884022e+01,    5.42257372e-01,  0.00000000e+00 ,    1.17808762e+00,   5.89043810e-01,   1.90201905e-01,   2.91988734e-01,   5.12415142e-02],
         [-3.54864422e-01, - 1.89442011e+01,    4.76095895e-01,  8.47661381e+00 ,    5.89043810e-01,   7.07870476e-01,   3.53935238e-01,   9.17537786e-02,   1.48993441e-01],
         [-2.65353320e-01, - 9.96376726e+00,    3.34540850e-01,  6.39166728e+00 ,    1.90201905e-01,   3.53935238e-01,   2.51321905e-01,   2.76593900e-03,   8.24757628e-02],
         [7.73119610e-03 , - 8.42708750e+00,    1.50954706e-02, - 2.09259705e+01,    2.91988734e-01,   9.17537786e-02,   2.76593900e-03,    1.08359866e+0,   4.07052879e-01],
         [5.92207474e-03 , - 1.64019424e+00,  - 4.11295338e-03, - 1.71154976e+01,    5.12415142e-02,   1.48993441e-01,   8.24757628e-02,    4.07052879e-0,   4.38377579e-01]],
    dtype="float32")

    b_vector = np.array([[0. ], [0. ], [1. ], [0.8], [0. ], [0. ], [0. ], [0. ], [0. ]], dtype="float32")
    w = tf.linalg.solve(A_matrix, b_vector)
    print(w)

    @tf.function(experimental_compile=True)
    def tf_f(A_matrix, b_vector):
        return tf.linalg.solve(A_matrix, b_vector)

    tf_f(A_matrix, b_vector)


