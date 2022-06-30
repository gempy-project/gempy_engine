from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends

import numpy as np

bt = BackendTensor


def kernel_reduction(cov, b, smooth=0.01):
    if BackendTensor.pykeops_enabled is True and BackendTensor.engine_backend is not AvailableBackends.tensorflow:
        w = cov.solve(np.asarray(b).astype('float32'),
                      alpha=smooth,
                      dtype_acc='float32',
                      backend="GPU"
                      )
    elif BackendTensor.pykeops_enabled is True and BackendTensor.engine_backend is AvailableBackends.tensorflow:
        w = cov.solve(b.numpy().astype('float32'), alpha=smooth, dtype_acc='float32')
    elif BackendTensor.pykeops_enabled is False and BackendTensor.engine_backend is AvailableBackends.tensorflow:
        # * NOTE: In GPU Tensorflow 2.4 is needs to increase memory usage and by default they allocate all what
        # * leads to a sudden process kill. Use the following to fix the problem
        # gpus = tf.config.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(gpus[0], True)

        w = bt.tfnp.linalg.solve(cov, b)
    elif BackendTensor.pykeops_enabled is False and BackendTensor.engine_backend is not AvailableBackends.tensorflow:
        w = bt.tfnp.linalg.solve(cov, b[:, 0])

        foo = np.array(
                [1.69e-06, 1.65e-06, 1.51e-06, 1.39e-06, 2.08e-06, 1.72e-06, 1.61e-09, 5.94e-10, - 1.54e-09
                , - 1.62e-08, 1.65e-08, 4.85e-10, 9.95e-08, 1.32e-06, 5.44e-07, 7.50e-07, 1.54e-06, 2.69e-08
                , - 1.09e-05, - 2.46e-05, - 1.14e-05, 8.75e-05, 6.10e-05, 9.50e-05, - 1.09e-04, - 5.96e-05, - 1.09e-04
                , 4.57e+00, 1.72e-05, 1.70e-05, - 4.77e-05, - 1.26e-04, 2.40e-06, - 1.78e-05, - 1.66e-05, 2.35e-05
                , 8.52e-06, 2.38e-05, 1.21e-04, 4.52e-05, 1.22e-04, - 5.83e-05, - 1.98e-05, 1.12e-04, 6.31e-05
                , 1.17e-04, 8.28e-06, 2.30e-05, 8.12e-06, - 8.61e-05, - 5.96e-05, - 8.65e-05, 1.80e-04, - 5.61e-06
                , 1.00e+00, - 9.14e-03])

        pass
    else:
        raise AttributeError('There is a weird combination of libraries?')
    return w
