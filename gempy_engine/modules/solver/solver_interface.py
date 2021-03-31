from gempy_engine.config import BackendTensor, AvailableBackends

import numpy as np

bt = BackendTensor


def kernel_reduction(cov, b, smooth = 0.01):

    if BackendTensor.pykeops_enabled is True and BackendTensor.engine_backend is not AvailableBackends.tensorflow:
        w = cov.solve(np.asarray(b).astype('float32'),
                      alpha=smooth,
                      dtype_acc='float32')
    elif BackendTensor.pykeops_enabled is True and BackendTensor.engine_backend is AvailableBackends.tensorflow:
        w = cov.solve(b.numpy().astype('float32'), alpha=smooth, dtype_acc='float32')
    elif BackendTensor.pykeops_enabled is False and BackendTensor.engine_backend is AvailableBackends.tensorflow:
        # NOTE: In GPU Tensorflow 2.4 is needs to increase memory usage and by default they allocate all what
        # leads to a sudden process kill. Use the following to fix the problem
        # gpus = tf.config.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(gpus[0], True)

        w = bt.tfnp.linalg.solve(cov, b)
    elif BackendTensor.pykeops_enabled is False and BackendTensor.engine_backend is not AvailableBackends.tensorflow:
        w = bt.tfnp.linalg.solve(cov, b[:, 0])
    else:
        raise AttributeError('There is a weird combination of libraries?')
    return w