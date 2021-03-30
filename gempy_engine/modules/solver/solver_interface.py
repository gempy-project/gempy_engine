from gempy_engine.config import BackendTensor, AvailableBackends

import numpy as np

b = BackendTensor


def kernel_reduction(cov, b, smooth = 0.01):

    if BackendTensor.pykeops_enabled is True and BackendTensor.engine_backend is not AvailableBackends.tensorflow:
        w = cov.solve(np.asarray(b).astype('float32'),
                      alpha=smooth,
                      dtype_acc='float32')
    elif BackendTensor.pykeops_enabled is True and BackendTensor.engine_backend is AvailableBackends.tensorflow:
        w = cov.solve(b.numpy().astype('float32'), alpha=smooth, dtype_acc='float32')
    elif BackendTensor.pykeops_enabled is False and BackendTensor.engine_backend is AvailableBackends.tensorflow:
        w = b.tfnp.linalg.solve(cov, b)
    elif BackendTensor.pykeops_enabled is False and BackendTensor.engine_backend is not AvailableBackends.tensorflow:
        w = b.tfnp.linalg.solve(cov, b[:, 0])
    else:
        raise AttributeError('There is a weird combination of libraries?')
    return w