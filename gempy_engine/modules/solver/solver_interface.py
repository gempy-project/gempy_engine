from gempy_engine.config import DEFAULT_DTYPE
from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends

import numpy as np

bt = BackendTensor


def kernel_reduction(cov, b, smooth=.00001, compute=True):
    if BackendTensor.engine_backend is AvailableBackends.numpyPykeopsCPU or BackendTensor.engine_backend is AvailableBackends.numpyPykeopsGPU:
        print("Compiling solver...")
        w_f = cov.solve(np.asarray(b).astype(DEFAULT_DTYPE),
                        alpha=smooth,
                        dtype_acc=DEFAULT_DTYPE,
                        backend=BackendTensor.get_backend(),
                        call=False
                        )
        print("Compilation done!")

        if compute:
            w = w_f()
        else:
            w = w_f
        # TODO: This is a hack to test pykeops only up to here:
        #bt.pykeops_enabled = False

    elif  BackendTensor.engine_backend is AvailableBackends.tensorflowCPU or\
          BackendTensor.engine_backend is AvailableBackends.tensorflowGPU:
        # NOTE: In GPU Tensorflow 2.4 is needs to increase memory usage and by default they allocate all what
        # leads to a sudden process kill. Use the following to fix the problem
        # gpus = tf.config.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(gpus[0], True)

        w = bt.tfnp.linalg.solve(cov, b)
    elif BackendTensor.engine_backend is AvailableBackends.numpy:
        w = bt.tfnp.linalg.solve(cov, b[:, 0])
    else:
        raise AttributeError('There is a weird combination of libraries?')
    return w
