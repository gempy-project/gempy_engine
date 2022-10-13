from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends

import numpy as np

bt = BackendTensor


def kernel_reduction(cov, b, smooth=0.000001):
    # ? Maybe we should always compute the conditional_number no matter the branch
    
    dtype = 'float32'    
    match (BackendTensor.engine_backend, BackendTensor.pykeops_enabled):
        case (AvailableBackends.tensorflow, True):
            raise NotImplementedError('Pykeops is not implemented for tensorflow yet')
            # w = cov.solve(b.numpy().astype('float32'), alpha=smooth, dtype_acc='float32')
        case (AvailableBackends.tensorflow, False):
            import tensorflow as tf            
            w = tf.linalg.solve(cov, b)
        case (AvailableBackends.numpy, True):
            w = cov.solve(
                np.asarray(b).astype(dtype),
                alpha=0,
                dtype_acc=dtype,
                backend="CPU"
            )
        case (AvailableBackends.numpy, False):
            if True:
                cond_number = np.linalg.cond(cov)
                svd = np.linalg.svd(cov)
                print(f'Condition number: {cond_number}')
            
            w = bt.tfnp.linalg.solve(cov.astype(dtype), b[:, 0])
        case _:
            raise AttributeError('There is a weird combination of libraries?')
        
    return w
