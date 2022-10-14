import warnings

import gempy_engine.config
from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends

import numpy as np

bt = BackendTensor


def kernel_reduction(cov, b, smooth=0.000001):
    # ? Maybe we should always compute the conditional_number no matter the branch
    
    dtype = gempy_engine.config.TENSOR_DTYPE    
    match (BackendTensor.engine_backend, BackendTensor.pykeops_enabled):
        case (AvailableBackends.tensorflow, True):
            raise NotImplementedError('Pykeops is not implemented for tensorflow yet')
            # w = cov.solve(b.numpy().astype('float32'), alpha=smooth, dtype_acc='float32')
        case (AvailableBackends.tensorflow, False):
            import tensorflow as tf            
            w = tf.linalg.solve(cov, b)
        case (AvailableBackends.numpy, True):
            # ! Only Positivie definite matrices are solved. Otherwise the kernel gets stuck
            # * Very interesting: https://stats.stackexchange.com/questions/386813/use-the-rbf-kernel-to-construct-a-positive-definite-covariance-matrix
            
            w = cov.solve(
                np.asarray(b).astype(dtype),
                alpha=900000,
                dtype_acc=dtype,
                backend="CPU"
            )
        case (AvailableBackends.numpy, False):
            if True:
                cond_number = np.linalg.cond(cov)
                svd = np.linalg.svd(cov)
                is_positive_definite = np.all(np.linalg.eigvals(cov) > 0)
                print(f'Condition number: {cond_number}. Is positive definite: {is_positive_definite}')
                if is_positive_definite == False:  # ! Careful numpy False
                    warnings.warn('The covariance matrix is not positive definite')
            
            w = bt.tfnp.linalg.solve(cov.astype(dtype), b[:, 0])
        case _:
            raise AttributeError('There is a weird combination of libraries?')
        
    return w
