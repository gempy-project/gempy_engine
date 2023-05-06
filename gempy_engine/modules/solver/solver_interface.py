import warnings

import gempy_engine.config
from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends

import numpy as np
from scipy.sparse.linalg import aslinearoperator, cg
bt = BackendTensor


def kernel_reduction(cov, b, compute_condition_number=False) -> np.ndarray:
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
            # ! Only Positive definite matrices are solved. Otherwise, the kernel gets stuck
            # * Very interesting: https://stats.stackexchange.com/questions/386813/use-the-rbf-kernel-to-construct-a-positive-definite-covariance-matrix

            if scipy_solver := True:
                A = aslinearoperator(cov)
                w, info = cg(A, b[:, 0], tol=0)
            else:
                w = cov.solve(
                    np.asarray(b).astype(dtype),
                    alpha=.00000,
                    dtype_acc=dtype,
                    backend="CPU"
                )
        case (AvailableBackends.numpy, False):
            if compute_condition_number:
                _compute_conditional_number(cov)

            w = bt.tfnp.linalg.solve(cov.astype(dtype), b[:, 0])
        case _:
            raise AttributeError('There is a weird combination of libraries?')

    return w


def _compute_conditional_number(cov):
    cond_number = np.linalg.cond(cov)
    svd = np.linalg.svd(cov)
    eigvals = np.linalg.eigvals(cov)
    is_positive_definite = np.all(eigvals > 0)
    print(f'Condition number: {cond_number}. Is positive definite: {is_positive_definite}')
    if not is_positive_definite:  # ! Careful numpy False
        warnings.warn('The covariance matrix is not positive definite')
