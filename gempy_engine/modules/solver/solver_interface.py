import warnings

import gempy_engine.config
from gempy_engine.core.data.kernel_classes.solvers import Solvers
from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends

import numpy as np
from scipy.sparse.linalg import aslinearoperator, cg, cgs, LinearOperator, gmres

bt = BackendTensor


def kernel_reduction(cov, b, solver: Solvers, compute_condition_number=False, ) -> np.ndarray:
    # ? Maybe we should always compute the conditional_number no matter the branch

    dtype = gempy_engine.config.TENSOR_DTYPE
    match (BackendTensor.engine_backend, BackendTensor.pykeops_enabled, solver):
        case (AvailableBackends.tensorflow, True, _):
            raise NotImplementedError('Pykeops is not implemented for tensorflow yet')
            # w = cov.solve(b.numpy().astype('float32'), alpha=smooth, dtype_acc='float32')
        case (AvailableBackends.tensorflow, False, _):
            import tensorflow as tf
            w = tf.linalg.solve(cov, b)
        case (AvailableBackends.numpy, True, Solvers.DEFAULT | Solvers.PYKEOPS_CG):
            # ! Only Positive definite matrices are solved. Otherwise, the kernel gets stuck
            # * Very interesting: https://stats.stackexchange.com/questions/386813/use-the-rbf-kernel-to-construct-a-positive-definite-covariance-matrix
            w = cov.solve(
                np.asarray(b).astype(dtype),
                alpha=.00000,
                dtype_acc=dtype,
                backend="CPU"
            )
        case (AvailableBackends.numpy, _, Solvers.SCIPY_CG):
            A = aslinearoperator(cov)
            w, info = cg(
                A=A,
                b=b[:, 0],
                maxiter=100,
                tol=.05  # * With this tolerance we do 8 iterations
            )
            w = np.atleast_2d(w).T
        
        case (AvailableBackends.numpy, _, Solvers.GMRES):
            A = aslinearoperator(cov)
            w, info = gmres(
                A=A,
                b=b[:, 0],
                maxiter=5,
                tol=1e-5
            )
            w = np.atleast_2d(w).T

        case (AvailableBackends.numpy, False, Solvers.DEFAULT):
            w = bt.tfnp.linalg.solve(cov.astype(dtype), b[:, 0])

            if compute_condition_number:
                _compute_conditional_number(cov)

        case _:
            raise AttributeError(f'There is a weird combination of libraries? '
                                 f'{BackendTensor.engine_backend}, {BackendTensor.pykeops_enabled}, {solver}')

    return w


def _compute_conditional_number(cov):
    cond_number = np.linalg.cond(cov)
    svd = np.linalg.svd(cov)
    eigvals = np.linalg.eigvals(cov)
    is_positive_definite = np.all(eigvals > 0)
    print(f'Condition number: {cond_number}. Is positive definite: {is_positive_definite}')
    if not is_positive_definite:  # ! Careful numpy False
        warnings.warn('The covariance matrix is not positive definite')
