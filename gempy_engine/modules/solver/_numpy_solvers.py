import numpy as np

from gempy_engine import optional_dependencies
from gempy_engine.core.backend_tensor import BackendTensor

bt = BackendTensor


def pykeops_numpy_cg(b, cov, dtype):
    # ! Only Positive definite matrices are solved. Otherwise, the kernel gets stuck
    # * Very interesting: https://stats.stackexchange.com/questions/386813/use-the-rbf-kernel-to-construct-a-positive-definite-covariance-matrix
    w = cov.solve(
        np.asarray(b).astype(dtype),
        alpha=.00000,
        dtype_acc=dtype,
        backend="CPU"
    )

    return w


def numpy_solve(b, cov, dtype):
    w = BackendTensor.tfnp.linalg.solve(cov.astype(dtype), b[:, 0])
    return w


def numpy_cg(b, cov):
    if bt.use_gpu is False and BackendTensor.pykeops_enabled is True:
        cov.backend = 'CPU'

    from ._pykeops_solvers.incomplete_cholesky import ichol
    from ._pykeops_solvers.cg import cg
    
    scipy = optional_dependencies.require_scipy()

    sparse_cov = cov.copy()
    sparse_cov[np.abs(cov) < 1e-10] = 0

    sparse_cov = scipy.sparse.csc_matrix(sparse_cov, )
    conditioner = ichol(sparse_cov)

    A = scipy.sparse.linalg.aslinearoperator(cov)
    print(f'A size: {A.shape}')

    w = cg(
        A=A,
        M=conditioner,
        b=b[:, 0],
        maxiter=10000,
        rtol=.000005,  # * With this tolerance we do 8 iterations
        # x0=x0
    )
    w = np.atleast_2d(w).T
    return w


def numpy_gmres(b, cov):
    #
    from scipy.sparse.linalg import aslinearoperator, gmres
    A = aslinearoperator(cov)
    w, info = gmres(
        A=A,
        b=b[:, 0],
        maxiter=5,
        tol=1e-5
    )
    w = np.atleast_2d(w).T
    return w
