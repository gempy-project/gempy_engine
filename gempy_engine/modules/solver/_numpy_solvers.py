import numpy as np
from gempy_engine.core.backend_tensor import BackendTensor
from scipy.sparse.linalg import aslinearoperator, cg, spsolve

bt = BackendTensor

global _n_iters

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

def pykeops_numpy_solve(b, cov, dtype):
    # ! Only Positive definite matrices are solved. Otherwise, the kernel gets stuck
    # * Very interesting: https://stats.stackexchange.com/questions/386813/use-the-rbf-kernel-to-construct-a-positive-definite-covariance-matrix
    A = aslinearoperator(cov)
    w = spsolve(A, b[:, 0])
    return w


def numpy_solve(b, cov, dtype):
    w = BackendTensor.tfnp.linalg.solve(cov.astype(dtype), b[:, 0])
    return w


def numpy_cg(b, cov, ):
    n_iters = 0
    if bt.use_gpu is False and BackendTensor.pykeops_enabled is True:
        cov.backend = 'CPU'
    A = aslinearoperator(cov)
    print(f'A size: {A.shape}')
    w, info = cg(
        A=A,
        b=b[:, 0],
        maxiter=1000,
        tol=.000005,  # * With this tolerance we do 8 iterations
        callback=callback,
        # x0=x0
    )
    w = np.atleast_2d(w).T
    print(f'CG iterations: {n_iters}')
    return w

def numpy_gmres(b, cov):
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



def callback(xk):
    global n_iters
    n_iters += 1

