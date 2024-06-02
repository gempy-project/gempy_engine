import numpy as np
from gempy_engine.core.backend_tensor import BackendTensor
from scipy.sparse.linalg import aslinearoperator, cg, spsolve

from gempy_engine.modules.solver._pykeops_solvers.custom_pykeops_solver import custom_pykeops_solver

bt = BackendTensor

global n_iters
n_iters = 0

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
    # A = aslinearoperator(cov)
    # w = spsolve(A, b[:, 0])
    # TODO: Clean this up

    from linear_operator.operators import DenseLinearOperator

    w = DenseLinearOperator(cov).solve(b)

    return w


def numpy_solve(b, cov, dtype):
    w = BackendTensor.tfnp.linalg.solve(cov.astype(dtype), b[:, 0])
    return w


def numpy_cg(b, cov, ):
    if bt.use_gpu is False and BackendTensor.pykeops_enabled is True:
        cov.backend = 'CPU'
    A = aslinearoperator(cov)
    print(f'A size: {A.shape}')
    w, info = cg(
        A=A,
        b=b[:, 0],
        maxiter=5000,
        tol=.000005,  # * With this tolerance we do 8 iterations
        callback=callback,
        # x0=x0
    )
    w = np.atleast_2d(w).T
    print(f'CG iterations: {n_iters}')
    return w


def numpy_cg_jacobi(b, cov):
    """In the model I tested the preconditioner was breaking it more than helping it. It was not converging."""
    if False: # * For pykeops
        matrix_shape = cov.shape[0]
        from pykeops.numpy import LazyTensor
        diag_ = np.arange(matrix_shape).reshape(-1, 1).astype(BackendTensor.dtype)
        diag_i = LazyTensor(diag_[:, None])
        diag_j = LazyTensor(diag_[None, :])
        foo = ((0.5 - (diag_i - diag_j)**2).step())
        M = foo/((foo * cov) + 1e-10)
        ra = M.sum(0)
    else:
        # Assuming cov is a dense matrix or can provide access to its diagonal
        if hasattr(cov, 'diagonal'):
            diag = cov.diagonal()  # If cov is a numpy array
        else:
            # If cov is a more complex type, like a LinearOperator, this needs custom handling
            diag = np.array([cov[i, i] for i in range(cov.shape[0])])

        # Handling small or zero diagonal elements
        small_number = 1e-10
        diag = np.where(diag < small_number, small_number, diag)

        # Create the Jacobi preconditioner (inverse of the diagonal)
        M = np.diag(1.0 / diag)
        ra = M.sum(0)

    M = aslinearoperator(M)
    A = aslinearoperator(cov)
    print(f'A size: {A.shape}')

    # Perform the CG with the Jacobi preconditioner
    w, info = cg(
        A=A,
        b=b[:, 0],
        M=M,  # Include the preconditioner
        maxiter=5000,
        tol=.000005,  # With this tolerance we do 8 iterations
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



