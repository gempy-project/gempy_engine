import warnings
from typing import Optional

from gempy_engine.core.data.kernel_classes.solvers import Solvers
from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends

import numpy as np

from gempy_engine.core.data.options import KernelOptions

bt = BackendTensor
global n_iters



def kernel_reduction(cov, b, kernel_options: KernelOptions, n_faults: int = 0, x0: Optional[BackendTensor.dtype_obj] = None) -> np.ndarray:
    global n_iters
    n_iters = 0
    
    solver: Solvers = kernel_options.kernel_solver
    compute_condition_number = kernel_options.compute_condition_number
    
    # ? Maybe we should always compute the conditional_number no matter the branch
    dtype = BackendTensor.dtype
    match (BackendTensor.engine_backend, BackendTensor.pykeops_enabled, solver):
        case (AvailableBackends.PYTORCH, False, _):
            if kernel_options.compute_condition_number:
                cond_number = BackendTensor.t.linalg.cond(cov)
                print(f'Condition number: {cond_number}.')

            w = bt.t.linalg.solve(cov, b)
            
        case (AvailableBackends.PYTORCH, True, _):
            from pykeops.torch import KernelSolve
            
           
            if False:
                solver = cov.solve(
                    b.view(-1,1), 
                    alpha=0,
                    backend="GPU",
                    call=False,
                    dtype_acc="float64",
                    sum_scheme="kahan_scheme"
                    
                )
                
                w = solver(eps=1e-5)
            else:
                from ._custom_pykeops_solver import solve
                solver = solve(
                    cov,
                    b.view(-1, 1),
                    alpha=0,
                    backend="GPU",
                    call=False,
                    dtype_acc="float64",
                    sum_scheme="kahan_scheme"
                )

                w = solver(
                    eps=1e-5, 
                    x0=x0
                )

        case (AvailableBackends.tensorflow, True, _):
            raise NotImplementedError('Pykeops is not implemented for tensorflow yet')
            # w = cov.solve(b.numpy().astype('float32'), alpha=smooth, dtype_acc='float32')
        case (AvailableBackends.tensorflow, False, _):
            import tensorflow as tf
            w = tf.linalg.solve(cov, b)
        case (AvailableBackends.numpy, True, Solvers.PYKEOPS_CG):
            # ! Only Positive definite matrices are solved. Otherwise, the kernel gets stuck
            # * Very interesting: https://stats.stackexchange.com/questions/386813/use-the-rbf-kernel-to-construct-a-positive-definite-covariance-matrix
            w = cov.solve(
                np.asarray(b).astype(dtype),
                alpha=.00000,
                dtype_acc=dtype,
                backend="CPU"
            )
        case (AvailableBackends.numpy, False, Solvers.DEFAULT):
            w = bt.tfnp.linalg.solve(cov.astype(dtype), b[:, 0])

            if compute_condition_number:
                _compute_conditional_number(cov)

        case (AvailableBackends.numpy, _, Solvers.DEFAULT |Solvers.SCIPY_CG):
            from scipy.sparse.linalg import aslinearoperator, cg
            if bt.use_gpu is False and BackendTensor.pykeops_enabled is True: 
                cov.backend = 'CPU'
                
            A = aslinearoperator(cov)
            print(f'A size: {A.shape}')
            w, info = cg(
                A=A,
                b=b[:, 0],
                maxiter=100,
                tol=.000005,  # * With this tolerance we do 8 iterations
                callback=callback,
                # x0=x0
            )
            w = np.atleast_2d(w).T
            print(f'CG iterations: {n_iters}')
        
        case (AvailableBackends.numpy, _, Solvers.GMRES):
            from scipy.sparse.linalg import aslinearoperator, gmres
            A = aslinearoperator(cov)
            w, info = gmres(
                A=A,
                b=b[:, 0],
                maxiter=5,
                tol=1e-5
            )
            w = np.atleast_2d(w).T

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
    
    idx = np.where(eigvals > 800)
    print(idx)
    import matplotlib.pyplot as plt
    if not is_positive_definite:  # ! Careful numpy False
        warnings.warn('The covariance matrix is not positive definite')
    # Plotting the histogram
    plt.hist(eigvals, bins=50, color='blue', alpha=0.7, log=True)
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.title('Histogram of Eigenvalues')
    plt.show()



def callback(xk):
    global n_iters
    n_iters += 1

