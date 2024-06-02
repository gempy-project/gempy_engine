import warnings
from typing import Optional

from gempy_engine.core.data.kernel_classes.solvers import Solvers
from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends

import numpy as np

from ...core.data.options import KernelOptions
from ._numpy_solvers import pykeops_numpy_cg, numpy_solve, numpy_cg, numpy_gmres, pykeops_numpy_solve, numpy_cg_jacobi
from ._torch_solvers import torch_solve, pykeops_torch_cg, pykeops_torch_direct

bt = BackendTensor



global ii
ii = 0
def kernel_reduction(cov, b, kernel_options: KernelOptions, x0: Optional[np.ndarray] = None) -> np.ndarray:

    solver: Solvers = kernel_options.kernel_solver
    compute_condition_number = kernel_options.compute_condition_number
    
    # ? Maybe we should always compute the conditional_number no matter the branch
    dtype = BackendTensor.dtype
    match (BackendTensor.engine_backend, BackendTensor.pykeops_enabled, solver):
        case (AvailableBackends.PYTORCH, False, _):
            if kernel_options.compute_condition_number:
                cond_number = BackendTensor.t.linalg.cond(cov)
                print(f'Condition number: {cond_number}.')
            w = torch_solve(b, cov)
        case (AvailableBackends.PYTORCH, True, _):
            w = pykeops_torch_cg(b, cov, x0)
        case (AvailableBackends.numpy, True, Solvers.PYKEOPS_CG):
            w = pykeops_numpy_cg(b, cov, dtype)
        case (AvailableBackends.numpy, True, Solvers.DEFAULT):
            w = pykeops_numpy_solve(b, cov, dtype)
        case (AvailableBackends.numpy, False, Solvers.DEFAULT):
            w = numpy_solve(b, cov, dtype)
            if compute_condition_number:
                _compute_conditional_number(cov)
        case (AvailableBackends.numpy, _, Solvers.DEFAULT |Solvers.SCIPY_CG):
            w = numpy_cg(b, cov)
        case (AvailableBackends.numpy, _, Solvers.GMRES):
            w = numpy_gmres(b, cov)
        case _:
            raise AttributeError(f'There is a weird combination of libraries? '
                                 f'{BackendTensor.engine_backend}, {BackendTensor.pykeops_enabled}, {solver}')
    global ii
    foo = cov.sum(0)

    if False:
        np.save(f'cov_{ii}.npy', cov)
        np.save(f"w_{ii}.npy", w)
    else:
        bar = np.load(f'cov_{ii}.npy').sum(0)
        weights = np.load(f"w_{ii}.npy")
        foo_ = foo.reshape(-1) - bar
        weights_ = weights - w

    ii += 1

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




