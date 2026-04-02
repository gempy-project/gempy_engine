import warnings
from typing import Optional

import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends
from gempy_engine.core.data.kernel_classes.solvers import Solvers
from ._numpy_solvers import numpy_solve, numpy_cg, numpy_gmres
from ._torch_solvers import torch_solve, pykeops_torch_cg
from ...core.data.options import KernelOptions

bt = BackendTensor


def kernel_reduction(cov, b, kernel_options: KernelOptions, x0: Optional[np.ndarray] = None) -> np.ndarray:

    solver: Solvers = kernel_options.kernel_solver
    # ? Maybe we should always compute the conditional_number no matter the branch
    dtype = BackendTensor.dtype
    match (BackendTensor.engine_backend, BackendTensor.pykeops_enabled, solver):
        case (AvailableBackends.PYTORCH, False, _):
            if kernel_options.compute_condition_number:
                cond_number = BackendTensor.t.linalg.cond(cov)
                kernel_options.condition_number = cond_number
                print(f'Condition number: {cond_number}.')
            w = torch_solve(b, cov)
        case (AvailableBackends.PYTORCH, True, _):
            if len(x0) == 0:
                x0 = None
            w = pykeops_torch_cg(b, cov, x0, bt.use_gpu)
        case (AvailableBackends.numpy, False, Solvers.DEFAULT):
            w = numpy_solve(b, cov, dtype)
            if kernel_options.compute_condition_number:
                kernel_options.condition_number = _compute_conditional_number(cov)
        case (AvailableBackends.numpy, False, Solvers.DEFAULT |Solvers.SCIPY_CG):
            w = numpy_cg(b, cov)
        case (AvailableBackends.numpy, False, Solvers.GMRES):
            w = numpy_gmres(b, cov)
        case _:
            raise AttributeError(f'There is a weird combination of libraries? '
                                 f'{BackendTensor.engine_backend}, {BackendTensor.pykeops_enabled}, {solver}')

    return w




def _compute_conditional_number(cov, plot=False):
    cond_number = np.linalg.cond(cov)
    svd = np.linalg.svd(cov)
    eigvals = np.linalg.eigvals(cov)
    is_positive_definite = np.all(eigvals > 0)
    print(f'Condition number: {cond_number}. Is positive definite: {is_positive_definite}')
    
    idx = np.where(eigvals > 800)
    print(idx)
    if not is_positive_definite:  # ! Careful numpy False
        warnings.warn('The covariance matrix is not positive definite')
    if plot:
        import matplotlib.pyplot as plt
        # Plotting the histogram
        plt.hist(eigvals, bins=50, color='blue', alpha=0.7, log=True)
        plt.xlabel('Eigenvalue')
        plt.ylabel('Frequency')
        plt.title('Histogram of Eigenvalues')
        plt.show()
        
    return cond_number




