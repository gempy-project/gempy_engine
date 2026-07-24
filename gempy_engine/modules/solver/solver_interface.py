from typing import Optional

import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends
from gempy_engine.core.data.kernel_classes.solvers import Solvers
from ._numpy_solvers import numpy_solve, numpy_cg, numpy_gmres
from ._torch_solvers import torch_solve, pykeops_torch_cg
from ...core.data.options import KernelOptions
from ..kernel_constructor.execution_mode import KernelExecutionMode

bt = BackendTensor


def kernel_reduction(
        cov,
        b,
        kernel_options: KernelOptions,
        x0: Optional[np.ndarray] = None,
        execution_mode: KernelExecutionMode = KernelExecutionMode.DENSE,
) -> np.ndarray:

    solver: Solvers = kernel_options.kernel_solver
    # ? Maybe we should always compute the conditional_number no matter the branch
    dtype = BackendTensor.dtype
    match (BackendTensor.engine_backend, execution_mode, solver):
        case (AvailableBackends.PYTORCH, KernelExecutionMode.DENSE, _):
            w = torch_solve(b, cov)
        case (AvailableBackends.PYTORCH, KernelExecutionMode.PYKEOPS, _):
            if x0 is not None and len(x0) == 0:
                x0 = None
            w = pykeops_torch_cg(b, cov, x0, bt.use_gpu)
        case (AvailableBackends.numpy, KernelExecutionMode.DENSE, Solvers.DEFAULT):
            w = numpy_solve(b, cov, dtype)
        case (AvailableBackends.numpy, KernelExecutionMode.DENSE, Solvers.DEFAULT |Solvers.SCIPY_CG):
            w = numpy_cg(b, cov)
        case (AvailableBackends.numpy, KernelExecutionMode.DENSE, Solvers.GMRES):
            w = numpy_gmres(b, cov)
        case _:
            raise AttributeError(f'There is a weird combination of libraries? '
                                 f'{BackendTensor.engine_backend}, {execution_mode}, {solver}')

    return w
