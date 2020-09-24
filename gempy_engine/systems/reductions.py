import numpy as np
from pykeops.numpy import LazyTensor as LazyTensor_np
from pykeops.numpy import Vi

from gempy_engine.systems.generators import tfnp


def solver(A: np.ndarray, b: np.ndarray):
    if False:
        # A = tfnp.make_ndarray(A)
        # b = tfnp.make_ndarray(b)
        a = A.astype('float64')
        x_1 = LazyTensor_np(a, axis=1)
        x_11 = x_1.sum(axis=1)
        x_2 = LazyTensor_np(b[:, None, :])
        w = x_11.solve(x_2, alpha=0.1)
        # x1 = Vi(a)
        # x2 = Vi(b)
        # w = x1.solve(x2, alpha=0.1)

    else:
        w = tfnp.linalg.solve(A, b)
        # w = tfnp.linalg.cholesky_solve(A, b)
        # w = tfnp.linalg.experimental.conjugate_gradient(A, b)

    return w
