# %%
import numpy as np

M, N = 1000, 2000
x = np.random.rand(M, 2)
y = np.random.rand(N, 2)
from pykeops.numpy import LazyTensor
import pykeops

pykeops.verbose = True

x_i = LazyTensor(
    x[:, None, :]
)  # (M, 1, 2) KeOps LazyTensor, wrapped around the numpy array x
y_j = LazyTensor(
    y[None, :, :]
)  # (1, N, 2) KeOps LazyTensor, wrapped around the numpy array y

D_ij = ((x_i - y_j) ** 2)  # **Symbolic** (M, N) matrix of squared distances
foo = D_ij.sum_reduction(axis=0, backend="GPU")

print(foo)

pykeops.test_numpy_bindings()