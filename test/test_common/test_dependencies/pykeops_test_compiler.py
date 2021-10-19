# %%
import numpy as np


from pykeops.numpy import LazyTensor
import pykeops
#pykeops.clean_pykeops()
pykeops.test_numpy_bindings()
# pykeops.verbose = True
#pykeops.config.build_type = 'Debug'
print(pykeops.config.gpu_available)



M, N = 1000000, 20000
x = np.random.rand(M, 2).astype("float32")
y = np.random.rand(N, 2).astype("float32")
x_i = LazyTensor(    x[:, None, :])  # (M, 1, 2) KeOps LazyTensor, wrapped around the numpy array x
y_j = LazyTensor(
    y[None, :, :]
)  # (1, N, 2) KeOps LazyTensor, wrapped around the numpy array y

D_ij = ((x_i - y_j) ** 2)  # **Symbolic** (M, N) matrix of squared distances
foo = D_ij.sum_reduction(axis=0, backend="GPU", dtype_acc = "float32", call=False)


#%%
foo()
print(foo())

#pykeops.test_numpy_bindings()