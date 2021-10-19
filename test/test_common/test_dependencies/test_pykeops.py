import pytest
import sys
sys.path.append("/home/miguel/libkeops")

from pykeops.numpy import LazyTensor, Genred, Vi, Vj, Pm
import numpy as np
import pykeops
pykeops.config.verbose = True
pykeops.config.build_type = 'Debug'

@pytest.mark.skip('Only trigger manually when there is something wrong with pykeops compilation', )
def test_keops_run():

    pykeops.verbose = True
    #pykeops.set_bin_folder("/home/miguel/.s")
 #   pykeops.clean_pykeops()  # just in case old build files are still present
    pykeops.test_numpy_bindings()

@pytest.mark.skip('Only trigger manually when there is something wrong with'
                  'pykeops compilation', )
def test_basic_op():
    #pykeops.config.build_type = 'Debug'
    pykeops.test_numpy_bindings()
    print(pykeops.config.gpu_available)

    M, N = 1000, 2000
    x = np.random.rand(M, 2).astype("float32")
    y = np.random.rand(N, 2).astype("float32")
    # pykeops.clean_pykeops()
    x_i = LazyTensor(
        x[:, None, :]
    )  # (M, 1, 2) KeOps LazyTensor, wrapped around the numpy array x
    y_j = LazyTensor(
        y[None, :, :]
    )  # (1, N, 2) KeOps LazyTensor, wrapped around the numpy array y

    D_ij = ((x_i - y_j) ** 2)  # **Symbolic** (M, N) matrix of squared distances
    foo = D_ij.sum_reduction(axis=0, backend="GPU", dtype_acc="float32")

    print(foo)
