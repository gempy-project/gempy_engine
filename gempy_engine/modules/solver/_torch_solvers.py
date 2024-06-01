from ._custom_pykeops_solver import solve
from gempy_engine.core.backend_tensor import BackendTensor
from pykeops.torch import KernelSolve

bt = BackendTensor


def pykeops_torch_cg(b, cov, x0):
    if False:
        solver = cov.solve(
            b.view(-1, 1),
            alpha=0,
            backend="GPU",
            call=False,
            dtype_acc="float64",
            sum_scheme="kahan_scheme"

        )

        w = solver(eps=1e-5)
    else:
        solver = solve(
            cov,
            b.view(-1, 1),
            alpha=0,
            backend="CPU",
            call=False,
            dtype_acc="float64",
            sum_scheme="kahan_scheme"
        )

        w = solver(
            eps=1e-5,
            x0=BackendTensor.t.array(x0)
        )
    return w

def pykeops_torch_direct(b, cov):

    from linear_operator.operators import DenseLinearOperator
    A = DenseLinearOperator(cov)
    w = A.solve(b)
    return w


def torch_solve(b, cov):
    w = bt.t.linalg.solve(cov, b)
    return w
