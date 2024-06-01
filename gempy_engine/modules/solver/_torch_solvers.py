from ._pykeops_solvers.custom_pykeops_solver import custom_pykeops_solver
from ...core.backend_tensor import BackendTensor

bt = BackendTensor


def pykeops_torch_cg(b, cov, x0):
    if PYKEOPS_SOLVER:=True:
        solver = cov.custom_pykeops_solver(
            b.view(-1, 1),
            alpha=0,
            backend="GPU",
            call=False,
            dtype_acc="float64",
            sum_scheme="kahan_scheme"

        )

        w = solver(eps=1e-5)
    else:
        solver = custom_pykeops_solver(
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
    raise NotImplementedError
    from linear_operator.operators import KernelLinearOperator
    A = KernelLinearOperator(cov)
    w = A.solve(b)
    return w


def torch_solve(b, cov):
    w = bt.t.linalg.solver(cov, b)
    return w
