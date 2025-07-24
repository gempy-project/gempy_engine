from ._pykeops_solvers.custom_pykeops_solver import custom_pykeops_solver
from ...core.backend_tensor import BackendTensor

bt = BackendTensor


def pykeops_torch_cg(b, cov, x0, use_gpu):
    if PYKEOPS_SOLVER:=False: # * Default pykeops solver. It is here as reference
        solver = cov.solver(
            b.view(-1, 1),
            alpha=0,
            backend="GPU",
            call=False,
            dtype_acc="float64",
            sum_scheme="kahan_scheme"

        )

        w = solver(eps=1e-5)
    else:  # * My solver and what we need to tweak
        solver = custom_pykeops_solver(
            cov,
            b.view(-1, 1),
            alpha=0,
            backend="GPU" if use_gpu else "CPU",
            call=False,
            dtype_acc="float64", # * For now we always use float 64 even on gpu
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
    w = bt.t.linalg.solve(cov, b)
    return w

''' 

tensor([[1.0000, 0.9922, 0.9916, 0.9920, 0.9920, 0.9918, 0.9919, 0.9920, 0.9917],
        [0.9922, 1.0000, 0.9919, 0.9915, 0.9914, 0.9914, 0.9919, 0.9918, 0.9914],
        [0.9916, 0.9919, 1.0000, 0.9911, 0.9917, 0.9920, 0.9915, 0.9917, 0.9916],
        [0.9920, 0.9915, 0.9911, 1.0000, 0.9913, 0.9918, 0.9920, 0.9924, 0.9911],
        [0.9920, 0.9914, 0.9917, 0.9913, 1.0000, 0.9919, 0.9916, 0.9916, 0.9920],
        [0.9918, 0.9914, 0.9920, 0.9918, 0.9919, 1.0000, 0.9916, 0.9921, 0.9917],
        [0.9919, 0.9919, 0.9915, 0.9920, 0.9916, 0.9916, 1.0000, 0.9921, 0.9919],
        [0.9920, 0.9918, 0.9917, 0.9924, 0.9916, 0.9921, 0.9921, 1.0000, 0.9912],
        [0.9917, 0.9914, 0.9916, 0.9911, 0.9920, 0.9917, 0.9919, 0.9912, 1.0000]],
       dtype
       '''