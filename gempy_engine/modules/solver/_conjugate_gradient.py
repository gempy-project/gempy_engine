from pykeops.common.utils import get_tools

from gempy_engine.core.backend_tensor import BackendTensor


def ConjugateGradientSolver(binding, linop, b, eps=1e-6, x0=None):
    # Conjugate gradient algorithm to solve linear system of the form
    # Ma=b where linop is a linear operation corresponding
    # to a symmetric and positive definite matrix
    tools = get_tools(binding)
    delta = tools.size(b) * eps**2

    # Initialize 'a' with 'x0' if provided, otherwise as zero vector
    if x0 is not None:
        a = tools.copy(x0.to(BackendTensor.dtype_obj))
    else:
        a = 0 * b

    r = tools.copy(b) - linop(a)  # Update the residual based on the initial guess
    nr2 = (r**2).sum()
    if nr2 < delta:
        return a
    p = tools.copy(r)
    k = 1
    prev_nr2 = nr2  # Initialize previous nr2 for divergence check
    max_iterations = 5000
    divergence_tolerance = 20  # Number of consecutive iterations allowed for increase in residual
    consecutive_divergence = 0  # Counter for consecutive divergence

    while k < max_iterations:
        Mp = linop(p)
        alp = nr2 / (p * Mp).sum()
        a += alp * p
        r -= alp * Mp
        nr2new = (r**2).sum()

        # Check for convergence
        if nr2new < delta:
            break

        # Check for divergence
        if nr2new > prev_nr2:
            consecutive_divergence += 1
            if consecutive_divergence >= divergence_tolerance:
                print(f"Diverging for {divergence_tolerance} consecutive iterations at iteration {k}. Stopping algorithm.")
                break
        else:
            consecutive_divergence = 0  # Reset counter if no divergence in this iteration
        # Update for next iteration
        p = r + (nr2new / nr2) * p
        nr2 = nr2new
        prev_nr2 = nr2  # Update previous nr2

        # Print every 100 iterations
        if k % 1000 == 0:
            print(f"Iteration {k}, Residual Norm: {nr2}")

        k += 1

    # Print final number of iterations
    print(f"Final Iteration {k}, Residual Norm: {nr2}")
    return a


