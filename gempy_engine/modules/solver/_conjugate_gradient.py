from pykeops.common.utils import get_tools


def ConjugateGradientSolver(binding, linop, b, eps=1e-6, x0=None):
    # Conjugate gradient algorithm to solve linear system of the form
    # Ma=b where linop is a linear operation corresponding
    # to a symmetric and positive definite matrix
    tools = get_tools(binding)
    delta = tools.size(b) * eps**2

    # Initialize 'a' with 'x0' if provided, otherwise as zero vector
    if x0 is not None:
        a = tools.copy(x0)
    else:
        a = 0 * b

    r = tools.copy(b) - linop(a)  # Update the residual based on the initial guess
    nr2 = (r**2).sum()
    if nr2 < delta:
        return a
    p = tools.copy(r)
    k = 0
    prev_nr2 = nr2  # Initialize previous nr2 for divergence check
    max_iterations = 1000

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
            print(f"Diverging at iteration {k}. Stopping algorithm.")
            break

        # Update for next iteration
        p = r + (nr2new / nr2) * p
        nr2 = nr2new
        prev_nr2 = nr2  # Update previous nr2

        # Print every 100 iterations
        if k % 100 == 0:
            print(f"Iteration {k}, Residual Norm: {nr2}")

        k += 1

    return a


