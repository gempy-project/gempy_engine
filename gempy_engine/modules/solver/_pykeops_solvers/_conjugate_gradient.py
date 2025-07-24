from pykeops.common.utils import get_tools
from gempy_engine.core.backend_tensor import BackendTensor


def ConjugateGradientSolver(binding, linop, b, eps=1e-6, x0=None):
    """
    Conjugate Gradient solver for linear systems using PyKeOps.
    
    Solves the linear system: linop(a) = b
    where linop represents a symmetric positive definite linear operator.
    
    This implementation is optimized for GPU computation and automatic differentiation
    through PyKeOps backend.
    
    Args:
        binding: PyKeOps backend binding (CPU/GPU)
        linop: Linear operator function (symmetric positive definite)
        b: Right-hand side vector/tensor
        eps: Convergence tolerance (default: 1e-6)
        x0: Initial guess (optional, defaults to zero vector)
    
    Returns:
        a: Solution vector where linop(a) = b
    """
    # =============================================================================
    # INITIALIZATION PHASE
    # =============================================================================
    
    # Get PyKeOps tools for tensor operations (backend-agnostic)
    tools = get_tools(binding)
    
    # Compute convergence threshold based on problem size and tolerance
    delta = tools.size(b) * eps**2
    
    # Initialize solution vector 'a'
    if x0 is not None:
        # Use provided initial guess, ensuring correct data type
        a = tools.copy(x0.to(BackendTensor.dtype_obj))
    else:
        # Start with zero vector (cold start)
        a = 0 * b

    # =============================================================================
    # CONJUGATE GRADIENT SETUP
    # =============================================================================
    
    # Compute initial residual: r = b - linop(a)
    r = tools.copy(b) - linop(a)
    
    # Calculate squared norm of residual for convergence check
    nr2 = (r**2).sum()
    
    # Early termination if already converged
    if nr2 < delta:
        return a
    
    # Initialize search direction (first iteration: p = r)
    p = tools.copy(r)
    
    # =============================================================================
    # ITERATION CONTROL PARAMETERS
    # =============================================================================
    
    k = 1  # Iteration counter
    prev_nr2 = nr2  # Previous residual norm for divergence detection
    max_iterations = 5000  # Maximum allowed iterations
    divergence_tolerance = 20  # Consecutive divergent iterations before stopping
    consecutive_divergence = 0  # Counter for consecutive divergence steps
    
    # =============================================================================
    # MAIN CONJUGATE GRADIENT LOOP
    # =============================================================================
    
    while k < max_iterations:
        
        # -------------------------------------------------------------------------
        # CORE CG STEP: Compute optimal step size
        # -------------------------------------------------------------------------
        
        # Matrix-vector product: Mp = linop(p)
        Mp = linop(p)
        
        # Optimal step size: α = r^T·r / p^T·linop(p)
        alp = nr2 / (p * Mp).sum()
        
        # Update solution: a = a + α·p
        a += alp * p
        
        # Update residual: r = r - α·linop(p)
        r -= alp * Mp
        
        # Compute new residual norm
        nr2new = (r**2).sum()
        
        # -------------------------------------------------------------------------
        # CONVERGENCE CHECK
        # -------------------------------------------------------------------------
        
        if nr2new < delta:
            print(f"Converged at iteration {k}, Final Residual Norm: {nr2new}")
            break
        
        # -------------------------------------------------------------------------
        # DIVERGENCE DETECTION AND HANDLING
        # -------------------------------------------------------------------------
        
        if nr2new > prev_nr2:
            # Residual is increasing - potential divergence
            consecutive_divergence += 1
            if consecutive_divergence >= divergence_tolerance:
                print(f"Algorithm diverging for {divergence_tolerance} consecutive iterations at iteration {k}.")
                print(f"Stopping early. Current residual norm: {nr2new}")
                break
        else:
            # Reset divergence counter if residual decreased
            consecutive_divergence = 0
        
        # -------------------------------------------------------------------------
        # PREPARE NEXT ITERATION
        # -------------------------------------------------------------------------
        
        # Compute conjugate direction coefficient: β = r_new^T·r_new / r_old^T·r_old
        beta = nr2new / nr2
        
        # Update search direction: p = r + β·p
        p = r + beta * p
        
        # Update residual norm for next iteration
        nr2 = nr2new
        prev_nr2 = nr2
        
        # -------------------------------------------------------------------------
        # PROGRESS MONITORING
        # -------------------------------------------------------------------------
        
        # Print progress every 1000 iterations
        if k % 1000 == 0:
            print(f"Iteration {k}, Residual Norm: {nr2}")
        
        k += 1
    
    # =============================================================================
    # FINALIZATION
    # =============================================================================
    
    # Report final status
    if k >= max_iterations:
        print(f"Maximum iterations ({max_iterations}) reached.")
    
    print(f"Final Iteration: {k}, Final Residual Norm: {nr2}")
    
    return a