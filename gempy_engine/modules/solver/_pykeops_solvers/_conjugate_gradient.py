from pykeops.common.utils import get_tools
from gempy_engine.core.backend_tensor import BackendTensor
import warnings


def ConjugateGradientSolver(binding, linop, b, eps=1e-6, x0=None,
                            regularization=None, preconditioning=None,
                            adaptive_tolerance=True, max_iterations=5000,
                            verbose=False
                            ):
    """
    Robust Conjugate Gradient solver for ill-conditioned linear systems using PyKeOps.
    
    Solves the linear system: linop(a) = b
    where linop represents a symmetric positive definite linear operator.
    
    Enhanced with stability features for ill-conditioned kriging matrices:
    - Adaptive regularization
    - Preconditioning support
    - Robust convergence criteria
    - Residual monitoring and restart capability
    
    Args:
        binding: PyKeOps backend binding (CPU/GPU)
        linop: Linear operator function (symmetric positive definite)
        b: Right-hand side vector/tensor
        eps: Base convergence tolerance (default: 1e-6)
        x0: Initial guess (optional, defaults to zero vector)
        regularization: Regularization strategy ('auto', 'fixed', None)
        preconditioning: Preconditioner function (optional)
        adaptive_tolerance: Whether to use adaptive convergence criteria
        max_iterations: Maximum iterations (default: 5000)
    
    Returns:
        a: Solution vector where linop(a) = b
    """
    # =============================================================================
    # INITIALIZATION AND STABILITY SETUP
    # =============================================================================

    tools = get_tools(binding)


    # --- inside INITIALIZATION AND STABILITY SETUP -----------------
    if adaptive_tolerance:
        b_norm = (b ** 2).sum().sqrt()
        # Relative part
        rel_thresh = eps * b_norm
        # Absolute part scaled by vector size
        abs_thresh = eps * tools.size(b)
        # Minimum practical threshold to avoid over-tightening
        min_thresh = 1e-4 * b_norm        # <- tweak to taste
        initial_residual_threshold = max(rel_thresh, abs_thresh, min_thresh)
        delta = initial_residual_threshold ** 2
    else:
        delta = tools.size(b) * eps ** 2


    # Initialize solution vector with better conditioning
    if x0 is not None:
        a = tools.copy(x0.to(BackendTensor.dtype_obj)).reshape(-1, 1) 

    else:
        # For ill-conditioned systems, start with small random perturbation
        # instead of pure zero to avoid getting stuck in numerical null space
        a = 0.001 * tools.randn_like(b) if hasattr(tools, 'randn_like') else 0 * b

    # =============================================================================
    # REGULARIZATION FOR ILL-CONDITIONED MATRICES
    # =============================================================================

    def regularized_linop(x):
        """Apply regularization to improve conditioning"""
        base_result = linop(x)

        if regularization == 'auto':
            # Adaptive regularization based on residual behavior
            reg_param = max(1e-8, eps * 0.1)  # Dynamic regularization
            return base_result + reg_param * x
        elif regularization == 'fixed':
            # Fixed Tikhonov regularization 
            reg_param = 1e-6  # Adjust based on your problem
            return base_result + reg_param * x
        else:
            return base_result

    # Use regularized operator if specified
    effective_linop = regularized_linop if regularization else linop

    # =============================================================================
    # PRECONDITIONING SETUP
    # =============================================================================

    if preconditioning is not None:
        # Apply preconditioning to both sides of the equation
        # M^(-1) * A * x = M^(-1) * b, where M is preconditioner
        b_preconditioned = preconditioning(b)

        def preconditioned_linop(x):
            return preconditioning(effective_linop(x))

        working_linop = preconditioned_linop
        working_b = b_preconditioned
    else:
        working_linop = effective_linop
        working_b = b

    # =============================================================================
    # ENHANCED CONJUGATE GRADIENT SETUP
    # =============================================================================

    # Compute initial residual
    r = tools.copy(working_b) - working_linop(a)
    nr2 = (r ** 2).sum()

    # Store initial residual for relative convergence check
    initial_nr2 = nr2

    if nr2 < delta:
        return a

    p = tools.copy(r)

    # =============================================================================
    # ENHANCED ITERATION CONTROL
    # =============================================================================

    k = 1
    prev_nr2 = nr2
    consecutive_divergence = 0
    divergence_tolerance = 10  # Reduced for ill-conditioned systems

    # Stagnation detection for ill-conditioned systems
    stagnation_threshold = 1e-12
    stagnation_counter = 0
    stagnation_tolerance = 50

    # Restart mechanism
    restart_threshold = max_iterations // 4  # Restart every 25% of max iterations
    last_restart = 0

    # Quality monitoring
    residual_history = []

    # =============================================================================
    # ROBUST CONJUGATE GRADIENT LOOP
    # =============================================================================

    while k < max_iterations:

        # -------------------------------------------------------------------------
        # CORE CG STEP WITH NUMERICAL STABILITY CHECKS
        # -------------------------------------------------------------------------

        Mp = working_linop(p)

        # Check for numerical breakdown in denominator
        denominator = (p * Mp).sum()
        if abs(denominator) < 1e-14:
            warnings.warn(f"Numerical breakdown detected at iteration {k}. "
                          f"Denominator too small: {denominator}")
            break

        alp = nr2 / denominator

        # Safeguard against excessive step sizes
        if abs(alp) > 1e6:
            warnings.warn(f"Excessive step size detected: {alp}. Reducing.")
            alp = alp / abs(alp) * 1e6  # Limit step size

        a += alp * p
        r -= alp * Mp
        nr2new = (r ** 2).sum()

        # -------------------------------------------------------------------------
        # ENHANCED CONVERGENCE CRITERIA
        # -------------------------------------------------------------------------

        # Relative convergence check (important for ill-conditioned systems)
        relative_residual = nr2new / initial_nr2
        absolute_residual = nr2new

        if adaptive_tolerance:
            # Use both absolute and relative criteria
            converged = (absolute_residual < delta) or (relative_residual < eps ** 2)
        else:
            converged = absolute_residual < delta

        if converged:
            print(f"Converged at iteration {k}")
            print(f"  Absolute residual: {absolute_residual:.2e}")
            print(f"  Relative residual: {relative_residual:.2e}")
            break

        # -------------------------------------------------------------------------
        # STAGNATION AND DIVERGENCE DETECTION
        # -------------------------------------------------------------------------

        residual_history.append(float(nr2new))

        # Check for stagnation (typical in ill-conditioned systems)
        residual_change = abs(nr2new - prev_nr2) / max(prev_nr2, 1e-16)
        if residual_change < stagnation_threshold:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        # Handle stagnation with restart
        if stagnation_counter >= stagnation_tolerance:
            if k - last_restart > restart_threshold:
                print(f"Stagnation detected at iteration {k}. Restarting CG...")
                # Restart: reset search direction to steepest descent
                p = tools.copy(r)
                stagnation_counter = 0
                last_restart = k
            else:
                print(f"Persistent stagnation detected. Algorithm may have converged "
                      f"to machine precision. Current residual: {nr2new:.2e}")
                break

        # Enhanced divergence detection
        if nr2new > prev_nr2 * 1.1:  # Allow small increases due to rounding
            consecutive_divergence += 1
            if consecutive_divergence >= divergence_tolerance:
                print(f"Algorithm diverging for {divergence_tolerance} iterations.")
                print(f"This may indicate severe ill-conditioning.")
                print(f"Consider stronger regularization or preconditioning.")
                break
        else:
            consecutive_divergence = 0

        # -------------------------------------------------------------------------
        # PREPARE NEXT ITERATION WITH STABILITY CHECKS
        # -------------------------------------------------------------------------

        beta = nr2new / nr2

        # Prevent beta from becoming too large (Fletcher-Reeves restart condition)
        if beta > 1.0:
            if verbose:
                print(f"Large beta detected ({beta:.2e}) at iteration {k}. Restarting.")
            p = tools.copy(r)  # Restart with steepest descent
        else:
            p = r + beta * p

        # Update for next iteration
        nr2 = nr2new
        prev_nr2 = nr2

        # -------------------------------------------------------------------------
        # PROGRESS MONITORING
        # -------------------------------------------------------------------------

        if k % 500 == 0:  # More frequent reporting for ill-conditioned systems
            print(f"Iteration {k:4d} | Residual: {nr2:.2e} | "
                  f"Relative: {relative_residual:.2e} | "
                  f"Stagnation: {stagnation_counter}")

        k += 1

    # =============================================================================
    # FINAL DIAGNOSTICS AND WARNINGS
    # =============================================================================

    final_relative_residual = nr2 / initial_nr2

    print(f"\n=== Solver Summary ===")
    print(f"Final iteration: {k}")
    print(f"Final absolute residual: {nr2:.2e}")
    print(f"Final relative residual: {final_relative_residual:.2e}")
    print(f"Convergence target: {delta:.2e}")

    # Warn about potential issues
    if k >= max_iterations:
        warnings.warn("Maximum iterations reached. Solution may not be converged.")

    if final_relative_residual > 1e-3:
        warnings.warn(f"High relative residual ({final_relative_residual:.2e}). "
                      f"Matrix may be severely ill-conditioned.")

    if len(residual_history) > 100:
        # Check convergence rate
        recent_residuals = residual_history[-50:]
        if len(recent_residuals) > 10:
            avg_reduction = (recent_residuals[0] / recent_residuals[-1]) ** (1 / len(recent_residuals))
            if avg_reduction < 1.01:
                warnings.warn("Very slow convergence detected. Consider preconditioning.")

    return a

