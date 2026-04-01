from pykeops.common.utils import get_tools
from gempy_engine.core.backend_tensor import BackendTensor
import warnings
import warnings


def ConjugateGradientSolver(binding, linop, b, eps=1e-5, nugget=1e-8, x0=None,
                            check_freq=10, max_iterations=5000, verbose=False):
    tools = get_tools(binding)

    # ---------------------------------------------------------
    # 1. INITIALIZATION & SETUP
    # ---------------------------------------------------------

    # Clean wrapper for the linear operator (Matrix-Vector multiplication)
    def apply_A(x):
        Ax = linop(x)
        if nugget > 0.0:
            Ax += nugget * x  # Adds diagonal jitter for stability
        return Ax

    # Initialize solution vector 'a'
    if x0 is not None:
        a = tools.copy(x0.to(BackendTensor.dtype_obj)).reshape(-1, 1)
    else:
        # Small random perturbation avoids getting stuck in numerical null spaces
        a = 0.001 * tools.randn_like(b) if hasattr(tools, 'randn_like') else 0 * b

    # Compute initial residual (r = b - Ax)
    r = tools.copy(b) - apply_A(a)
    nr2 = (r ** 2).sum()

    # Calculate convergence thresholds based on the initial state
    initial_nr2_val = nr2.item()  # Force ONE sync at the very beginning
    target_threshold = initial_nr2_val * (eps ** 2)

    if initial_nr2_val < target_threshold:
        return a

    # Initialize search direction
    p = tools.copy(r)

    # Trackers
    k = 1
    prev_nr2_val = initial_nr2_val

    while k < max_iterations:
        # -- ASYNCHRONOUS TENSOR MATH (GPU runs free here) --
        Mp = apply_A(p)

        # Add tiny epsilon to denominator to prevent division by zero without CPU syncs
        denominator = (p * Mp).sum() + 1e-16
        alp = nr2 / denominator

        a = a + alp * p
        r = r - alp * Mp
        nr2new = (r ** 2).sum()

        beta = nr2new / nr2
        p = r + beta * p

        # Swap references for next iteration
        nr2 = nr2new
        k += 1

        # LAZY CHECK BLOCK
        if k % check_freq == 0:
            current_nr2_val = nr2new.item()

            # 1. Convergence Check
            if current_nr2_val < target_threshold:
                return a, True  # SUCCESS!

            # ---------------------------------------------------------
            # NEW: EARLY BAILOUT TRIGGERS
            # ---------------------------------------------------------

            # Trigger A: The Flatline (Check at iteration 20)
            if k == check_freq * 2:
                reduction_rate = current_nr2_val / initial_nr2_val
                if reduction_rate > 0.5:  # Error hasn't even halved in 20 steps
                    if verbose: print(f"Bailout at {k}: Matrix too stiff. (Reduction: {reduction_rate:.2f})")
                    return None, False  # ABORT CG

            # Trigger B: Immediate Divergence
            if current_nr2_val > initial_nr2_val * 2.0:
                if verbose: print(f"Bailout at {k}: Algorithm diverging instantly.")
                return None, False  # ABORT CG

            # Trigger C: Persistent Stagnation (Slightly more aggressive now)
            residual_change = abs(current_nr2_val - prev_nr2_val) / max(prev_nr2_val, 1e-16)
            if residual_change < 1e-6:
                if verbose: print(f"Bailout at {k}: Solver hit a mathematical flatline.")
                return None, False  # ABORT CG

            # 4. Fletcher-Reeves Safety Override
            if beta.item() > 1.0:
                p = tools.copy(r)
                
            prev_nr2_val = current_nr2_val

    return a, False  # Max iterations reached without converging -> Fail


def ConjugateGradientSolver_(binding, linop, b, eps=1e-5, x0=None,
                             nugget=1e-5, preconditioning=None,
                             check_freq=10, max_iterations=5000,
                             verbose=False):
    """
    Robust, Asynchronous Conjugate Gradient solver for PyKeOps.

    Uses 'Lazy Checking' to prevent CPU-GPU synchronization bottlenecks.
    The GPU computes `check_freq` iterations completely asynchronously 
    before the CPU pauses to check for convergence or stagnation.
    """
    tools = get_tools(binding)

    # ---------------------------------------------------------
    # 1. INITIALIZATION & SETUP
    # ---------------------------------------------------------

    # Define the base target vector
    working_b = preconditioning(b) if preconditioning else b

    # Clean wrapper for the linear operator (Matrix-Vector multiplication)
    def apply_A(x):
        Ax = linop(x)
        if nugget > 0.0:
            Ax += nugget * x  # Adds diagonal jitter for stability
        if preconditioning:
            Ax = preconditioning(Ax)
        return Ax

    # Initialize solution vector 'a'
    if x0 is not None:
        a = tools.copy(x0.to(BackendTensor.dtype_obj)).reshape(-1, 1)
    else:
        # Small random perturbation avoids getting stuck in numerical null spaces
        a = 0.001 * tools.randn_like(b) if hasattr(tools, 'randn_like') else 0 * b

    # Compute initial residual (r = b - Ax)
    r = tools.copy(working_b) - apply_A(a)
    nr2 = (r ** 2).sum()

    # Calculate convergence thresholds based on the initial state
    initial_nr2_val = nr2.item()  # Force ONE sync at the very beginning
    target_threshold = initial_nr2_val * (eps ** 2)

    if initial_nr2_val < target_threshold:
        return a

    # Initialize search direction
    p = tools.copy(r)

    # ---------------------------------------------------------
    # 2. TRACKERS FOR LAZY CHECKING
    # ---------------------------------------------------------
    k = 1
    prev_nr2_val = initial_nr2_val
    stagnation_counter = 0
    consecutive_divergence = 0
    last_restart = 0

    # Tuning parameters for stability
    stagnation_threshold = 1e-12
    stagnation_tolerance = 50
    restart_limit = max_iterations // 4

    # ---------------------------------------------------------
    # 3. THE ASYNCHRONOUS CG LOOP
    # ---------------------------------------------------------
    while k < max_iterations:

        # -- ASYNCHRONOUS TENSOR MATH (GPU runs free here) --
        Mp = apply_A(p)

        # Add tiny epsilon to denominator to prevent division by zero without CPU syncs
        denominator = (p * Mp).sum() + 1e-16
        alp = nr2 / denominator

        a = a + alp * p
        r = r - alp * Mp
        nr2new = (r ** 2).sum()

        beta = nr2new / nr2
        p = r + beta * p

        # Swap references for next iteration
        nr2 = nr2new
        k += 1

        # -- LAZY CHECK BLOCK (CPU syncs only every `check_freq` iterations) --
        if k % check_freq == 0:

            # .item() pulls the scalar to the CPU, forcing the hardware to sync
            current_nr2_val = nr2new.item()

            # 1. Convergence Check
            if current_nr2_val < target_threshold:
                if verbose:
                    print(f"Converged at iteration {k} | Relative Residual: {current_nr2_val / initial_nr2_val:.2e}")
                break

            # 2. Stagnation Check
            residual_change = abs(current_nr2_val - prev_nr2_val) / max(prev_nr2_val, 1e-16)
            if residual_change < stagnation_threshold:
                stagnation_counter += check_freq
            else:
                stagnation_counter = 0

            if stagnation_counter >= stagnation_tolerance:
                if k - last_restart > restart_limit:
                    if verbose: print(f"Stagnation at {k}. Restarting search direction...")
                    p = tools.copy(r)  # Reset to steepest descent
                    stagnation_counter = 0
                    last_restart = k
                else:
                    warnings.warn(f"Persistent stagnation. Stopping at {current_nr2_val:.2e}")
                    break

            # 3. Divergence Check
            if current_nr2_val > prev_nr2_val * 1.5:
                consecutive_divergence += check_freq
                if consecutive_divergence >= 30:  # Hardcoded tolerance for divergence
                    warnings.warn("Algorithm diverging. Matrix may be too ill-conditioned.")
                    break
            else:
                consecutive_divergence = 0

            # 4. Fletcher-Reeves Safety Override
            if beta.item() > 1.0:
                p = tools.copy(r)

            # Update tracker for the next check block
            prev_nr2_val = current_nr2_val

    # ---------------------------------------------------------
    # 4. FINAL DIAGNOSTICS
    # ---------------------------------------------------------
    if k >= max_iterations:
        warnings.warn("Maximum iterations reached without full convergence.")

    return a


def ConjugateGradientSolver__(binding, linop, b, eps=1e-6, x0=None,
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
        min_thresh = 1e-4 * b_norm  # <- tweak to taste
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
