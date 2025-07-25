from pykeops.common.utils import get_tools

# =============================================================================
# HELPER FUNCTIONS FOR PRECONDITIONING
# =============================================================================

def diagonal_preconditioner(binding, linop, x_sample=None):
    """
    Create a simple diagonal preconditioner for ill-conditioned systems.

    This creates an approximate diagonal preconditioner by extracting diagonal elements
    of the linear operator. For kriging matrices, this can significantly improve
    conditioning and convergence rates.

    Args:
        binding: PyKeOps backend ('torch', 'numpy', etc.)
        linop: Linear operator function
        x_sample: Sample vector to determine size (optional)

    Returns:
        Preconditioner function that applies M^(-1) where M ≈ diag(A)
    """
    tools = get_tools(binding)

    # Determine the size of the problem
    if x_sample is not None:
        n = tools.size(x_sample)
        sample_shape = x_sample.shape
    else:
        # We need to determine the size somehow - this is tricky without a sample
        # For now, we'll create a simple fallback
        raise ValueError("x_sample is required for diagonal preconditioner construction")

    # Extract diagonal elements by applying linop to unit vectors
    diagonal_elements = []

    for i in range(n):
        # Create i-th unit vector
        if binding == 'torch':
            # Use torch-specific operations from torchtools
            ei = tools.zeros(sample_shape, dtype=x_sample.dtype, device=x_sample.device)
            if len(sample_shape) == 1:
                ei[i] = 1.0
            else:
                # Handle multi-dimensional case
                flat_ei = ei.view(-1)
                flat_ei[i] = 1.0
                ei = flat_ei.view(sample_shape)
        else:
            # For other backends, use generic approach
            ei = tools.zeros_like(x_sample) if x_sample is not None else tools.zeros(n)
            if len(ei.shape) == 1:
                ei[i] = 1.0
            else:
                flat_ei = ei.view(-1)
                flat_ei[i] = 1.0

        # Apply linear operator to get i-th column
        Aei = linop(ei)

        # Extract diagonal element (i-th component of i-th column)
        if len(Aei.shape) == 1:
            diag_elem = Aei[i]
        else:
            flat_Aei = Aei.view(-1)
            diag_elem = flat_Aei[i]

        diagonal_elements.append(diag_elem)

    # Stack diagonal elements into a vector
    if binding == 'torch':
        import torch
        diagonal = torch.stack(diagonal_elements)
    else:
        # For other backends
        diagonal = tools.array(diagonal_elements)

    # Avoid division by very small numbers (regularize diagonal)
    min_diag_value = 1e-12
    if binding == 'torch':
        import torch
        diagonal = torch.maximum(torch.abs(diagonal),
                                 torch.full_like(diagonal, min_diag_value))
    else:
        # Generic approach - may need adjustment for other backends
        diagonal = tools.maximum(tools.abs(diagonal) if hasattr(tools, 'abs') else diagonal,
                                 min_diag_value)

    def preconditioner(x):
        """
        Apply diagonal preconditioning: M^(-1) * x where M = diag(A)
        """
        if len(x.shape) == len(diagonal.shape):
            return x / diagonal
        else:
            # Handle shape mismatches
            if len(x.shape) > len(diagonal.shape):
                # x is multi-dimensional, diagonal is 1D
                diag_expanded = diagonal.view(-1)
                x_flat = x.view(-1)
                result_flat = x_flat / diag_expanded
                return result_flat.view(x.shape)
            else:
                # Should not happen in typical use cases
                return x / diagonal

    return preconditioner


def create_jacobi_preconditioner(binding, linop, x_sample=None, damping=0.8):
    """
    Create a Jacobi (diagonal) preconditioner with damping for better stability.

    This is often more robust than pure diagonal preconditioning for 
    ill-conditioned kriging systems.

    Args:
        binding: PyKeOps backend
        linop: Linear operator
        x_sample: Sample vector
        damping: Damping factor (0 < damping <= 1)

    Returns:
        Damped Jacobi preconditioner function
    """
    diagonal_prec = diagonal_preconditioner(binding, linop, x_sample)

    def jacobi_preconditioner(x):
        # Damped Jacobi: M^(-1) = damping * D^(-1) + (1-damping) * I
        preconditioned = diagonal_prec(x)
        return damping * preconditioned + (1 - damping) * x

    return jacobi_preconditioner


def create_adaptive_preconditioner(binding, linop, x_sample=None):
    """
    Create an adaptive preconditioner that adjusts based on the problem characteristics.

    For kriging matrices, this tries different strategies and picks the most effective one.
    """
    tools = get_tools(binding)

    # Try to estimate the condition number heuristically
    if x_sample is not None:
        # Apply operator to a few random vectors to estimate spectral properties
        test_vectors = []
        results = []

        for _ in range(min(5, tools.size(x_sample))):
            if binding == 'torch':
                import torch
                test_vec = torch.randn_like(x_sample)
            else:
                # For other backends, create simple perturbation
                test_vec = x_sample * (1 + 0.1 * (_ + 1))  # Simple variation

            test_vectors.append(test_vec)
            results.append(linop(test_vec))

        # Estimate if the system is severely ill-conditioned
        result_norms = [tools.norm(r) if hasattr(tools, 'norm') else (r ** 2).sum().sqrt()
                        for r in results]
        input_norms = [tools.norm(v) if hasattr(tools, 'norm') else (v ** 2).sum().sqrt()
                       for v in test_vectors]

        condition_estimates = [r_norm / i_norm for r_norm, i_norm in zip(result_norms, input_norms)]
        avg_condition = sum(condition_estimates) / len(condition_estimates)

        print(f"Estimated condition number range: {min(condition_estimates):.2e} - {max(condition_estimates):.2e}")

        # Choose preconditioning strategy based on estimated conditioning
        if avg_condition > 1e6:
            print("Severely ill-conditioned system detected. Using damped Jacobi preconditioner.")
            return create_jacobi_preconditioner(binding, linop, x_sample, damping=0.6)
        elif avg_condition > 1e3:
            print("Moderately ill-conditioned system. Using standard diagonal preconditioner.")
            return diagonal_preconditioner(binding, linop, x_sample)
        else:
            print("Well-conditioned system. Using light diagonal preconditioning.")
            return create_jacobi_preconditioner(binding, linop, x_sample, damping=0.9)

    # Fallback to standard diagonal preconditioner
    return diagonal_preconditioner(binding, linop, x_sample)

