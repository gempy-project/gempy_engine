from pykeops.common.utils import get_tools
import warnings
import torch


# =============================================================================
# HELPER FUNCTIONS FOR PRECONDITIONING
# =============================================================================

def create_adaptive_nystrom_preconditioner(binding, linop, x_sample,
                                           strategy='conservative', **kwargs):
    """
    Create an adaptive Nyström preconditioner with automatic parameter tuning.

    Args:
        binding: PyKeOps backend
        linop: Linear operator (kernel matrix)  
        x_sample: Sample vector
        strategy: 'aggressive', 'conservative', or 'minimal'
        **kwargs: Additional parameters for fine-tuning

    Returns:
        Adaptive Nyström preconditioner
    """
    tools = get_tools(binding)
    n = tools.size(x_sample)

    print(f"Creating adaptive Nyström preconditioner (strategy: {strategy})")

    # Set parameters based on strategy
    if strategy == 'aggressive':
        params = {
                'rank'             : min(n // 5, 200),
                'max_rank'         : min(n // 3, 500),
                'tolerance'        : 1e-8,
                'pivoting_strategy': 'greedy'
        }
    elif strategy == 'conservative':
        params = {
                'rank'             : min(n // 10, 100),
                'max_rank'         : min(n // 4, 250),
                'tolerance'        : 1e-6,
                'pivoting_strategy': 'greedy'
        }
    else:  # minimal
        params = {
                'rank'             : min(n // 20, 50),
                'max_rank'         : min(n // 8, 100),
                'tolerance'        : 1e-4,
                'pivoting_strategy': 'diagonal'
        }

    # Override with user-provided parameters
    params.update(kwargs)

    print(f"Nyström parameters: {params}")

    # Try Nyström with fallback
    try:
        return nystrom_preconditioner(binding, linop, x_sample, **params)
    except Exception as e:
        print(f"Adaptive Nyström failed: {e}")
        print("Falling back to identity preconditioner")
        return _create_identity_preconditioner()



def nystrom_preconditioner(binding, linop, x_sample, rank=None, tolerance=1e-6,
                           max_rank=None, pivoting_strategy='greedy'):
    """
    Create a Nyström (pivoted Cholesky) preconditioner for kernel matrices.
    
    This method constructs a low-rank approximation of the kernel matrix:
    K ≈ C * C^T where C is n × rank
    
    The preconditioner is then M^(-1) = (C * C^T + λI)^(-1)
    
    Args:
        binding: PyKeOps backend ('torch', 'numpy', etc.)
        linop: Linear operator function (kernel matrix)
        x_sample: Sample vector to determine size and structure
        rank: Fixed rank for approximation (optional)
        tolerance: Tolerance for adaptive rank selection
        max_rank: Maximum rank to consider
        pivoting_strategy: 'greedy', 'random', or 'diagonal'
    
    Returns:
        Nyström preconditioner function
    """
    tools = get_tools(binding)
    n = tools.size(x_sample)

    print(f"Creating Nyström preconditioner for system size: {n}")

    # Set reasonable defaults for rank
    if max_rank is None:
        max_rank = min(n // 4, 500)  # Limit computational cost

    if rank is None:
        rank = min(n // 10, 100)  # Start with modest rank

    print(f"Target rank: {rank}, Max rank: {max_rank}")

    try:
        # Step 1: Select pivot points
        pivot_indices = _select_pivots(binding, linop, x_sample, rank,
                                       pivoting_strategy, max_rank, tolerance)

        actual_rank = len(pivot_indices)
        print(f"Selected {actual_rank} pivot points")

        # Step 2: Construct the Nyström approximation
        C = _construct_nystrom_factor(binding, linop, x_sample, pivot_indices)

        # Step 3: Create the preconditioner
        preconditioner = _create_nystrom_preconditioner(binding, C, tolerance)

        return preconditioner

    except Exception as e:
        print(f"Nyström preconditioner construction failed: {e}")
        print("Falling back to identity preconditioner")
        return _create_identity_preconditioner()


def _select_pivots(binding, linop, x_sample, target_rank, strategy, max_rank, tolerance):
    """
    Select pivot points for Nyström approximation using various strategies.
    """
    tools = get_tools(binding)
    n = tools.size(x_sample)

    if strategy == 'greedy':
        return _greedy_pivot_selection(binding, linop, x_sample, target_rank, max_rank, tolerance)
    elif strategy == 'random':
        return _random_pivot_selection(n, target_rank)
    elif strategy == 'diagonal':
        return _diagonal_pivot_selection(binding, linop, x_sample, target_rank)
    else:
        raise ValueError(f"Unknown pivoting strategy: {strategy}")


def _greedy_pivot_selection(binding, linop, x_sample, target_rank, max_rank, tolerance):
    """
    Greedy pivot selection based on diagonal residuals (pivoted Cholesky).
    """
    tools = get_tools(binding)
    n = tools.size(x_sample)

    selected_pivots = []
    diagonal_residuals = []

    # Initialize: compute diagonal elements
    print("Computing initial diagonal elements...")
    for i in range(n):
        ei = _create_unit_vector(binding, x_sample, i)
        diag_elem = _extract_diagonal_element(linop, ei, i)
        diagonal_residuals.append(float(diag_elem))

    # Convert to tensor for easier manipulation
    if binding == 'torch':
        import torch
        diag_residuals = torch.tensor(diagonal_residuals, dtype=x_sample.dtype, device=x_sample.device)

    # Greedy selection loop
    for k in range(min(target_rank, max_rank, n)):
        # Find the pivot with largest residual diagonal
        if binding == 'torch':
            pivot_idx = torch.argmax(diag_residuals).item()
            max_residual = diag_residuals[pivot_idx].item()
        else:
            pivot_idx = diagonal_residuals.index(max(diagonal_residuals))
            max_residual = diagonal_residuals[pivot_idx]

        # Check stopping criterion
        if max_residual < tolerance:
            print(f"Stopping pivot selection: max residual {max_residual:.2e} < tolerance {tolerance:.2e}")
            break

        selected_pivots.append(pivot_idx)
        print(f"Selected pivot {k + 1}: index {pivot_idx}, residual: {max_residual:.2e}")

        # Update residual diagonal elements
        if k < min(target_rank, max_rank, n) - 1:  # Don't update on last iteration
            _update_residual_diagonal(binding, linop, x_sample, pivot_idx,
                                      selected_pivots, diag_residuals if binding == 'torch' else diagonal_residuals)

    return selected_pivots


def _random_pivot_selection(n, target_rank):
    """
    Random pivot selection (for comparison/fallback).
    """
    import random
    indices = list(range(n))
    random.shuffle(indices)
    return indices[:target_rank]


def _diagonal_pivot_selection(binding, linop, x_sample, target_rank):
    """
    Select pivots based on largest diagonal elements.
    """
    tools = get_tools(binding)
    n = tools.size(x_sample)

    diagonal_elements = []
    for i in range(n):
        ei = _create_unit_vector(binding, x_sample, i)
        diag_elem = _extract_diagonal_element(linop, ei, i)
        diagonal_elements.append((float(diag_elem), i))

    # Sort by diagonal value (descending)
    diagonal_elements.sort(reverse=True)

    return [idx for _, idx in diagonal_elements[:target_rank]]


def _create_unit_vector(binding, x_sample, index):
    """
    Create the i-th unit vector with same properties as x_sample.
    """
    tools = get_tools(binding)

    if binding == 'torch':
        import torch
        ei = torch.zeros_like(x_sample)
        if len(x_sample.shape) == 1:
            ei[index] = 1.0
        else:
            flat_ei = ei.view(-1)
            flat_ei[index] = 1.0
            ei = flat_ei.view(x_sample.shape)
    else:
        ei = tools.zeros_like(x_sample)
        if len(ei.shape) == 1:
            ei[index] = 1.0
        else:
            flat_ei = ei.view(-1)
            flat_ei[index] = 1.0

    return ei


def _extract_diagonal_element(linop, unit_vector, index):
    """
    Extract diagonal element by applying linop to unit vector.
    """
    result = linop(unit_vector)

    if len(result.shape) == 1:
        return result[index]
    else:
        flat_result = result.view(-1)
        return flat_result[index]


def _update_residual_diagonal(binding, linop, x_sample, new_pivot, selected_pivots, diag_residuals):
    """
    Update residual diagonal after adding a new pivot (for greedy selection).
    
    This implements the pivoted Cholesky update:
    diag_residual[i] -= (K[i, new_pivot])^2 / K[new_pivot, new_pivot]
    """
    # Get the new pivot column
    pivot_vector = _create_unit_vector(binding, x_sample, new_pivot)
    pivot_column = linop(pivot_vector)

    # Get diagonal element of new pivot
    pivot_diag = _extract_diagonal_element(linop, pivot_vector, new_pivot)

    if abs(pivot_diag) < 1e-12:
        print(f"Warning: very small pivot diagonal {pivot_diag}")
        return

    # Update residual diagonal
    tools = get_tools(binding)
    n = tools.size(x_sample)

    for i in range(n):
        if i not in selected_pivots:  # Don't update already selected pivots
            # Extract K[i, new_pivot]
            if len(pivot_column.shape) == 1:
                kij = pivot_column[i]
            else:
                flat_column = pivot_column.view(-1)
                kij = flat_column[i]

            # Update residual: diag[i] -= kij^2 / pivot_diag
            update = (kij * kij) / pivot_diag

            if binding == 'torch':
                diag_residuals[i] -= update
                # Ensure non-negative
                diag_residuals[i] = torch.maximum(diag_residuals[i], torch.tensor(0.0, device=diag_residuals.device))
            else:
                diag_residuals[i] -= float(update)
                diag_residuals[i] = max(diag_residuals[i], 0.0)


def _construct_nystrom_factor(binding, linop, x_sample, pivot_indices):
    """
    Construct the Nyström factor C such that K ≈ C * C^T.
    
    C[i,j] = K[i, pivot_j] / sqrt(K[pivot_j, pivot_j])
    """
    tools = get_tools(binding)
    n = tools.size(x_sample)
    rank = len(pivot_indices)

    print(f"Constructing Nyström factor: {n} × {rank}")

    # Initialize factor matrix
    if binding == 'torch':
        import torch
        C = torch.zeros(n, rank, dtype=x_sample.dtype, device=x_sample.device)
    else:
        # For other backends, we'll build column by column
        C_columns = []

    # Construct each column of C
    for j, pivot_idx in enumerate(pivot_indices):
        # Get the pivot column from kernel matrix
        pivot_vector = _create_unit_vector(binding, x_sample, pivot_idx)
        kernel_column = linop(pivot_vector)

        # Get diagonal element for normalization
        pivot_diag = _extract_diagonal_element(linop, pivot_vector, pivot_idx)

        if abs(pivot_diag) < 1e-12:
            print(f"Warning: very small pivot diagonal {pivot_diag} at pivot {j}")
            pivot_diag = 1e-12  # Regularize

        # Normalize: C[:, j] = K[:, pivot_idx] / sqrt(K[pivot_idx, pivot_idx])
        normalizer = (abs(pivot_diag)) ** 0.5

        if binding == 'torch':
            if len(kernel_column.shape) > 1:
                kernel_column = kernel_column.view(-1)
            C[:, j] = kernel_column / normalizer
        else:
            if len(kernel_column.shape) > 1:
                kernel_column = kernel_column.view(-1)
            normalized_column = kernel_column / normalizer
            C_columns.append(normalized_column)

        if (j + 1) % 10 == 0:
            print(f"  Constructed {j + 1}/{rank} columns")

    if binding != 'torch':
        # Stack columns for other backends
        C = tools.stack(C_columns, axis=1) if hasattr(tools, 'stack') else C_columns

    return C


def _create_nystrom_preconditioner(binding, C, regularization=1e-6):
    """
    Create preconditioner from Nyström factor.
    
    The preconditioner solves: (C * C^T + λI) * x = b
    using the Woodbury matrix identity.
    """
    tools = get_tools(binding)

    if binding == 'torch':
        import torch

        # Precompute: (C^T * C + λI)^(-1)
        CtC = torch.matmul(C.T, C)
        regularized_CtC = CtC + regularization * torch.eye(CtC.shape[0],
                                                           dtype=C.dtype, device=C.device)

        try:
            CtC_inv = torch.linalg.inv(regularized_CtC)
            print(f"Nyström preconditioner ready with rank {C.shape[1]}")
        except Exception as e:
            print(f"Matrix inversion failed: {e}, using pseudo-inverse")
            CtC_inv = torch.linalg.pinv(regularized_CtC)

        def nystrom_preconditioner(x):
            """
            Apply Nyström preconditioner using Woodbury identity:
            (C*C^T + λI)^(-1) * x = (1/λ) * (x - C * (C^T*C + λI)^(-1) * C^T * x)
            """
            if len(x.shape) > 1:
                x_flat = x.view(-1)
            else:
                x_flat = x

            # Woodbury formula application
            Ctx = torch.matmul(C.T, x_flat)
            temp = torch.matmul(CtC_inv, Ctx)
            Ctemp = torch.matmul(C, temp)

            result = (x_flat - Ctemp) / regularization

            return result.view(x.shape) if len(x.shape) > 1 else result

    else:
        # For other backends, create a simpler version
        print("Using simplified Nyström preconditioner for non-torch backend")

        def nystrom_preconditioner(x):
            # Simple scaling approximation
            return x / regularization

    return nystrom_preconditioner


def _create_identity_preconditioner():
    """Create a trivial identity preconditioner."""

    def identity_preconditioner(x):
        return x

    return identity_preconditioner
