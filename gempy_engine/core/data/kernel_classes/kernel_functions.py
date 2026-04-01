from gempy_engine.core.backend_tensor import BackendTensor

dtype = BackendTensor.dtype

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
import numpy as np

# ============================================================================
# NumPy implementations (always available)
# ============================================================================
def cubic_function(r, a):
    a = float(a)
    return 1 - 7 * (r / a) ** 2 + 35 * r ** 3 / (4 * a ** 3) - 7 * r ** 5 / (2 * a ** 5) + 3 * r ** 7 / (4 * a ** 7)


def cubic_function_p_div_r(r, a):
    a = float(a)
    return (-14 / a ** 2) + 105 * r / (4 * a ** 3) - 35 * r ** 3 / (2 * a ** 5) + 21 * r ** 5 / (4 * a ** 7)


def cubic_function_a(r, a):
    a = float(a)
    return 7 * (9 * r ** 5 - 20 * a ** 2 * r ** 3 + 15 * a ** 4 * r - 4 * a ** 5) / (2 * a ** 7)


def cubic_function_numpy(r: np.ndarray, a: float) -> np.ndarray:
    c2, c3, c5, c7 = -7.0, 8.75, -3.5, 0.75
    x = r / a
    x2 = x * x
    return 1.0 + x2 * (c2 + x * (c3 + x2 * (c5 + x2 * c7)))

def cubic_function_numpy_stable(r, a):
    x = r / a
    x2 = x * x
    # Calculate terms individually to keep magnitudes small
    return 1.0 - 7.0*x2 + 8.75*x2*x - 3.5*x2*x2*x + 0.75*x2*x2*x2*x


def cubic_function_factorized(r: np.ndarray, a: float) -> np.ndarray:
    """
    Computes the kernel using the factorized form (1-x)^4 * P(x).
    This eliminates catastrophic cancellation near r = a.
    """
    x = r / a

    # We compute (1 - x) explicitly. 
    # This term handles the decay to zero safely.
    q = 1.0 - x

    # The remaining polynomial part: (1 + 4x + 3x^2 + 0.75x^3)
    # Since all coefficients here are positive, there is NO subtraction 
    # and therefore NO cancellation error.
    poly_part = 1.0 + x * (4.0 + x * (3.0 + x * 0.75))

    return (q ** 4) * poly_part

def cubic_function_p_div_r_numpy(r: np.ndarray, a: float) -> np.ndarray:
    a_inv = 1.0 / a
    a2_inv = a_inv * a_inv
    x = r * a_inv
    x2 = x * x
    return a2_inv * (-14.0 + x * (26.25 + x2 * (-17.5 + 5.25 * x2)))


def cubic_function_a_numpy(r: np.ndarray, a: float) -> np.ndarray:
    return 7.0 * (9.0 * r ** 5 - 20.0 * (a ** 2) * (r ** 3) + 15.0 * (a ** 4) * r - 4.0 * (a ** 5)) / (2.0 * (a ** 7))


def exp_function_numpy(sq_r: np.ndarray, a: float) -> np.ndarray:
    return np.exp(-(sq_r / (2.0 * a * a)))


def exp_function_p_div_r_numpy(sq_r: np.ndarray, a: float) -> np.ndarray:
    return -(1.0 / (a * a)) * np.exp(-(sq_r / (2.0 * a * a)))


def exp_function_a_numpy(sq_r: np.ndarray, a: float) -> np.ndarray:
    a2 = a * a
    a4 = a2 * a2
    return (sq_r / a4 - 1.0 / a2) * np.exp(-(sq_r / (2.0 * a2)))


def matern_function_5_2_numpy(r: np.ndarray, a: float) -> np.ndarray:
    s5 = 2.2360679775
    x = r / a
    s5_x = s5 * x
    poly = 1.0 + s5_x + (1.6666666667 * x * x)
    return poly * np.exp(-s5_x)


def matern_function_5_2_p_div_r_numpy(r: np.ndarray, a: float) -> np.ndarray:
    s5 = 2.2360679775
    x = r / a
    term_exp = np.exp(-s5 * x)
    return -5.0 * term_exp * (a + s5 * r) / (3.0 * (a * a * a))


def matern_function_5_2_a_numpy(r: np.ndarray, a: float) -> np.ndarray:
    s5 = 2.2360679775
    x = r / a
    term_exp = np.exp(-s5 * x)
    poly = (a * a) + (s5 * a * r) - (5.0 * r * r)
    return -5.0 * term_exp * poly / (3.0 * (a * a * a * a))


# ============================================================================
# KernelFunction with dual-backend support
# ============================================================================

@dataclass
class KernelFunction:
    base_function: Callable
    derivative_div_r: Callable
    second_derivative: Callable
    consume_sq_distance: bool
    _kernel_name: str = ""  # Used for torch lookup


class AvailableKernelFunctions(Enum):
    cubic = KernelFunction(
        cubic_function,
        cubic_function_p_div_r,
        cubic_function_a,
        consume_sq_distance=False, 
        _kernel_name='cubic'
    )
    exponential = KernelFunction(
        exp_function_numpy, exp_function_p_div_r_numpy, exp_function_a_numpy,
        consume_sq_distance=True, _kernel_name='exponential'
    )
    matern_5_2 = KernelFunction(
        matern_function_5_2_numpy, matern_function_5_2_p_div_r_numpy, matern_function_5_2_a_numpy,
        consume_sq_distance=False, _kernel_name='matern_5_2'
    )
