from dataclasses import dataclass
from enum import Enum

from typing import Callable

from gempy_engine.core.backend_tensor import BackendTensor

import numpy

dtype = BackendTensor.dtype

from dataclasses import dataclass
from enum import Enum
from typing import Callable
import torch

from gempy_engine.core.backend_tensor import BackendTensor

# We define JIT-compiled versions for GPU/PyTorch performance.
# These fuse all element-wise operations into a single kernel execution.

@torch.jit.script
def cubic_function(r: torch.Tensor, a: float) -> torch.Tensor:
    # Horner's method for stability and fewer ops:
    # 1 - 7x^2 + 35/4 x^3 - 7/2 x^5 + 3/4 x^7
    # where x = r/a

    # Pre-calculate constants
    c2 = -7.0
    c3 = 8.75      # 35/4
    c5 = -3.5      # 7/2
    c7 = 0.75      # 3/4

    x = r / a
    x2 = x * x
    # Factor out x^2 to reduce powers: 1 + x^2 * (-7 + x * (8.75 + x^2 * (-3.5 + 0.75 * x^2)))
    # But standard Horner on the polynomial in x is likely best or just explicit fused math
    # 1 + x^2 * (-7 + x * (35/4 + x^2 * (-7/2 + x^2 * 3/4)))

    return 1.0 + x2 * (c2 + x * (c3 + x2 * (c5 + x2 * c7)))


@torch.jit.script
def cubic_function_p_div_r(r: torch.Tensor, a: float) -> torch.Tensor:
    # (-14 / a^2) + 105 r / (4 a^3) - 35 r^3 / (2 a^5) + 21 r^5 / (4 a^7)
    a_inv = 1.0 / a
    a2_inv = a_inv * a_inv
    x = r * a_inv
    x2 = x * x

    t0 = -14.0 * a2_inv
    t1 = 26.25 * a2_inv * a_inv # 105/4 / a^3 -> 26.25 * a^-3 = 26.25 * (r/a) / r / a^2 ... logic check
    # Let's stick to the structure: 
    # term1 = -14/a^2
    # term2 = 26.25 * r / a^3
    # term3 = -17.5 * r^3 / a^5
    # term4 = 5.25 * r^5 / a^7

    # Optimized:
    # a^-2 * ( -14 + x * (26.25 + x^2 * (-17.5 + 5.25 * x^2)))
    return a2_inv * (-14.0 + x * (26.25 + x2 * (-17.5 + 5.25 * x2)))


@torch.jit.script
def cubic_function_a(r: torch.Tensor, a: float) -> torch.Tensor:
    # This one is complex, simpler to let JIT fuse the raw expression than optimize manually and risk bugs
    # 7 * (9 * r^5 - 20 * a^2 * r^3 + 15 * a^4 * r - 4 * a^5) / (2 * a^7)

    # However, ensuring float literals helps JIT
    return 7.0 * (9.0 * r ** 5 - 20.0 * (a ** 2) * (r ** 3) + 15.0 * (a ** 4) * r - 4.0 * (a ** 5)) / (2.0 * (a ** 7))


@torch.jit.script
def exp_function(sq_r: torch.Tensor, a: float) -> torch.Tensor:
    # exp(-(r^2 / (2 a^2)))
    return torch.exp(-(sq_r / (2.0 * a * a)))


@torch.jit.script
def exp_function_p_div_r(sq_r: torch.Tensor, a: float) -> torch.Tensor:
    # -(1 / a^2) * exp(...)
    val = torch.exp(-(sq_r / (2.0 * a * a)))
    return -(1.0 / (a * a)) * val


@torch.jit.script
def exp_function_a(sq_r: torch.Tensor, a: float) -> torch.Tensor:
    # (sq_r / a^4 - 1/a^2) * exp(...)
    a2 = a * a
    a4 = a2 * a2
    term1 = sq_r / a4
    term2 = 1.0 / a2
    term3 = torch.exp(-(sq_r / (2.0 * a2)))
    return (term1 - term2) * term3


square_root_3 = 1.73205080757
sqrt5 = 2.2360679775

@torch.jit.script
def matern_function_5_2(r: torch.Tensor, a: float) -> torch.Tensor:
    # (1 + sqrt5 * r/a + 5/3 * r^2/a^2) * exp(-sqrt5 * r/a)
    # a is float.
    # Precompute constants
    s5 = 2.2360679775

    # Common term x = r/a
    x = r / a
    s5_x = s5 * x

    # Polynomial part: 1 + s5_x + (5/3) * x^2
    poly = 1.0 + s5_x + (1.6666666667 * x * x)

    return poly * torch.exp(-s5_x)


@torch.jit.script
def matern_function_5_2_p_div_r(r: torch.Tensor, a: float) -> torch.Tensor:
    # -(5 * exp(...) * (a + sqrt5 * r)) / (3 * a^3)
    s5 = 2.2360679775
    x = r / a

    term_exp = torch.exp(-s5 * x)
    numerator = -5.0 * term_exp * (a + s5 * r)
    denominator = 3.0 * (a * a * a)

    return numerator / denominator


@torch.jit.script
def matern_function_5_2_a(r: torch.Tensor, a: float) -> torch.Tensor:
    s5 = 2.2360679775
    x = r / a
    term_exp = torch.exp(-s5 * x)

    # (a^2 + sqrt5 * a * r - 5 * r^2)
    poly = (a * a) + (s5 * a * r) - (5.0 * r * r)

    return -5.0 * term_exp * poly / (3.0 * (a * a * a * a))


@dataclass
class KernelFunction:
    base_function: Callable
    derivative_div_r: Callable
    second_derivative: Callable
    consume_sq_distance: bool


class AvailableKernelFunctions(Enum):
    # We plug in the JIT functions here. 
    # Note: For NumPy compatibility, GemPy usually handles backend switching elsewhere.
    # Since we are optimizing for GPU/PyTorch, providing JIT functions here is safe 
    # provided the inputs 'r' are Tensors.

    cubic = KernelFunction(cubic_function, cubic_function_p_div_r, cubic_function_a, consume_sq_distance=False)
    exponential = KernelFunction(exp_function, exp_function_p_div_r, exp_function_a, consume_sq_distance=True)
    matern_5_2 = KernelFunction(matern_function_5_2, matern_function_5_2_p_div_r, matern_function_5_2_a, consume_sq_distance=False)
