from gempy_engine.core.backend_tensor import BackendTensor

dtype = BackendTensor.dtype

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
import numpy as np

# ============================================================================
# NumPy implementations (always available)
# ============================================================================

def cubic_function_numpy(r: np.ndarray, a: float) -> np.ndarray:
    c2, c3, c5, c7 = -7.0, 8.75, -3.5, 0.75
    x = r / a
    x2 = x * x
    return 1.0 + x2 * (c2 + x * (c3 + x2 * (c5 + x2 * c7)))


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
# Torch implementations (lazy loaded)
# ============================================================================

_torch_kernels: Optional[dict] = None


def _get_torch_kernels() -> dict:
    """Lazily compile and cache torch kernel functions."""
    global _torch_kernels
    if _torch_kernels is not None:
        return _torch_kernels

    import torch

    @torch.compile(fullgraph=True, mode="default")
    def cubic_function_torch(r, a: float):
        c2, c3, c5, c7 = -7.0, 8.75, -3.5, 0.75
        x = r / a
        x2 = x * x
        return 1.0 + x2 * (c2 + x * (c3 + x2 * (c5 + x2 * c7)))

    @torch.compile(fullgraph=True, mode="default")
    def cubic_function_p_div_r_torch(r, a: float):
        a_inv = 1.0 / a
        a2_inv = a_inv * a_inv
        x = r * a_inv
        x2 = x * x
        return a2_inv * (-14.0 + x * (26.25 + x2 * (-17.5 + 5.25 * x2)))

    @torch.compile(fullgraph=True, mode="default")
    def cubic_function_a_torch(r, a: float):
        return 7.0 * (9.0 * r ** 5 - 20.0 * (a ** 2) * (r ** 3) + 15.0 * (a ** 4) * r - 4.0 * (a ** 5)) / (2.0 * (a ** 7))

    @torch.compile(fullgraph=True, mode="default")
    def exp_function_torch(sq_r, a: float):
        return torch.exp(-(sq_r / (2.0 * a * a)))

    @torch.compile(fullgraph=True, mode="default")
    def exp_function_p_div_r_torch(sq_r, a: float):
        return -(1.0 / (a * a)) * torch.exp(-(sq_r / (2.0 * a * a)))

    @torch.compile(fullgraph=True, mode="default")
    def exp_function_a_torch(sq_r, a: float):
        a2 = a * a
        a4 = a2 * a2
        return (sq_r / a4 - 1.0 / a2) * torch.exp(-(sq_r / (2.0 * a2)))

    @torch.compile(fullgraph=True, mode="default")
    def matern_function_5_2_torch(r, a: float):
        s5 = 2.2360679775
        x = r / a
        s5_x = s5 * x
        poly = 1.0 + s5_x + (1.6666666667 * x * x)
        return poly * torch.exp(-s5_x)

    @torch.compile(fullgraph=True, mode="default")
    def matern_function_5_2_p_div_r_torch(r, a: float):
        s5 = 2.2360679775
        x = r / a
        term_exp = torch.exp(-s5 * x)
        return -5.0 * term_exp * (a + s5 * r) / (3.0 * (a * a * a))

    @torch.compile(fullgraph=True, mode="default")
    def matern_function_5_2_a_torch(r, a: float):
        s5 = 2.2360679775
        x = r / a
        term_exp = torch.exp(-s5 * x)
        poly = (a * a) + (s5 * a * r) - (5.0 * r * r)
        return -5.0 * term_exp * poly / (3.0 * (a * a * a * a))

    _torch_kernels = {
            'cubic': (cubic_function_torch, cubic_function_p_div_r_torch, cubic_function_a_torch),
            'exponential': (exp_function_torch, exp_function_p_div_r_torch, exp_function_a_torch),
            'matern_5_2': (matern_function_5_2_torch, matern_function_5_2_p_div_r_torch, matern_function_5_2_a_torch),
    }
    return _torch_kernels


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

    def for_torch(self) -> "KernelFunction":
        """Return a KernelFunction using torch-compiled implementations."""
        torch_kernels = _get_torch_kernels()
        funcs = torch_kernels[self._kernel_name]
        return KernelFunction(
            base_function=funcs[0],
            derivative_div_r=funcs[1],
            second_derivative=funcs[2],
            consume_sq_distance=self.consume_sq_distance,
            _kernel_name=self._kernel_name,
        )


class AvailableKernelFunctions(Enum):
    cubic = KernelFunction(
        cubic_function_numpy, cubic_function_p_div_r_numpy, cubic_function_a_numpy,
        consume_sq_distance=False, _kernel_name='cubic'
    )
    exponential = KernelFunction(
        exp_function_numpy, exp_function_p_div_r_numpy, exp_function_a_numpy,
        consume_sq_distance=True, _kernel_name='exponential'
    )
    matern_5_2 = KernelFunction(
        matern_function_5_2_numpy, matern_function_5_2_p_div_r_numpy, matern_function_5_2_a_numpy,
        consume_sq_distance=False, _kernel_name='matern_5_2'
    )
