from dataclasses import dataclass
from enum import Enum

from typing import Callable

from gempy_engine.core.backend_tensor import BackendTensor


def cubic_function(r, a):
    return 1 - 7 * (r / a) ** 2 + 35 * r ** 3 / (4 * a ** 3) - 7 * r ** 5 / (2 * a ** 5) + 3 * r ** 7 / (4 * a ** 7)


def cubic_function_p_div_r(r, a):
    return (-14 / a ** 2) + 105 * r / (4 * a ** 3) - 35 * r ** 3 / (2 * a ** 5) + 21 * r ** 5 / (4 * a ** 7)


def cubic_function_a(r, a):
    return 7 * (9 * r ** 5 - 20 * a ** 2 * r ** 3 + 15 * a ** 4 * r - 4 * a ** 5) / (2 * a ** 7)


def exp_function(sq_r, a):
    return BackendTensor.tfnp.exp(-(sq_r / (2 * a ** 2)))


def exp_function_p_div_r(sq_r, a):
    return -(1 / (a ** 2) * BackendTensor.tfnp.exp(-(sq_r / (2 * a ** 2))))


def exp_function_a(sq_r, a):
    first_term = BackendTensor.tfnp.divide(sq_r, (a ** 4)) # ! This term is almost always zero. I thnk we can just remove it
    second_term = 1 / (a ** 2)
    third_term = BackendTensor.tfnp.exp(-(sq_r / (2 * a ** 2)))
    return (first_term - second_term) * third_term


@dataclass
class KernelFunction:
    base_function: Callable
    derivative_div_r: Callable
    second_derivative: Callable
    consume_sq_distance: bool  # * Some kernels can be expressed as a function of the squared distance


class AvailableKernelFunctions(Enum):
    cubic = KernelFunction(cubic_function, cubic_function_p_div_r, cubic_function_a, consume_sq_distance=False)
    exponential = KernelFunction(exp_function, exp_function_p_div_r, exp_function_a, consume_sq_distance=True)
