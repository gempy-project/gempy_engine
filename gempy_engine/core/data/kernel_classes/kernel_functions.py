from dataclasses import dataclass
from enum import Enum

from typing import Callable

from gempy_engine.core.backend_tensor import BackendTensor


def cubic_function(r, a):
    c = (1 - 7 * (r / a) ** 2 +
         35 * r ** 3 / (4 * a ** 3) -
         7 * r ** 5 / (2 * a ** 5) +
         3 * r ** 7 / (4 * a ** 7))
    return c


def cubic_function_p_div_r(r, a):
    c = ((-14 / a ** 2) +
         105 * r / (4 * a ** 3) -  # 105 / 4 * r / a ** 3 -
         35 * r ** 3 / (2 * a ** 5) +
         21 * r ** 5 / (4 * a ** 7))
    return c


def cubic_function_a(r, a):
    c = 7 * (9 * r ** 5 - 20 * a ** 2 * r ** 3 +
             15 * a ** 4 * r - 4 * a ** 5) / (2 * a ** 7)
    return c




def exp_function(r, a):
    exp_den = (2 * a ** 2)
    if BackendTensor.pykeops_enabled:
        return (-(r / exp_den)).exp()
    else:
        return BackendTensor.tfnp.exp(-(r / exp_den))


def exp_function_p_div_r(r, a):
    exp_den = (2 * a ** 2)
    if BackendTensor.pykeops_enabled:
        return -1 / exp_den  * (-(r / exp_den)).exp()
    else:
        return  -1 / exp_den *  BackendTensor.tfnp.exp(-(r / a ** 2))


def exp_function_a(r, a):
    exp_den = (2 * a ** 2)
    if BackendTensor.pykeops_enabled:
        return -1 / exp_den**2  * (-(r / exp_den)).exp()
    else:
        return  -1 / exp_den**2 *  BackendTensor.tfnp.exp(-(r / a ** 2))


@dataclass
class KernelFunction:
    base_function: Callable
    derivative_div_r: Callable
    second_derivative: Callable

    def __hash__(self):
        return hash(self.__repr__())

class AvailableKernelFunctions(Enum):
    cubic = KernelFunction(cubic_function, cubic_function_p_div_r, cubic_function_a)
    exponential = KernelFunction(exp_function_a, exp_function_p_div_r, exp_function_a)
