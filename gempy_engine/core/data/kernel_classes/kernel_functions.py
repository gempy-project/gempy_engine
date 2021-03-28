from dataclasses import dataclass
from enum import Enum, auto

from typing import Callable


def cubic_function(r, a):
    c = (1 - 7 * (r / a) ** 2 +
         35 * r ** 3 / (4 * a ** 3) -
         7 * r ** 5 / (2 * a ** 5) +
         3 * r ** 7 / (4 * a ** 7))
    return c


def cubic_function_p_div_r(r, a):
    c = ((-14 / a ** 2) +
         105 * r / (4 * a ** 3) - # 105 / 4 * r / a ** 3 -
         35 * r ** 3 / (2 * a ** 5) +
         21 * r ** 5 / (4 * a ** 7))
    return c


def cubic_function_a(r, a):
    c = 7 * (9 * r ** 5 - 20 * a ** 2 * r ** 3 +
             15 * a ** 4 * r - 4 * a ** 5) / (2 * a ** 7)
    return c


def exp_function(r, a):
    return (-(r / a) ** 2).exp()


def exp_function_p_div_r(r, a):
    return -2 / a ** 2 * (-(r / a) ** 2).exp()


def exp_function_a(r, a):
    return (-4 * r ** 2 + 2 * a ** 2) / a ** 4 * (-(r / a) ** 2).exp()
#

@dataclass
class KernelFunction:
    base_function: Callable
    derivative_div_r: Callable
    second_derivative: Callable



class AvailableKernelFunctions(Enum):
    cubic = KernelFunction(cubic_function, cubic_function_p_div_r, cubic_function_a)
    exponential = KernelFunction(exp_function_a, exp_function_p_div_r, exp_function_a)