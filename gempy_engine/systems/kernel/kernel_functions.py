import numpy as np


def cubic_function(r, a):
    c = (1 - 7 * (r / a) ** 2 +
         35 / 4 * (r / a) ** 3 -
         7 / 2 * (r / a) ** 5 +
         3 / 4 * (r / a) ** 7)
    return c


def cubic_function_p_div_r(r, a):
    c = ((-14 / a ** 2) + 105 / 4 * r / a ** 3 -
         35 / 2 * r ** 3 / a ** 5 +
         21 / 4 * r ** 5 / a ** 7)
    return c


def cubic_function_a(r, a):
    c = 7 * (9 * r ** 5 - 20 * a ** 2 * r ** 3 +
             15 * a ** 4 * r - 4 * a ** 5) / (2 * a ** 7)
    return c


def exp_function(r, a):
    return np.exp(-(r / a) ** 2)


def exp_function_p_div_r(r, a):
    return -2 * r / a ** 2 * np.exp(-(r / a) ** 2)


def exp_function_a(r, a):
    return (-4 * r ** 2 + 2 * a ** 2) / a ** 4 * np.exp(-(r / a) ** 2)
