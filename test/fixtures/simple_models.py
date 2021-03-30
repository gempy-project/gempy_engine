import logging

import numpy as np
import pytest

from gempy_engine.config import BackendTensor
from gempy_engine.core.data.data_shape import TensorsStructure
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.options import InterpolationOptions


@pytest.fixture(scope='session')
def simple_model_2():
    print(BackendTensor.describe_conf())

    sp_coords = np.array([[4, 0],
                          [0, 0],
                          [2, 0],
                          [3, 0],
                          [3, 3],
                          [0, 2],
                          [2, 2]])

    nugget_effect_scalar = 0
    spi = SurfacePoints(sp_coords, nugget_effect_scalar)

    dip_positions = np.array([[0, 6],
                              [2, 13]])

    nugget_effect_grad = 0.0000001
    ori_i = Orientations(dip_positions, nugget_effect_grad)

    kri = InterpolationOptions(5, 5 ** 2 / 14 / 3, 0, i_res=1, gi_res=1,
                               number_dimensions=2, kernel_function = AvailableKernelFunctions.cubic)

    _ = np.ones(3)
    tensor_structure = TensorsStructure(np.array([3, 2]), _, _, _, _)

    return spi, ori_i, kri, tensor_structure