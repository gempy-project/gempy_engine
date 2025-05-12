"""
This script is to be run in the console. This will compile all the pykeops operations
necessary for the gempy interpolation and once they are compiled we can simply invoke them
from test or anywhere else.

"""

from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends
from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.options import InterpolationOptions
import numpy as np

from gempy_engine.modules.kernel_constructor._test_assembler import _test_covariance_items
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, \
    orientations_preprocess
from gempy_engine.modules.kernel_constructor._vectors_preparation import cov_vectors_preparation
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance


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

    range = 5 ** 2
    kri = InterpolationOptions.from_args(range, 1, 0, i_res=1, gi_res=1,
                               number_dimensions=2)

    _ = np.ones(3)
    tensor_structure = TensorsStructure(np.array([3, 2]))

    return spi, ori_i, kri, tensor_structure


BackendTensor.change_backend(
    engine_backend=AvailableBackends.numpy,
    use_gpu=True,
    pykeops_enabled=True
)

surface_points, orientations, options, tensors_structure = simple_model_2()

sp_internals = surface_points_preprocess(surface_points, tensors_structure)
ori_internals = orientations_preprocess(orientations)
kernel_data = cov_vectors_preparation(sp_internals, ori_internals, options)

cov = yield_covariance(sp_internals, ori_internals, options)

cov_grad = _test_covariance_items(kernel_data, options, "cov_grad")

# cov_grad = _test_covariance_items(kernel_data, options, "cov_grad")

# Compile and run cov sum axis 0
# print(cov.sum(axis=0))

# TODO: Compile and run solver
