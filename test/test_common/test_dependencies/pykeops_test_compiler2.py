# %%
import numpy as np


from pykeops.numpy import LazyTensor
import pykeops

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import SurfacePoints, Orientations, InterpolationOptions, TensorsStructure
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, orientations_preprocess
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance, yield_b_vector
from gempy_engine.modules.solver.solver_interface import kernel_reduction

#pykeops.clean_pykeops()
#pykeops.test_numpy_bindings()
#pykeops.verbose = True
#pykeops.config.build_type = 'Debug'
print(pykeops.config.gpu_available)


# =======
dip_positions = np.array([
    [0.25010, 0.50010, 0.54177],
    [0.66677, 0.50010, 0.62510],
], dtype=BackendTensor.default_dtype)
sp = np.array([
    [0.25010, 0.50010, 0.37510],
    [0.50010, 0.50010, 0.37510],
    [0.66677, 0.50010, 0.41677],
    [0.70843, 0.50010, 0.47510],
    [0.75010, 0.50010, 0.54177],
    [0.58343, 0.50010, 0.39177],
    [0.73343, 0.50010, 0.50010],
], dtype=BackendTensor.default_dtype)

nugget_effect_scalar = 0
spi = SurfacePoints(sp, nugget_effect_scalar)

dip_gradients = np.array([[0, 0, 1],
                          [-.6, 0, .8]], dtype=BackendTensor.default_dtype)
nugget_effect_grad = 0

range_ = 4.166666666667
co = 0.1428571429

ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)

kri = InterpolationOptions(range_, co, 0,
                           number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)

tensor_structure = TensorsStructure(np.array([7]))

surface_points = spi
orientations = ori_i
options = kri
tensors_structure = tensor_structure

# options.i_res = 1
# options.gi_res = 1

sp_internals = surface_points_preprocess(surface_points,
                                         tensors_structure.number_of_points_per_surface)
ori_internals = orientations_preprocess(orientations)


options.kernel_function = AvailableKernelFunctions.cubic
# options.range = 4.464646446464646464
# options.i_res = 4
# options.gi_res =2

# Test cov
cov = yield_covariance(SolverInput(sp_internals, ori_internals, options))
print("\n")
print(cov)

# Test weights and b vector
b_vec = yield_b_vector(ori_internals, cov.shape[0])
weights = kernel_reduction(cov, b_vec, compute=True)
print(weights)