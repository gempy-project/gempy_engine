import dataclasses
from typing import Tuple

from gempy_engine.config import BackendTensor
from gempy_engine.core.data.data_shape import TensorsStructure
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.modules.kernel_constructor._b_vector_assembler import b_vector_assembly
from gempy_engine.modules.kernel_constructor._covariance_assembler import create_covariance
from gempy_engine.modules.kernel_constructor._input_preparation import orientations_preprocess, surface_points_preprocess
from gempy_engine.modules.kernel_constructor._structs import SurfacePointsInternals, OrientationsInternals
from gempy_engine.modules.kernel_constructor._vectors_preparation import _vectors_preparation

tensor_types = BackendTensor.tensor_types

def yield_kriging_eq(surface_points: SurfacePoints, orientations: Orientations,
                     options: InterpolationOptions, data_shape: TensorsStructure)->Tuple[tensor_types, tensor_types]:

    sp_internals = surface_points_preprocess(surface_points, data_shape.number_of_points_per_surface)
    ori_internals = orientations_preprocess(orientations)

    cov = yield_covariance(sp_internals, ori_internals, options)
    b = yield_b_vector(ori_internals, cov.shape[0])
    return cov, b

def yield_covariance(sp_internals: SurfacePointsInternals, ori_internals: OrientationsInternals,
                     options: InterpolationOptions)->tensor_types:

    kernel_data = _vectors_preparation(sp_internals, ori_internals, options)
    cov = create_covariance(kernel_data, options)
    return cov

def yield_b_vector(ori_internals: OrientationsInternals, cov_size: int) -> tensor_types:
    return b_vector_assembly(ori_internals, cov_size)


