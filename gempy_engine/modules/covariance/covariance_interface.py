import dataclasses

from gempy_engine.config import BackendConf
from gempy_engine.core.data.data_shape import TensorsStructure
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.modules.covariance._covariance_assembler import create_covariance
from gempy_engine.modules.covariance._input_preparation import orientations_preprocess, surface_points_preprocess
from gempy_engine.modules.covariance._vectors_preparation import _vectors_preparation

tensor_types = BackendConf.tensor_types

def yield_covariance(surface_points: SurfacePoints, orientations: Orientations,
                     options: InterpolationOptions, data_shape: TensorsStructure)->tensor_types:
    sp_internals = surface_points_preprocess(surface_points, data_shape.number_of_points_per_surface)
    ori_internals = orientations_preprocess(orientations)

    kernel_data = _vectors_preparation(sp_internals, ori_internals, options)

    cov = create_covariance(kernel_data, options)
    return cov


