from gempy_engine.config import BackendTensor

from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.modules.kernel_constructor._b_vector_assembler import b_vector_assembly
from gempy_engine.modules.kernel_constructor._covariance_assembler import create_kernel
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePointsInternals
from gempy_engine.core.data.kernel_classes.orientations import OrientationsInternals
from gempy_engine.modules.kernel_constructor._vectors_preparation import _vectors_preparation

tensor_types = BackendTensor.tensor_types


def yield_covariance(sp_internals: SurfacePointsInternals, ori_internals: OrientationsInternals,
                     options: InterpolationOptions)->tensor_types:

    kernel_data = _vectors_preparation(sp_internals, ori_internals, options)
    cov = create_kernel(kernel_data, options)
    return cov

def yield_b_vector(ori_internals: OrientationsInternals, cov_size: int) -> tensor_types:
    return b_vector_assembly(ori_internals, cov_size)

def yield_evaluation_kernel(sp_internals: SurfacePointsInternals, ori_internals: OrientationsInternals,
                     options: InterpolationOptions):
    pass
