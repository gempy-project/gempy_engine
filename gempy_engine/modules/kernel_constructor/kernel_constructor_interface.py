from gempy_engine.core.backend_tensor import BackendTensor

from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.modules.kernel_constructor._b_vector_assembler import b_vector_assembly
from gempy_engine.modules.kernel_constructor._covariance_assembler import create_cov_kernel, create_scalar_kernel, \
    create_grad_kernel
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePointsInternals
from gempy_engine.core.data.kernel_classes.orientations import OrientationsInternals
from gempy_engine.modules.kernel_constructor._vectors_preparation import cov_vectors_preparation, \
    evaluation_vectors_preparations

tensor_types = BackendTensor.tensor_types


def yield_covariance(sp_internals: SurfacePointsInternals, ori_internals: OrientationsInternals,
                     options: InterpolationOptions) -> tensor_types:
    kernel_data = cov_vectors_preparation(sp_internals, ori_internals, options)
    cov = create_cov_kernel(kernel_data, options)
    return cov


def yield_b_vector(ori_internals: OrientationsInternals, cov_size: int) -> tensor_types:
    return b_vector_assembly(ori_internals, cov_size)


def yield_evaluation_kernel(grid: tensor_types, sp_internals: SurfacePointsInternals,
                            ori_internals: OrientationsInternals, options: InterpolationOptions):
    kernel_data = evaluation_vectors_preparations(grid, sp_internals, ori_internals, options)
    return create_scalar_kernel(kernel_data, options)

def yield_evaluation_grad_kernel(grid: tensor_types, sp_internals: SurfacePointsInternals,
                            ori_internals: OrientationsInternals, options: InterpolationOptions, axis:int=0):
    kernel_data = evaluation_vectors_preparations(grid, sp_internals, ori_internals, options, axis=axis)
    return create_grad_kernel(kernel_data, options)

