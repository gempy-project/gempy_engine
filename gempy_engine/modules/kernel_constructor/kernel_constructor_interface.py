from ...core.backend_tensor import BackendTensor
from ...core.data.internal_structs import SolverInput
from ...core.data.kernel_classes.orientations import OrientationsInternals

from ._b_vector_assembler import b_vector_assembly
from ._kernels_assembler import create_cov_kernel, create_scalar_kernel, create_grad_kernel
from ._vectors_preparation import cov_vectors_preparation, evaluation_vectors_preparations
from ...core.data.options import KernelOptions

tensor_types = BackendTensor.tensor_types


def yield_covariance(interp_input: SolverInput, kernel_options: KernelOptions) -> tensor_types:
    kernel_data = cov_vectors_preparation(interp_input, kernel_options)
    cov = create_cov_kernel(kernel_data, kernel_options)
    return cov


def yield_b_vector(ori_internals: OrientationsInternals, cov_size: int) -> tensor_types:
    return b_vector_assembly(ori_internals, cov_size)


def yield_evaluation_kernel(interp_input: SolverInput, kernel_options: KernelOptions, slice_array = None):
    
    kernel_data = evaluation_vectors_preparations(interp_input, kernel_options, axis=None, slice_array=slice_array)
    return create_scalar_kernel(kernel_data, kernel_options)


def yield_evaluation_grad_kernel(interp_input: SolverInput, kernel_options: KernelOptions, axis: int = 0, slice_array = None):
    kernel_data = evaluation_vectors_preparations(interp_input, kernel_options, axis, slice_array)
    return create_grad_kernel(kernel_data, kernel_options)
