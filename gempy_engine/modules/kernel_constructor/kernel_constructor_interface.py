from ...core.backend_tensor import BackendTensor
from ...core.data.internal_structs import SolverInput
from ...core.data.kernel_classes.orientations import OrientationsInternals

from ._b_vector_assembler import b_vector_assembly
from ._kernels_assembler import create_cov_kernel, create_scalar_kernel, create_grad_kernel
from ._vectors_preparation import cov_vectors_preparation, evaluation_vectors_preparations

tensor_types = BackendTensor.tensor_types


def yield_covariance(interp_input: SolverInput) -> tensor_types:
    kernel_data = cov_vectors_preparation(interp_input)
    cov = create_cov_kernel(kernel_data, interp_input.options)
    return cov


def yield_b_vector(ori_internals: OrientationsInternals, cov_size: int) -> tensor_types:
    return b_vector_assembly(ori_internals, cov_size)


def yield_evaluation_kernel(grid: tensor_types, interp_input: SolverInput):

    kernel_data = evaluation_vectors_preparations(grid, interp_input)
    return create_scalar_kernel(kernel_data, interp_input.options)


def yield_evaluation_grad_kernel(grid: tensor_types, interp_input: SolverInput, axis:int=0):
    kernel_data = evaluation_vectors_preparations(grid, interp_input, axis=axis)
    return create_grad_kernel(kernel_data, interp_input.options)

