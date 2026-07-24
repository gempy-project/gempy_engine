from ...core.backend_tensor import BackendTensor
from ...core.data.internal_structs import SolverInput
from ...core.data.kernel_classes.orientations import OrientationsInternals

from ._b_vector_assembler import b_vector_assembly
from ._kernels_assembler import create_cov_kernel, create_scalar_kernel, create_grad_kernel
from ._internalDistancesMatrices import DistancesBuffer
from ._vectors_preparation import cov_vectors_preparation, evaluation_vectors_preparations
from .execution_mode import KernelExecutionMode
from ...core.data.options import KernelOptions

tensor_types = BackendTensor.tensor_types


def yield_covariance(
        interp_input: SolverInput,
        kernel_options: KernelOptions,
        execution_mode: KernelExecutionMode = KernelExecutionMode.DENSE,
) -> tensor_types:
    kernel_data = cov_vectors_preparation(interp_input, kernel_options)
    cov = create_cov_kernel(kernel_data, kernel_options, execution_mode=execution_mode)
    return cov


def yield_b_vector(ori_internals: OrientationsInternals, cov_size: int) -> tensor_types:
    return b_vector_assembly(ori_internals, cov_size)


def yield_evaluation_kernel(
        interp_input: SolverInput,
        kernel_options: KernelOptions,
        slice_array=None,
        pykeops: bool = False,
        distance_buffer: DistancesBuffer | None = None,
):
    distance_cache_key = _distance_cache_key(interp_input, slice_array)
    
    kernel_data = evaluation_vectors_preparations(interp_input, kernel_options, axis=None, slice_array=slice_array)
    if pykeops: kernel_data = kernel_data.upgrade_tensors()
    execution_mode = KernelExecutionMode.SYMBOLIC if pykeops else KernelExecutionMode.DENSE
    return create_scalar_kernel(
        kernel_data,
        kernel_options,
        execution_mode=execution_mode,
        distance_buffer=distance_buffer,
        distance_cache_key=distance_cache_key,
    )


def yield_evaluation_grad_kernel(
        interp_input: SolverInput,
        kernel_options: KernelOptions,
        axis: int = 0,
        slice_array=None,
        pykeops: bool = False,
        distance_buffer: DistancesBuffer | None = None,
):
    distance_cache_key = _distance_cache_key(interp_input, slice_array)
    kernel_data = evaluation_vectors_preparations(interp_input, kernel_options, axis, slice_array)
    if pykeops: kernel_data = kernel_data.upgrade_tensors()
    execution_mode = KernelExecutionMode.SYMBOLIC if pykeops else KernelExecutionMode.DENSE
    return create_grad_kernel(
        kernel_data,
        kernel_options,
        execution_mode=execution_mode,
        distance_buffer=distance_buffer,
        distance_cache_key=distance_cache_key,
    )


def _distance_cache_key(interp_input: SolverInput, slice_array) -> tuple[int, tuple[int | None, int | None, int | None] | None]:
    slice_key = (
        (slice_array.start, slice_array.stop, slice_array.step)
        if isinstance(slice_array, slice)
        else None
    )
    return id(interp_input), slice_key
