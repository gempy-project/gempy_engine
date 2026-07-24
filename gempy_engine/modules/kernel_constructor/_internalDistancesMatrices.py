from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...core.backend_tensor import BackendTensor
from gempy_engine.config import DEBUG_MODE, AvailableBackends
from .execution_mode import KernelExecutionMode


@dataclass
class InternalDistancesMatrices:
    dif_ref_ref: np.ndarray
    dif_rest_rest: np.ndarray
    hu: np.ndarray
    hv: np.ndarray
    huv_ref: np.ndarray
    huv_rest: np.ndarray
    perp_matrix: np.ndarray
    r_ref_ref: np.ndarray
    r_ref_rest: np.ndarray
    r_rest_ref: np.ndarray
    r_rest_rest: np.ndarray
    hu_ref: np.ndarray
    hu_rest: np.ndarray
    hu_ref_grad: Optional[np.ndarray]  # These are only used by grad eval
    hu_rest_grad: Optional[np.ndarray]  # These are only used by grad eval
    # hu_ref_sum: np.ndarray   # These are only used for caching
    # hu_rest_sum: np.ndarray  # These are only used for caching
    # 
    
    def __post_init__(self):
        if DEBUG_MODE and BackendTensor.engine_backend != AvailableBackends.PYTORCH:
            assert self.dif_ref_ref.dtype == BackendTensor.dtype, f"Wrong dtype for dif_ref_ref: {self.dif_ref_ref.dtype}. should be {BackendTensor.dtype}"
            assert self.dif_rest_rest.dtype == BackendTensor.dtype, f"Wrong dtype for dif_rest_rest: {self.dif_rest_rest.dtype}. should be {BackendTensor.dtype}"
            assert self.hu.dtype == BackendTensor.dtype, f"Wrong dtype for hu: {self.hu.dtype}. should be {BackendTensor.dtype}"
            assert self.hv.dtype == BackendTensor.dtype, f"Wrong dtype for hv: {self.hv.dtype}. should be {BackendTensor.dtype}"
            assert self.huv_ref.dtype == BackendTensor.dtype, f"Wrong dtype for huv_ref: {self.huv_ref.dtype}. should be {BackendTensor.dtype}"
            assert self.huv_rest.dtype == BackendTensor.dtype, f"Wrong dtype for huv_rest: {self.huv_rest.dtype}. should be {BackendTensor.dtype}"
            assert (self.perp_matrix.dtype == "int8" or self.perp_matrix.dtype == BackendTensor.dtype), f"Wrong dtype for perp_matrix: {self.perp_matrix.dtype}. should be int8 or float32 for pykeops"
            assert self.r_ref_ref.dtype == BackendTensor.dtype, f"Wrong dtype for r_ref_ref: {self.r_ref_ref.dtype}. should be {BackendTensor.dtype}"
            assert self.r_ref_rest.dtype == BackendTensor.dtype, f"Wrong dtype for r_ref_rest: {self.r_ref_rest.dtype}. should be {BackendTensor.dtype}"
            assert self.r_rest_ref.dtype == BackendTensor.dtype, f"Wrong dtype for r_rest_ref: {self.r_rest_ref.dtype}. should be {BackendTensor.dtype}"
            assert self.r_rest_rest.dtype == BackendTensor.dtype, f"Wrong dtype for r_rest_rest: {self.r_rest_rest.dtype}. should be {BackendTensor.dtype}"
            assert self.hu_ref.dtype == BackendTensor.dtype, f"Wrong dtype for hu_ref: {self.hu_ref.dtype}. should be {BackendTensor.dtype}"
            assert self.hu_rest.dtype == BackendTensor.dtype, f"Wrong dtype for hu_rest: {self.hu_rest.dtype}. should be {BackendTensor.dtype}"


@dataclass(frozen=True)
class SharedDistanceMatrices:
    dif_ref_ref: object
    dif_rest_rest: object
    hu: object
    r_ref_ref: object
    r_ref_rest: object
    r_rest_ref: object
    r_rest_rest: object
    hu_ref: object
    hu_rest: object

    @classmethod
    def from_distance_matrices(cls, matrices: InternalDistancesMatrices) -> "SharedDistanceMatrices":
        return cls(
            dif_ref_ref=matrices.dif_ref_ref,
            dif_rest_rest=matrices.dif_rest_rest,
            hu=matrices.hu,
            r_ref_ref=matrices.r_ref_ref,
            r_ref_rest=matrices.r_ref_rest,
            r_rest_ref=matrices.r_rest_ref,
            r_rest_rest=matrices.r_rest_rest,
            hu_ref=matrices.hu_ref,
            hu_rest=matrices.hu_rest,
        )


@dataclass
class DistancesBuffer:
    shared: Optional[SharedDistanceMatrices] = None
    cache_key: object = None
    square_distance: Optional[bool] = None
    execution_mode: Optional[KernelExecutionMode] = None
    matrix_shape: Optional[tuple[int, ...]] = None

    def store(
            self,
            matrices: InternalDistancesMatrices,
            cache_key: object,
            matrix_shape: tuple[int, ...],
            square_distance: bool,
            execution_mode: KernelExecutionMode,
    ) -> None:
        self.shared = SharedDistanceMatrices.from_distance_matrices(matrices)
        self.cache_key = cache_key
        self.square_distance = square_distance
        self.execution_mode = execution_mode
        self.matrix_shape = matrix_shape

    def matches(
            self,
            cache_key: object,
            matrix_shape: tuple[int, ...],
            square_distance: bool,
            execution_mode: KernelExecutionMode,
    ) -> bool:
        return (
            self.shared is not None
            and self.cache_key == cache_key
            and self.matrix_shape == matrix_shape
            and self.square_distance is square_distance
            and self.execution_mode is execution_mode
        )
            
