from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...core.backend_tensor import BackendTensor
from gempy_engine.config import DEBUG_MODE, AvailableBackends


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
            
