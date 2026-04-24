import copy
from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np

from gempy_engine.core.utils import cast_type_inplace
from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends

tensor_types = BackendTensor.tensor_types


def _upgrade_kernel_input_to_keops_tensor_numpy(struct_data_instance):
    from pykeops.numpy import LazyTensor

    for key, val in struct_data_instance.__dict__.items():
        if key == "n_faults_i": continue
        if val is None: continue
        if isinstance(val, (int, float)): continue
        
        if (len(val.shape) == 2):
            struct_data_instance.__dict__[key] = LazyTensor(val.astype(BackendTensor.dtype), axis=0)
        else:
            struct_data_instance.__dict__[key] = LazyTensor(val.astype(BackendTensor.dtype))
    return struct_data_instance


def _upgrade_kernel_input_to_keops_tensor_pytorch(struct_data_instance):
    from pykeops.torch import LazyTensor
    import torch

    for key, array_like in struct_data_instance.__dict__.items():
        if key == "n_faults_i": continue
        if not isinstance(array_like, torch.Tensor):
            array_like = BackendTensor.t.array(array_like)

        if (len(array_like.shape) == 2):
            struct_data_instance.__dict__[key] = LazyTensor(array_like.type(BackendTensor.dtype_obj), axis=0)  # default to axis 0 for 2D
        else:
            struct_data_instance.__dict__[key] = LazyTensor(array_like.type(BackendTensor.dtype_obj))
    return struct_data_instance


def _cast_tensors(data_class_instance):
    cloned_instance = copy.copy(data_class_instance)
    if BackendTensor.engine_backend == AvailableBackends.numpy:
        _upgrade_kernel_input_to_keops_tensor_numpy(cloned_instance)
    else:
        _upgrade_kernel_input_to_keops_tensor_pytorch(cloned_instance)
    return cloned_instance


# --- 1. The New Helper Function (Replaces _cast_tensors logic) ---
def _secure_cast(array_like):
    return array_like


@dataclass
class OrientationSurfacePointsCoords:
    dip_ref_i: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    dip_ref_j: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))
    diprest_i: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    diprest_j: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))

    def __init__(self, x_ref: np.ndarray, y_ref: np.ndarray, x_rest: np.ndarray, y_rest: np.ndarray):
        # 1. Do the logic
        dips_points0 = x_ref[:, None, :]
        dips_points1 = y_ref[None, :, :]
        dips_points2 = x_rest[:, None, :]
        dips_points3 = y_rest[None, :, :]

        # 2. EXPLICIT CASTING (The Compiler loves this)
        # No loops, no __dict__ magic. Just straight assignments.
        self.dip_ref_i = _secure_cast(dips_points0)
        self.dip_ref_j = _secure_cast(dips_points1)
        self.diprest_i = _secure_cast(dips_points2)
        self.diprest_j = _secure_cast(dips_points3)

    def upgrade_tensors(self):
        return _cast_tensors(self)


@dataclass
class OrientationsDrift:
    dips_ug_ai: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    dips_ug_aj: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))
    dips_ug_bi: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    dips_ug_bj: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))
    dips_ug_ci: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    dips_ug_cj: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))
    selector_ci: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    selector_cj: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))

    def __init__(self,
                 x_degree_1: np.ndarray, y_degree_1: np.ndarray,
                 x_degree_2: np.ndarray, y_degree_2: np.ndarray,
                 x_degree_2b: np.ndarray, y_degree_2b: np.ndarray,
                 selector_degree_2: np.ndarray):
        self.dips_ug_ai = _secure_cast(x_degree_1[:, None, :])
        self.dips_ug_aj = _secure_cast(y_degree_1[None, :, :])
        self.dips_ug_bi = _secure_cast(x_degree_2[:, None, :])
        self.dips_ug_bj = _secure_cast(y_degree_2[None, :, :])
        self.dips_ug_ci = _secure_cast(x_degree_2b[:, None, :])
        self.dips_ug_cj = _secure_cast(y_degree_2b[None, :, :])
        self.selector_ci = _secure_cast(selector_degree_2[:, None, :])
        self.selector_cj = _secure_cast(selector_degree_2[None, :, :])

    def upgrade_tensors(self):
        return _cast_tensors(self)


@dataclass
class PointsDrift:
    dipsPoints_ui_ai: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    dipsPoints_ui_aj: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))
    dipsPoints_ui_bi1: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    dipsPoints_ui_bj1: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))
    dipsPoints_ui_bi2: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    dipsPoints_ui_bj2: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))

    def __init__(self, x_degree_1: np.ndarray, y_degree_1: np.ndarray, x_degree_2a: np.ndarray,
                 y_degree_2a: np.ndarray, x_degree_2b: np.ndarray, y_degree_2b: np.ndarray):
        self.dipsPoints_ui_ai = _secure_cast(x_degree_1[:, None, :])
        self.dipsPoints_ui_aj = _secure_cast(y_degree_1[None, :, :])
        self.dipsPoints_ui_bi1 = _secure_cast(x_degree_2a[:, None, :])
        self.dipsPoints_ui_bj1 = _secure_cast(y_degree_2a[None, :, :])
        self.dipsPoints_ui_bi2 = _secure_cast(x_degree_2b[:, None, :])
        self.dipsPoints_ui_bj2 = _secure_cast(y_degree_2b[None, :, :])

    def upgrade_tensors(self):
        return _cast_tensors(self)


@dataclass
class FaultDrift:
    faults_i: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    faults_j: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))

    n_faults_i: int = 0

    def __init__(self, x_degree_1: np.ndarray, y_degree_1: np.ndarray, ):
        self.faults_i = _secure_cast(x_degree_1[:, None, :])
        self.faults_j = _secure_cast(y_degree_1[None, :, :])

        self.n_faults_i = x_degree_1.shape[1]

    def upgrade_tensors(self):
        return _cast_tensors(self)


@dataclass
class CartesianSelector:
    hu_sel_i: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    hu_sel_j: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))
    hv_sel_i: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    hv_sel_j: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))

    h_sel_ref_i: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    h_sel_ref_j: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))

    h_sel_rest_i: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    h_sel_rest_j: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))

    # is_gradient: bool = False (June) This seems to be unused
    def __init__(self,
                 x_sel_hu, y_sel_hu,
                 x_sel_hv, y_sel_hv,
                 x_sel_h_ref, y_sel_h_ref,
                 x_sel_h_rest, y_sel_h_rest,
                 is_gradient=False):
        # Explicit Casts for all 8 fields
        self.hu_sel_i = _secure_cast(x_sel_hu[:, None, :])
        self.hu_sel_j = _secure_cast(y_sel_hu[None, :, :])

        self.hv_sel_i = _secure_cast(x_sel_hv[:, None, :])
        self.hv_sel_j = _secure_cast(y_sel_hv[None, :, :])

        self.h_sel_ref_i = _secure_cast(x_sel_h_ref[:, None, :])
        self.h_sel_ref_j = _secure_cast(y_sel_h_ref[None, :, :])

        self.h_sel_rest_i = _secure_cast(x_sel_h_rest[:, None, :])
        self.h_sel_rest_j = _secure_cast(y_sel_h_rest[None, :, :])

    def upgrade_tensors(self):
        return _cast_tensors(self)


@dataclass
class DriftMatrixSelector:
    sel_ui: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    sel_vj: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))

    def __init__(self, x_size: int, y_size: int, n_drift_eq: int, drift_start_post_x: int, drift_start_post_y: int):
        # Logic remains the same
        # sel_i = np.zeros((x_size, 2), dtype=BackendTensor.dtype)
        # sel_j = np.zeros((y_size, 2), dtype=BackendTensor.dtype)

        sel_i = BackendTensor.t.zeros((x_size, 2), dtype=BackendTensor.dtype)
        sel_j = BackendTensor.t.zeros((y_size, 2), dtype=BackendTensor.dtype)

        drift_pos_0_x = drift_start_post_x
        drift_pos_1_x = drift_start_post_x + n_drift_eq + 1
        drift_pos_0_y = drift_start_post_y
        drift_pos_1_y = drift_start_post_y + n_drift_eq + 1

        if n_drift_eq != 0:
            sel_i[:drift_pos_0_x, 0] = 1
            sel_i[drift_pos_0_x:drift_pos_1_x, 1] = 1

            sel_j[:drift_pos_0_y, 0] = -1
            sel_j[drift_pos_0_y:drift_pos_1_y, 1] = -1

        # Explicit Cast
        self.sel_ui = _secure_cast(sel_i[:, None, :])
        self.sel_vj = _secure_cast(sel_j[None, :, :])

    def upgrade_tensors(self):
        return _cast_tensors(self)


@dataclass
class KernelInput:
    # Used for CG, CI and CGI
    ori_sp_matrices: OrientationSurfacePointsCoords
    cartesian_selector: CartesianSelector
    nugget_scalar: Optional[float]  # TODO This has to be have the ref rest treatment to be able to activate one per point
    nugget_grad: Optional[float]  # * They are optional because they are not used in evaluation

    # Used for Drift
    ori_drift: OrientationsDrift
    ref_drift: PointsDrift
    rest_drift: PointsDrift
    drift_matrix_selector: DriftMatrixSelector

    ref_fault: Optional[FaultDrift]
    rest_fault: Optional[FaultDrift]

    def upgrade_tensors(self):
        return KernelInput(
            ori_sp_matrices=self.ori_sp_matrices.upgrade_tensors(),
            cartesian_selector=self.cartesian_selector.upgrade_tensors(),
            nugget_scalar=self.nugget_scalar,
            nugget_grad=self.nugget_grad,
            ori_drift=self.ori_drift.upgrade_tensors(),
            ref_drift=self.ref_drift.upgrade_tensors(),
            rest_drift=self.rest_drift.upgrade_tensors(),
            drift_matrix_selector=self.drift_matrix_selector.upgrade_tensors(),
            ref_fault=self.ref_fault.upgrade_tensors() if self.ref_fault is not None else None,
            rest_fault=self.rest_fault.upgrade_tensors() if self.rest_fault is not None else None,
        )

    def __repr__(self):
        return f"KernelInput(ori_sp_matrices={self.ori_sp_matrices}, cartesian_selector={self.cartesian_selector}, nugget_scalar={self.nugget_scalar}, nugget_grad={self.nugget_grad}, ori_drift={self.ori_drift}, ref_drift={self.ref_drift}, rest_drift={self.rest_drift}, drift_matrix_selector={self.drift_matrix_selector}, ref_fault={self.ref_fault})"
