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
        struct_data_instance.__dict__[key] = LazyTensor(val.astype(BackendTensor.dtype))  # ! This as type is quite expensive


def _upgrade_kernel_input_to_keops_tensor_pytorch(struct_data_instance):
    from pykeops.torch import LazyTensor

    for key, val in struct_data_instance.__dict__.items():
        if key == "n_faults_i": continue
        if (val.is_contiguous() is False):
            raise ValueError("Input tensors are not contiguous")
        
        struct_data_instance.__dict__[key] = LazyTensor(val.type(BackendTensor.dtype_obj))


def _cast_tensors(data_class_instance):
    match (BackendTensor.engine_backend, BackendTensor.pykeops_enabled):
        case (AvailableBackends.numpy, True):
            _upgrade_kernel_input_to_keops_tensor_numpy(data_class_instance)
        case (AvailableBackends.PYTORCH, False):
            cast_type_inplace(data_class_instance)
        case (AvailableBackends.PYTORCH, True):
            cast_type_inplace(data_class_instance, requires_grad=False)
            _upgrade_kernel_input_to_keops_tensor_pytorch(data_class_instance)
        case (_, _):
            pass


@dataclass
class OrientationSurfacePointsCoords:
    dip_ref_i: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    dip_ref_j: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))
    diprest_i: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    diprest_j: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))

    def __init__(self, x_ref: np.ndarray, y_ref: np.ndarray, x_rest: np.ndarray, y_rest: np.ndarray):
        def _assembly(x, y) -> Tuple[np.ndarray, np.ndarray]:
            dips_points0 = x[:, None, :]  # i
            dips_points1 = y[None, :, :]  # j
            return dips_points0, dips_points1

        self.dip_ref_i, self.dip_ref_j = _assembly(x_ref, y_ref)
        self.diprest_i, self.diprest_j = _assembly(x_rest, y_rest)

        _cast_tensors(self)


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
        self.dips_ug_ai = x_degree_1[:, None, :]
        self.dips_ug_aj = y_degree_1[None, :, :]
        self.dips_ug_bi = x_degree_2[:, None, :]
        self.dips_ug_bj = y_degree_2[None, :, :]
        self.dips_ug_ci = x_degree_2b[:, None, :]
        self.dips_ug_cj = y_degree_2b[None, :, :]
        self.selector_ci = selector_degree_2[:, None, :]
        self.selector_cj = selector_degree_2[None, :, :]

        _cast_tensors(self)


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
        self.dipsPoints_ui_ai = x_degree_1[:, None, :]
        self.dipsPoints_ui_aj = y_degree_1[None, :, :]
        self.dipsPoints_ui_bi1 = x_degree_2a[:, None, :]
        self.dipsPoints_ui_bj1 = y_degree_2a[None, :, :]
        self.dipsPoints_ui_bi2 = x_degree_2b[:, None, :]
        self.dipsPoints_ui_bj2 = y_degree_2b[None, :, :]

        _cast_tensors(self)


@dataclass
class FaultDrift:
    faults_i: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    faults_j: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))

    n_faults_i: int = 0

    def __init__(self, x_degree_1: np.ndarray, y_degree_1: np.ndarray, ):
        self.faults_i = x_degree_1[:, None, :]
        self.faults_j = y_degree_1[None, :, :]

        self.n_faults_i = x_degree_1.shape[1]

        _cast_tensors(self)


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
        self.hu_sel_i = x_sel_hu[:, None, :]
        self.hu_sel_j = y_sel_hu[None, :, :]

        self.hv_sel_i = x_sel_hv[:, None, :]
        self.hv_sel_j = y_sel_hv[None, :, :]

        self.h_sel_ref_i = x_sel_h_ref[:, None, :]
        self.h_sel_ref_j = y_sel_h_ref[None, :, :]

        self.h_sel_rest_i = x_sel_h_rest[:, None, :]
        self.h_sel_rest_j = y_sel_h_rest[None, :, :]

        _cast_tensors(self)


@dataclass
class DriftMatrixSelector:
    sel_ui: tensor_types = field(default_factory=lambda: np.empty((0, 1, 3)))
    sel_vj: tensor_types = field(default_factory=lambda: np.empty((1, 0, 3)))
    
    def __init__(self, x_size: int, y_size: int, n_drift_eq: int, drift_start_post_x: int, drift_start_post_y: int):
        sel_i = np.zeros((x_size, 2), dtype=BackendTensor.dtype)
        sel_j = np.zeros((y_size, 2), dtype=BackendTensor.dtype)

        drift_pos_0_x = drift_start_post_x
        drift_pos_1_x = drift_start_post_x + n_drift_eq + 1

        drift_pos_0_y = drift_start_post_y
        drift_pos_1_y = drift_start_post_y + n_drift_eq + 1

        if n_drift_eq != 0:
            sel_i[:drift_pos_0_x, 0] = 1
            sel_i[drift_pos_0_x:drift_pos_1_x, 1] = 1

            sel_j[:drift_pos_0_y, 0] = -1
            sel_j[drift_pos_0_y:drift_pos_1_y, 1] = -1

        self.sel_ui = sel_i[:, None, :]
        self.sel_vj = sel_j[None, :, :]

        _cast_tensors(self)


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
