from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends

tensor_types = BackendTensor.tensor_types


def _upgrade_kernel_input_to_keops_tensor(struct_data_instance):
    from pykeops.numpy import LazyTensor

    for key, val in struct_data_instance.__dict__.items():
        struct_data_instance.__dict__[key] = LazyTensor(val.astype('float32'))


@dataclass
class OrientationSurfacePointsCoords:
    dip_ref_i: tensor_types = np.empty((0, 1, 3))
    dip_ref_j: tensor_types = np.empty((1, 0, 3))
    diprest_i: tensor_types = np.empty((0, 1, 3))
    diprest_j: tensor_types = np.empty((1, 0, 3))

    def __init__(self, x_ref: np.ndarray, y_ref: np.ndarray, x_rest: np.ndarray, y_rest: np.ndarray):
        def _assembly(x, y) -> Tuple[np.ndarray, np.ndarray]:
            dips_points0 = x[:, None, :]  # i
            dips_points1 = y[None, :, :]  # j
            return dips_points0, dips_points1

        self.dip_ref_i, self.dip_ref_j = _assembly(x_ref, y_ref)
        self.diprest_i, self.diprest_j = _assembly(x_rest, y_rest)

        if BackendTensor.engine_backend == AvailableBackends.numpy and BackendTensor.pykeops_enabled:
            _upgrade_kernel_input_to_keops_tensor(self)


@dataclass
class OrientationsDrift:
    dips_ug_ai: tensor_types = np.empty((0, 1, 3))
    dips_ug_aj: tensor_types = np.empty((1, 0, 3))
    dips_ug_bi: tensor_types = np.empty((0, 1, 3))
    dips_ug_bj: tensor_types = np.empty((1, 0, 3))
    dips_ug_ci: tensor_types = np.empty((0, 1, 3))
    dips_ug_cj: tensor_types = np.empty((1, 0, 3))
    selector_ci: tensor_types = np.empty((0, 1, 3))
    selector_cj: tensor_types = np.empty((1, 0, 3))

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

        if BackendTensor.engine_backend == AvailableBackends.numpy and BackendTensor.pykeops_enabled:
            _upgrade_kernel_input_to_keops_tensor(self)


@dataclass
class PointsDrift:
    dipsPoints_ui_ai: tensor_types = np.empty((0, 1, 3))
    dipsPoints_ui_aj: tensor_types = np.empty((1, 0, 3))
    dipsPoints_ui_bi1: tensor_types = np.empty((0, 1, 3))
    dipsPoints_ui_bj1: tensor_types = np.empty((1, 0, 3))
    dipsPoints_ui_bi2: tensor_types = np.empty((0, 1, 3))
    dipsPoints_ui_bj2: tensor_types = np.empty((1, 0, 3))

    def __init__(self, x_degree_1: np.ndarray, y_degree_1: np.ndarray, x_degree_2a: np.ndarray,
                 y_degree_2a: np.ndarray, x_degree_2b: np.ndarray, y_degree_2b: np.ndarray):
        self.dipsPoints_ui_ai = x_degree_1[:, None, :]
        self.dipsPoints_ui_aj = y_degree_1[None, :, :]
        self.dipsPoints_ui_bi1 = x_degree_2a[:, None, :]
        self.dipsPoints_ui_bj1 = y_degree_2a[None, :, :]
        self.dipsPoints_ui_bi2 = x_degree_2b[:, None, :]
        self.dipsPoints_ui_bj2 = y_degree_2b[None, :, :]
        if BackendTensor.engine_backend == AvailableBackends.numpy and BackendTensor.pykeops_enabled:
            _upgrade_kernel_input_to_keops_tensor(self)


@dataclass
class FaultDrift:
    faults_i: tensor_types = np.empty((0, 1, 3))
    faults_j: tensor_types = np.empty((1, 0, 3))

    def __init__(self, x_degree_1: np.ndarray, y_degree_1: np.ndarray, ):
        self.faults_i = x_degree_1[:, None, :]
        self.faults_j = y_degree_1[None, :, :]

        if BackendTensor.engine_backend == AvailableBackends.numpy and BackendTensor.pykeops_enabled:
            _upgrade_kernel_input_to_keops_tensor(self)


@dataclass
class CartesianSelector:
    hu_sel_i: tensor_types = np.empty((0, 1, 3))
    hu_sel_j: tensor_types = np.empty((1, 0, 3))
    hv_sel_i: tensor_types = np.empty((0, 1, 3))
    hv_sel_j: tensor_types = np.empty((1, 0, 3))

    h_sel_ref_i: tensor_types = np.empty((0, 1, 3))
    h_sel_ref_j: tensor_types = np.empty((1, 0, 3))

    h_sel_rest_i: tensor_types = np.empty((0, 1, 3))
    h_sel_rest_j: tensor_types = np.empty((1, 0, 3))

    is_gradient: bool = False

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

        self.is_gradient = is_gradient

        if BackendTensor.engine_backend == AvailableBackends.numpy and BackendTensor.pykeops_enabled:
            _upgrade_kernel_input_to_keops_tensor(self)


@dataclass
class DriftMatrixSelector:
    sel_ui: tensor_types = np.empty((0, 1, 3))
    sel_vj: tensor_types = np.empty((1, 0, 3))
    
    def __init__(self, x_size: int, y_size: int, n_drift_eq: int, drift_start_post_x: int, drift_start_post_y: int):
        sel_i = np.zeros((x_size, 2))
        sel_j = np.zeros((y_size, 2))

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

        if BackendTensor.engine_backend == AvailableBackends.numpy and BackendTensor.pykeops_enabled:
            _upgrade_kernel_input_to_keops_tensor(self)
    
    
    
    @classmethod
    def old_method(cls, x_size: int, y_size: int, n_drift_eq: int):  # * This does not account for faults
        sel_i = np.zeros((x_size, 2))
        sel_j = np.zeros((y_size, 2))

        # ! TODO [x]: This need to account for faults too. This is what the new __init__ does.

        sel_i[:-n_drift_eq, 0] = 1
        sel_i[-n_drift_eq:, 1] = 1

        sel_j[:-n_drift_eq, 0] = -1
        sel_j[-n_drift_eq:, 1] = -1

        foo = cls(1, 1, 1, 1)
        foo.sel_ui = sel_i[:, None, :]
        foo.sel_vj = sel_j[None, :, :]

        if BackendTensor.engine_backend == AvailableBackends.numpy and BackendTensor.pykeops_enabled:
            _upgrade_kernel_input_to_keops_tensor(cls)
        return foo


@dataclass
class KernelInput:
    # Used for CG, CI and CGI
    ori_sp_matrices: OrientationSurfacePointsCoords
    cartesian_selector: CartesianSelector

    # Used for Drift
    ori_drift: OrientationsDrift
    ref_drift: PointsDrift
    rest_drift: PointsDrift
    drift_matrix_selector: DriftMatrixSelector

    ref_fault: Optional[FaultDrift]
    rest_fault: Optional[FaultDrift]
