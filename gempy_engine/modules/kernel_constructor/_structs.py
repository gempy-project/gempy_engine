from dataclasses import dataclass
import numpy as np

from gempy_engine.config import BackendTensor

tensor_types = BackendTensor.tensor_types

@dataclass
class OrientationSurfacePointsCoords:
    dip_ref_i: tensor_types = np.empty((0, 1, 3))
    dip_ref_j: tensor_types = np.empty((1, 0, 3))
    diprest_i: tensor_types = np.empty((0, 1, 3))
    diprest_j: tensor_types = np.empty((1, 0, 3))


@dataclass
class OrientationsDrift:
    dips_ug_ai: tensor_types = np.empty((0, 1, 3))
    dips_ug_aj: tensor_types = np.empty((1, 0, 3))
    dips_ug_bi: tensor_types = np.empty((0, 1, 3))
    dips_ug_bj: tensor_types = np.empty((1, 0, 3))

@dataclass
class PointsDrift:
    dipsPoints_ui_ai: tensor_types = np.empty((0, 1, 3))
    dipsPoints_ui_aj: tensor_types = np.empty((1, 0, 3))
    dipsPoints_ui_bi1: tensor_types = np.empty((0, 1, 3))
    dipsPoints_ui_bj1: tensor_types = np.empty((1, 0, 3))
    dipsPoints_ui_bi2: tensor_types = np.empty((0, 1, 3))
    dipsPoints_ui_bj2: tensor_types = np.empty((1, 0, 3))


@dataclass
class CartesianSelector:
    hu_sel_i: tensor_types = np.empty((0, 1, 3))
    hv_sel_i: tensor_types = np.empty((0, 1, 3))
    hv_sel_points_i: tensor_types = np.empty((0, 1, 3))

    hu_sel_j: tensor_types = np.empty((1, 0, 3))
    hv_sel_j: tensor_types = np.empty((1, 0, 3))
    hu_sel_points_j: tensor_types = np.empty((1, 0, 3))

@dataclass
class DriftMatrixSelector:
    sel_ui: tensor_types = np.empty((0, 1, 3))
    sel_uj: tensor_types = np.empty((1, 0, 3))
    sel_vi: tensor_types = np.empty((0, 1, 3))
    sel_vj: tensor_types = np.empty((1, 0, 3))


@dataclass
class KernelInput:
    ori_sp_matrices: OrientationSurfacePointsCoords
    cartesian_selector: CartesianSelector
    ori_drift: OrientationsDrift
    ref_drift: PointsDrift
    rest_drift: PointsDrift
    drift_matrix_selector: DriftMatrixSelector

