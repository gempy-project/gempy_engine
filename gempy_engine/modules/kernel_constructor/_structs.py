from dataclasses import dataclass
from typing import Tuple

import numpy as np

from gempy_engine.config import BackendTensor

tensor_types = BackendTensor.tensor_types

@dataclass
class OrientationSurfacePointsCoords:
    dip_ref_i: tensor_types = np.empty((0, 1, 3))
    dip_ref_j: tensor_types = np.empty((1, 0, 3))
    diprest_i: tensor_types = np.empty((0, 1, 3))
    diprest_j: tensor_types = np.empty((1, 0, 3))

    def __init__(self,x_ref: np.ndarray, y_ref: np.ndarray, x_rest: np.ndarray, y_rest: np.ndarray):
        def _assembly(x, y) -> Tuple[np.ndarray, np.ndarray]:
            dips_points0 = x[:, None, :]  # i
            dips_points1 = y[None, :, :]  # j
            return dips_points0, dips_points1

        self.dips__ref_i, self.dip_ref_j = _assembly(x_ref, y_ref)
        self.diprest_i, self.diprest_j = _assembly(x_rest, y_rest)


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
    hu_sel_j: tensor_types = np.empty((1, 0, 3))
    hv_sel_i: tensor_types = np.empty((0, 1, 3))
    hv_sel_j: tensor_types = np.empty((1, 0, 3))

    h_sel_ref_i: tensor_types = np.empty((0, 1, 3))
    h_sel_ref_j: tensor_types = np.empty((1, 0, 3))

    h_sel_rest_i: tensor_types = np.empty((0, 1, 3))
    h_sel_rest_j: tensor_types = np.empty((1, 0, 3))



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

