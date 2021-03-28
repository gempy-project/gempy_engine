# TODO: This class should be only in the module
from dataclasses import dataclass
import numpy as np

from gempy_engine.config import BackendTensor
from gempy_engine.core.data.kernel_classes.orientations import Orientations

tensor_types = BackendTensor.tensor_types


@dataclass
class SurfacePointsInternals:
    ref_surface_points: tensor_types
    rest_surface_points: tensor_types
    nugget_effect_ref_rest: tensor_types

    @property
    def n_points(self):
        return self.ref_surface_points.shape[0]


@dataclass
class OrientationsInternals:
    orientations: Orientations
    dip_positions_tiled: tensor_types

    @property
    def n_orientations_tiled(self):
        return self.dip_positions_tiled.shape[0]

    @property
    def n_orientations(self):
        return int(self.dip_positions_tiled.shape[0]/self.dip_positions_tiled.shape[1])


# TODO: Rename these classes
@dataclass
class OrientationSurfacePointsCoords:
    dip_ref_i: tensor_types = np.empty((0, 1, 3))
    dip_ref_j: tensor_types = np.empty((1, 0, 3))
    diprest_i: tensor_types = np.empty((0, 1, 3))
    diprest_j: tensor_types = np.empty((1, 0, 3))


@dataclass
class CartesianSelector:
    hu_sel_i: tensor_types = np.empty((0, 1, 3))
    hu_sel_j: tensor_types = np.empty((1, 0, 3))
    hv_sel_i: tensor_types = np.empty((0, 1, 3))
    hv_sel_j: tensor_types = np.empty((1, 0, 3))
    hu_sel_points_j: tensor_types = np.empty((1, 0, 3))
    hv_sel_points_i: tensor_types = np.empty((0, 1, 3))


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

