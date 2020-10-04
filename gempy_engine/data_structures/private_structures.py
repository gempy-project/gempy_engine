from dataclasses import dataclass
from typing import Union

import numpy as np

from gempy_engine.data_structures.public_structures import OrientationsInput

try:
    import tensorflow as tf
    tensor_types = Union[np.ndarray, tf.Tensor, tf.Variable]
    tensor_types_or_scalar = Union[np.ndarray, tf.Tensor, tf.Variable, float]
except ImportError:
    tensor_types = np.ndarray
    tensor_types_or_scalar = Union[np.ndarray, float]


@dataclass
class OrientationsInternals:
    dip_positions_tiled: tensor_types = np.empty((0, 3))
    gx_tiled: np.array = np.empty((0, 3))
    gy_tiled: np.array = np.empty((0, 3))
    gz_tiled: np.array = np.empty((0, 3))
    ori_input: OrientationsInput = None

    @property
    def n_orientations_tiled(self):
        return self.dip_positions_tiled.shape[0]

    @property
    def n_orientations(self):
        return int(self.dip_positions_tiled.shape[0]/self.dip_positions_tiled.shape[1])


@dataclass
class SurfacePointsInternals:
    ref_surface_points: tensor_types = np.empty((0, 3))
    rest_surface_points: tensor_types = np.empty((0, 3))
    #ref_nugget: tensor_types = np.empty((0, 1))
    #rest_nugget: tensor_types = np.empty((0, 1))
    nugget_effect_ref_rest: tensor_types_or_scalar = np.empty((0, 1))

    @property
    def n_points(self):
        return self.ref_surface_points.shape[0]


@dataclass
class OrientationsGradients:
    gx: np.array = np.empty((0, 3))
    gy: np.array = np.empty((0, 3))
    gz: np.array = np.empty((0, 3))


@dataclass
class KernelInput:
    dip_ref_i: np.array = np.empty((0, 1, 3))
    dip_ref_j: np.array = np.empty((1, 0, 3))
    diprest_i: np.array = np.empty((0, 1, 3))
    diprest_j: np.array = np.empty((1, 0, 3))
    hu_sel_i: np.array = np.empty((0, 1, 3))
    hu_sel_j: np.array = np.empty((1, 0, 3))
    hv_sel_i: np.array = np.empty((0, 1, 3))
    hv_sel_j: np.array = np.empty((1, 0, 3))
    hu_sel_points_j: np.array = np.empty((1, 0, 3))
    hv_sel_points_i: np.array = np.empty((0, 1, 3))
    dips_ug_ai: np.array = np.empty((0, 1, 3))
    dips_ug_aj: np.array = np.empty((1, 0, 3))
    dips_ug_bi: np.array = np.empty((0, 1, 3))
    dips_ug_bj: np.array = np.empty((1, 0, 3))
    dipsref_ui_ai: np.array = np.empty((0, 1, 3))
    dipsref_ui_aj: np.array = np.empty((1, 0, 3))
    dipsref_ui_bi1: np.array = np.empty((0, 1, 3))
    dipsref_ui_bj1: np.array = np.empty((1, 0, 3))
    dipsref_ui_bi2: np.array = np.empty((0, 1, 3))
    dipsref_ui_bj2: np.array = np.empty((1, 0, 3))
    dipsrest_ui_ai: np.array = np.empty((0, 1, 3))
    dipsrest_ui_aj: np.array = np.empty((1, 0, 3))
    dipsrest_ui_bi1: np.array = np.empty((0, 1, 3))
    dipsrest_ui_bj1: np.array = np.empty((1, 0, 3))
    dipsrest_ui_bi2: np.array = np.empty((0, 1, 3))
    dipsrest_ui_bj2: np.array = np.empty((1, 0, 3))
    sel_ui: np.array = np.empty((0, 1, 3))
    sel_uj: np.array = np.empty((1, 0, 3))
    sel_vi: np.array = np.empty((0, 1, 3))
    sel_vj: np.array = np.empty((1, 0, 3))

@dataclass
class ExportInput:
    dips_i: np.array = np.empty((0, 1, 3))
    grid_j: np.array = np.empty((0, 1, 3))
    sel_i: np.array = np.empty((0, 1, 3))
    ref_i: np.array = np.empty((0, 1, 3))
    rest_i: np.array = np.empty((0, 1, 3))

# @dataclass
# class InterpolationOptions:
#     number_dimensions: int = 3

