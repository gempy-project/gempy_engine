from dataclasses import dataclass

import numpy as np


@dataclass
class OrientationsInput:
    dip_positions: np.ndarray
    dip_gradients: np.ndarray = None
    dip: np.ndarray = None
    azimuth: np.ndarray = None
    polarity: np.ndarray = None
    nugget_effect_grad: np.ndarray = 0.01


@dataclass
class SurfacePointsInput:
    sp_positions: np.ndarray
    nugget_effect_scalar: np.ndarray = 1e-6


@dataclass
class TensorsStructure:
    number_of_points_per_surface: np.int32
    len_c_g: np.int32
    len_c_gi: np.int32
    len_sp: np.int32
    len_faults: np.int32


@dataclass
class InterpolationOptions:

    range: float
    c_o: float
    uni_degree: int = 1
    i_res: float = 4.
    gi_res: float = 2.
    number_dimensions: int = 3

    @property
    def n_uni_eq(self):
        if self.uni_degree == 1:
            n = self.number_dimensions
        elif self.uni_degree == 2:
            n = self.number_dimensions * 3
        elif self.uni_degree == 0:
            n = 0
        else:
            raise AttributeError('uni_degree must be 0,1 or 2')

        return n
