import dataclasses
from dataclasses import dataclass
from typing import Type

import numpy as np

def _cast_type_inplace(struct_data_instance):
    for key, val in struct_data_instance.__dict__.items():
        if key == "dtype": continue
        struct_data_instance.__dict__[key] = val.astype(struct_data_instance.dtype)

@dataclass
class TensorsStructure:
    number_of_points_per_surface: np.ndarray # TODO This needs to start with 0 and possibly ignore the last item
    len_grids: np.ndarray
    dtype: Type = np.int32
    _number_of_points_per_surface_vector:np.ndarray = np.ones(1)

    def __post_init__(self): # TODO: Move this to init
        _cast_type_inplace(self)
        self._number_of_points_per_surface_vector = np.concatenate(
            [np.array([0]), self.number_of_points_per_surface.cumsum()])[:-1]

    def __hash__(self):
        return hash(656)

    @property
    def len_all_grids(self):
        return self.len_grids.sum(axis=0)

    @property
    def nspv(self):
        return self._number_of_points_per_surface_vector