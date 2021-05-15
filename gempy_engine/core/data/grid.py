from dataclasses import dataclass
from typing import Union, List

import numpy as np


@dataclass
class Grid:
    values: np.ndarray
    len_grids: Union[np.ndarray, List]
    regular_grid_shape: Union[np.ndarray, List] # Shape(3)
    dxdydz: Union[np.ndarray, List]  # Shape(3)

    @property
    def len_all_grids(self):
        return self.len_grids.sum(axis=0)

    @property
    def regular_grid(self) -> np.ndarray: # shape(nx, ny, nz, 3)
        return self.values[:self.len_grids[0]].reshape(*self.regular_grid_shape, 3)

    @property
    def regular_grid_dx(self):
        return self.dxdydz[0]

    @property
    def regular_grid_dy(self):
        return self.dxdydz[1]

    @property
    def regular_grid_dz(self):
        return self.dxdydz[2]
