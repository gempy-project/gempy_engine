from dataclasses import dataclass
from typing import Union, List, Dict

import numpy as np
from numpy import ndarray

from .regular_grid import RegularGrid


def _check_and_convert_list_to_array(field):
    if type(field) == list:
        field = np.array(field)
    return field


# TODO: [ ] values is independent field to regular grid. Proabably we want to have an extra field for them
# TODO: (custom_grid?) and then having a values as a property that brings both together?
@dataclass
class Grid:
    values: np.ndarray
    len_grids: Union[np.ndarray, List] = None  # TODO: This should be a bit more automatic?
    regular_grid: RegularGrid = None
    custom_grid: Dict[str, np.ndarray] = None

    debug_vals = None

    def __post_init__(self):
        if self.len_grids is None:
            self.len_grids = [self.values.shape[0]]

        self.len_grids = _check_and_convert_list_to_array(self.len_grids)

    @classmethod
    def from_regular_grid(cls, regular_grid: RegularGrid) -> "Grid":
        return cls(regular_grid.values, regular_grid=regular_grid)

    @property
    def len_all_grids(self) -> int:
        return self.len_grids.sum(axis=0)

    @property
    def regular_grid_values(self) -> np.ndarray:  # shape(nx, ny, nz, 3)
        if self.len_grids[0] != self.regular_grid_shape.prod():
            raise ValueError("The values and the regular grid do not match.")
        return self.values[:self.len_grids[0]].reshape(*self.regular_grid_shape, 3)

    @property
    def custom_grid_values(self) -> np.ndarray:
        return self.values[self.len_grids[0]:self.len_grids[1]]

    @property
    def regular_grid_shape(self) -> ndarray | list:
        return self.regular_grid.regular_grid_shape

    @property
    def dxdydz(self) -> tuple[float, float, float]:
        return self.regular_grid.dxdydz
