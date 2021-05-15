from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    values: np.ndarray
    len_grids: np.ndarray
    regular_grid_shape: np.ndarray # Shape(3)

    @property
    def len_all_grids(self):
        return self.len_grids.sum(axis=0)
