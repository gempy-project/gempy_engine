import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np

from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.core.data.exported_structs import MaskMatrices
from gempy_engine.core.data.grid import Grid


@dataclass
class ScalarFieldOutput:
    weights: np.ndarray
    grid: Grid

    exported_fields: ExportedFields
    mask_components: Optional[MaskMatrices]
    
    values_block: Optional[np.ndarray]  # final values ignoring unconformities
    _values_block: Optional[np.ndarray] = dataclasses.field(init=False, repr=False)

    @property
    def values_block(self) -> Optional[np.ndarray]:
        if self.n_surface_points is None or self.n_surface_points == 0:
            return self._values_block
        else:
            return self._values_block[:, :-self.n_surface_points]

    @values_block.setter
    def values_block(self, value: np.ndarray):
        self._values_block = value

    @property
    def n_surface_points(self):
        return self.exported_fields.n_surface_points

    @property
    def n_points_per_surface(self):
        return self.exported_fields.n_points_per_surface

    @property
    def grid_size(self):
        return self.values_block.shape[1]

    @property
    def scalar_field_at_sp(self):
        return self.exported_fields.scalar_field_at_surface_points

    @property
    def exported_fields_regular_grid(self):
        scalar_field = self.exported_fields.scalar_field[:self.grid.len_grids[0]]
        gx_field = self.exported_fields.scalar_field[:self.grid.len_grids[0]]
        gy_field = self.exported_fields.scalar_field[:self.grid.len_grids[0]]
        gz_field = self.exported_fields.scalar_field[:self.grid.len_grids[0]]

        return ExportedFields(scalar_field, gx_field, gy_field, gz_field)

    @property
    def values_block_regular_grid(self):
        return self.values_block[:, self.grid.len_grids[0]]
