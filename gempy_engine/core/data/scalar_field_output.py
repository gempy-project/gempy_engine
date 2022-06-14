import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np

from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.core.data.exported_structs import MaskMatrices
from gempy_engine.core.data.grid import Grid


@dataclass(init=True)
class ScalarFieldOutput:
    weights: np.ndarray
    grid: Grid

    exported_fields: ExportedFields
    values_block: Optional[np.ndarray]  # final values ignoring unconformities
    _values_block: Optional[np.ndarray] = dataclasses.field(init=False, repr=False)
    mask_components: Optional[MaskMatrices]

    @property
    def values_block(self) -> Optional[np.ndarray]:
        return self._values_block[:self.grid.len_grids[0]]

    @values_block.setter
    def values_block(self, value: np.ndarray):
        self._values_block = value

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
