from dataclasses import dataclass
from typing import List

import numpy as np

from gempy_engine.core.data.grid import Grid


@dataclass
class ExportedFields:
    scalar_field: np.ndarray
    gx_field: np.ndarray
    gy_field: np.ndarray
    gz_field: np.ndarray = None


@dataclass(init=False)
class InterpOutput:
    weights: np.ndarray
    grid: Grid

    scalar_field_at_sp: np.ndarray
    exported_fields: ExportedFields
    values_block: np.ndarray  # final values ignoring unconformities
    final_block: np.ndarray  # Masked array containing only the active voxels

    # Remember this is only for regular grid
    octrees: List[np.ndarray]

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

    @property
    def ids_block_regular_grid(self):
        return np.rint(self.values_block[0, :self.grid.len_grids[0]].reshape(self.grid.regular_grid_shape))


@dataclass(init=True)
class OctreeLevel():
    # Used for octree
    xyz_coords: np.ndarray
    id_block: np.ndarray = None

    # Used for dual contouring
    exported_fields: ExportedFields = None

    # Outcome
    edges_id: np.ndarray = None
    count_edges: np.ndarray = None

    is_root: bool = False # When root is true arrays are dim 3