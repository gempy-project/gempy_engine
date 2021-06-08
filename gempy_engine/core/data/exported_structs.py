from dataclasses import dataclass
from typing import List

import numpy as np

from gempy_engine.core.data.grid import Grid, RegularGrid


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
    octrees: List[np.ndarray]  # TODO: This probably should be one level higher

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

    @property
    def ids_block(self):
        return np.rint(self.values_block[0, :self.grid.len_grids[0]])


@dataclass(init=False)
class OctreeLevel:
    # Input
    grid_centers: Grid
    grid_faces: Grid
    output_centers: InterpOutput
    output_faces: InterpOutput
    is_root: bool = False  # When root is true arrays are dim 3

    # Topo
    edges_id: np.ndarray = None
    count_edges: np.ndarray = None
    marked_edges: List[np.ndarray] = None  # 3 arrays in x, y, z

    def set_interpolation(self, grid_centers: Grid, grid_faces: Grid,
                          output_centers: InterpOutput, output_faces: InterpOutput):
        self.grid_centers: Grid = grid_centers
        self.grid_faces: Grid = grid_faces
        self.output_centers: InterpOutput = output_centers
        self.output_faces: InterpOutput = output_faces

        return self

    @property
    def dxdydz(self):
        return self.grid_centers.dxdydz


@dataclass(init=True)
class OctreeLevel_DEP():
    grid: Grid

    # TODO: Probably I want to just pass the full output too
    output: InterpOutput

    # Used for octree
    # id_block: np.ndarray = None
    # id_block_centers: np.ndarray = None
    # # Used for dual contouring
    # exported_fields: ExportedFields = None

    # topo
    edges_id: np.ndarray = None
    count_edges: np.ndarray = None
    marked_edges: List[np.ndarray] = None  # 3 arrays in x, y, z

    is_root: bool = False  # When root is true arrays are dim 3

    @property
    def xyz_coords(self):
        return self.grid.values

    @property
    def id_block(self):
        return self.output.ids_block

    @property
    def exported_fields(self):
        return self.output.exported_fields
