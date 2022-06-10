import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from gempy_engine.core.data.grid import Grid


@dataclass(init=True)
class ExportedFields:
    _scalar_field: Optional[np.ndarray]
    _gx_field: np.ndarray
    _gy_field: np.ndarray
    _gz_field: np.ndarray = None
    n_points_per_surface: np.ndarray = None
    n_surface_points: Optional[int] = None
    scalar_field_shift: Optional[float] = 0

    @property
    def scalar_field_at_surface_points(self):
        return self._scalar_field[-self.n_surface_points:][self.npf] + self.scalar_field_shift

    @property
    def scalar_field(self):
        if self.n_surface_points is None:
            return self._scalar_field

        return self._scalar_field[:-self.n_surface_points]

    @property
    def gx_field(self):
        if self.n_surface_points is None:
            return self._gx_field
        return self._gx_field[:-self.n_surface_points]

    @property
    def gy_field(self):
        if self.n_surface_points is None:
            return self._gy_field
        return self._gy_field[:-self.n_surface_points]

    @property
    def gz_field(self):
        if self.n_surface_points is None:
            return self._gz_field
        return self._gz_field[:-self.n_surface_points]

    @property
    def npf(self):
        return self.n_points_per_surface

    @classmethod
    def from_interpolation(cls, scalar_field, gx_field, gy_field, gz_field, grid_size: int):
        return cls(scalar_field[:grid_size], gx_field[:grid_size], gy_field[:grid_size], gz_field[:grid_size])


@dataclass()
class MaskMatrices:
    mask_lith: np.ndarray
    mask_fault: Optional[np.ndarray]


@dataclass(init=False)
class InterpOutput:
    weights: np.ndarray
    grid: Grid

    exported_fields: ExportedFields
    final_exported_fields: ExportedFields  # Masked array containing only the active voxels
    values_block: np.ndarray  # final values ignoring unconformities
    final_block: np.ndarray  # Masked array containing only the active voxels

    mask_components: MaskMatrices
    squeezed_mask_array: np.ndarray

    # Remember this is only for regular grid
    octrees: List[np.ndarray]  # TODO: This probably should be one level higher. (unsure what I meant here)

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

    @property
    def ids_block_regular_grid(self):
        if self.final_block is None:
            self.final_block = self.values_block
        
        return np.rint(self.final_block[:self.grid.len_grids[0]].reshape(self.grid.regular_grid_shape))

    @property
    def ids_block(self) -> np.ndarray:
        return np.rint(self.final_block[:self.grid.len_grids[0]])
    
    @ids_block.setter
    def ids_block(self, value):
        self.final_block[:self.grid.len_grids[0]] = value
        


@dataclass(init=False)
class OctreeLevel:
    # Input
    grid_centers: Grid
    grid_corners: Grid
    outputs_centers: List[InterpOutput]
    outputs_corners: List[InterpOutput]
    is_root: bool = False  # When root is true arrays are dim 3

    # Topo
    edges_id: np.ndarray = None
    count_edges: np.ndarray = None
    marked_edges: List[np.ndarray] = None  # 3 arrays in x, y, z

    def set_interpolation_values(self, grid_centers: Grid, grid_faces: Grid,
                                 outputs_centers: List[InterpOutput], outputs_faces: List[InterpOutput]):
        self.grid_centers: Grid = grid_centers
        self.grid_corners: Grid = grid_faces
        self.outputs_centers: List[InterpOutput] = outputs_centers
        self.outputs_corners: List[InterpOutput] = outputs_faces

        return self

    @property
    def dxdydz(self):
        return self.grid_centers.dxdydz

    @property
    def output_centers(self):  # * Alias
        warnings.warn('This function is deprecated', DeprecationWarning)
        return self.last_output_center

    @property
    def last_output_center(self):
        return self.outputs_centers[-1]

    @property
    def output_corners(self):  # * Alias
        warnings.warn('This function is deprecated', DeprecationWarning)
        return self.last_output_corners

    @property
    def last_output_corners(self):
        return self.outputs_corners[-1]


@dataclass(init=True)
class DualContouringData:
    xyz_on_edge: np.ndarray
    valid_edges: np.ndarray
    
    xyz_on_centers: np.ndarray
    dxdydz: np.ndarray
    
    exported_fields_on_edges: ExportedFields
    _gradients: np.ndarray = None
    
    # Water tight 
    mask: np.ndarray = None
    n_surfaces: int =None

    bias_center_mass: np.ndarray = None  # * Only for testing
    bias_normals: np.ndarray = None  # * Only for testing

    def __post_init__(self):
        ef = self.exported_fields_on_edges
        self._gradients = np.stack((ef.gx_field, ef.gy_field, ef.gz_field), axis=0).T  # ! When we are computing the edges for dual contouring there is no surface points
    
    @property
    def gradients(self):
        return self._gradients
    
    
    @property
    def valid_voxels(self):
        return self.valid_edges.sum(axis=1, dtype=bool)
    
    @property
    def n_edges(self):
        return self.valid_edges.shape[0]

@dataclass
class DualContouringMesh:
    vertices: np.ndarray
    edges: np.ndarray
    dc_data: Optional[DualContouringData] = None  # * In principle we need this just for testing
    


@dataclass(init=False, )
class Solutions:
    octrees_output: List[OctreeLevel]
    dc_meshes: List[DualContouringMesh]
    # ------
    gravity: np.ndarray
    magnetics: np.ndarray

    debug_input_data = None
