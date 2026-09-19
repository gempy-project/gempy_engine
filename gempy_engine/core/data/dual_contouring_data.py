from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .options.evaluation_options import TriangulationMethod


@dataclass(init=True)
class DualContouringData:
    xyz_on_edge: np.ndarray
    valid_edges: np.ndarray

    xyz_on_centers: np.ndarray
    dxdydz: np.ndarray | tuple[float, float, float]

    n_surfaces_to_export: int
    left_right_codes: np.ndarray
    gradients: np.ndarray = None

    tree_depth: int = -1
    base_number: tuple[int, int, int] = None
    # Water tight 

    bias_center_mass: np.ndarray = None  # * Only for testing
    bias_normals: np.ndarray = None  # * Only for testing

    # Weighted QEF: extra constraints from other surfaces at overlapping voxels
    extra_edge_xyz: Optional[np.ndarray] = None      # (n_valid_voxels, K, 3)
    extra_edge_normals: Optional[np.ndarray] = None   # (n_valid_voxels, K, 3)
    extra_weights: Optional[np.ndarray] = None        # (n_valid_voxels, K)
    triangulation_method: TriangulationMethod = TriangulationMethod.LEGACY
    generated_cell_coordinates: Optional[np.ndarray] = None  # Before geological masking.
    triangulation_report: dict = field(default_factory=dict)  # Quad support before overlap/fault triangle removal.
    strict_crossings: bool = False

    @property
    def valid_voxels(self):
        return self.valid_edges.sum(axis=1, dtype=bool)

    @property
    def n_valid_edges(self):
        return self.valid_edges.shape[0]
    
    @property
    def n_evaluations_on_edges(self):
        return self.xyz_on_edge.shape[0]
