from dataclasses import dataclass

import numpy as np


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
    # Water tight 

    bias_center_mass: np.ndarray = None  # * Only for testing
    bias_normals: np.ndarray = None  # * Only for testing

    @property
    def valid_voxels(self):
        return self.valid_edges.sum(axis=1, dtype=bool)

    @property
    def n_valid_edges(self):
        return self.valid_edges.shape[0]
    
    @property
    def n_evaluations_on_edges(self):
        return self.xyz_on_edge.shape[0]
    