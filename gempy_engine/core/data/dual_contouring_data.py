from dataclasses import dataclass
from typing import Optional

import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.exported_fields import ExportedFields


@dataclass(init=True)
class DualContouringData:
    xyz_on_edge: np.ndarray
    valid_edges: np.ndarray

    xyz_on_centers: np.ndarray
    dxdydz: np.ndarray | tuple[float, float, float]

    exported_fields_on_edges: Optional[ExportedFields]

    n_surfaces_to_export: int
    _gradients: np.ndarray = None

    tree_depth: int = -1
    # Water tight 
    mask: np.ndarray = None

    bias_center_mass: np.ndarray = None  # * Only for testing
    bias_normals: np.ndarray = None  # * Only for testing

    def __post_init__(self):
        if self.exported_fields_on_edges is not None:
            ef = self.exported_fields_on_edges
            self._gradients = BackendTensor.t.stack((ef.gx_field, ef.gy_field, ef.gz_field), axis=0).T  # ! When we are computing the edges for dual contouring there is no surface points

    @property
    def gradients(self):
        return self._gradients

    @property
    def valid_voxels(self):
        return self.valid_edges.sum(axis=1, dtype=bool)

    @property
    def n_edges(self):
        return self.valid_edges.shape[0]