import warnings
from dataclasses import dataclass
from typing import List

import numpy as np

from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.interp_output import InterpOutput


@dataclass(init=True)
class OctreeLevel:
    # Input
    grid: EngineGrid
    outputs: list[InterpOutput]  #: List of output (one per stack)

    # Topo
    edges_id: np.ndarray = None
    count_edges: np.ndarray = None
    marked_edges: List[np.ndarray] = None  # 3 arrays in x, y, z

    def __repr__(self):
        return f"OctreeLevel({len(self.outputs)} outputs_centers"

    def _repr_html_(self):
        return f"<b>OctreeLevel:</b> {len(self.outputs)} outputs_centers"

    @property
    def dxdydz(self):
        return self.grid.octree_dxdydz

    @property
    def last_output_center(self):
        return self.outputs[-1]
    
    @property
    def outputs_centers(self):
        warnings.warn('This function is deprecated', DeprecationWarning)
        return self.outputs

    @property
    def litho_faults_ids_corners_grid(self):
        return self.outputs[-1].litho_faults_ids_corners_grid

    @property
    def number_of_outputs(self):
        return len(self.outputs)