import warnings
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

from gempy_engine.core.data.grid import Grid
from gempy_engine.core.data.interp_output import InterpOutput


@dataclass(init=True)
class OctreeLevel:
    # Input
    grid_centers: Grid
    grid_corners: Optional[Grid]
    outputs_centers: List[InterpOutput]  #: List of output (one per stack)
    outputs_corners: list[InterpOutput]

    # Topo
    edges_id: np.ndarray = None
    count_edges: np.ndarray = None
    marked_edges: List[np.ndarray] = None  # 3 arrays in x, y, z

    def __repr__(self):
        return f"OctreeLevel({len(self.outputs_centers)} outputs_centers, {len(self.outputs_corners)} outputs_corners)"

    def _repr_html_(self):
        return f"<b>OctreeLevel:</b> {len(self.outputs_centers)} outputs_centers, {len(self.outputs_corners)} outputs_corners"

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
    
    @property
    def number_of_outputs(self):
        return len(self.outputs_centers)