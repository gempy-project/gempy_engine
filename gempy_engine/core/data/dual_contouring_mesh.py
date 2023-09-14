import numpy as np
from dataclasses import dataclass
from typing import Optional

from gempy_engine.core.data.dual_contouring_data import DualContouringData


@dataclass
class DualContouringMesh:
    vertices: np.ndarray
    edges: np.ndarray
    dc_data: Optional[DualContouringData] = None  # * In principle we need this just for testing

    def __repr__(self):
        return f"DualContouringMesh({self.vertices.shape[0]} vertices, {self.edges.shape[0]} edges)"

    def _repr_html_(self):
        return f"<b>DualContouringMesh:</b> {self.vertices.shape[0]} vertices, {self.edges.shape[0]} edges"
