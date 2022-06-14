from dataclasses import dataclass
from typing import Optional

import numpy as np

from gempy_engine.core.data.dual_contouring_data import DualContouringData


@dataclass
class DualContouringMesh:
    vertices: np.ndarray
    edges: np.ndarray
    dc_data: Optional[DualContouringData] = None  # * In principle we need this just for testing
