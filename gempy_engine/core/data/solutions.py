from dataclasses import dataclass, field
from typing import List

import numpy as np

from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.core.data.octree_level import OctreeLevel


class Solutions:
    octrees_output: List[OctreeLevel]
    dc_meshes: List[DualContouringMesh] = None
    # ------
    gravity: np.ndarray = None
    magnetics: np.ndarray = None

    debug_input_data: dict = {}
    
    def __init__(self, octrees_output: List[OctreeLevel]):
        self.octrees_output = octrees_output
        