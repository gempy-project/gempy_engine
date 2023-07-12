from dataclasses import dataclass, field
from typing import List

import numpy as np

from .dual_contouring_mesh import DualContouringMesh
from .legacy_solutions import LegacySolution
from .octree_level import OctreeLevel


class Solutions:
    octrees_output: List[OctreeLevel]
    dc_meshes: List[DualContouringMesh] = None
    raw_arrays: LegacySolution = None 
    # ------
    gravity: np.ndarray = None
    magnetics: np.ndarray = None

    debug_input_data: dict = {}
    
    def __init__(self, octrees_output: List[OctreeLevel]):
        self.octrees_output = octrees_output
        
        # TODO: Probably here is the place to fill the LegacySolution
        self.raw_arrays = LegacySolution.from_gempy_engine_solutions(self)
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.octrees_output})"