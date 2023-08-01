from dataclasses import dataclass, field
from typing import List

import numpy as np

from .dual_contouring_mesh import DualContouringMesh
from .legacy_solutions import LegacySolution
from .octree_level import OctreeLevel


class Solutions:
    octrees_output: List[OctreeLevel]
    dc_meshes: List[DualContouringMesh]
    _raw_arrays: LegacySolution = field(init=False)
    # ------
    gravity: np.ndarray = None
    magnetics: np.ndarray = None

    debug_input_data: dict = {}
    
    def __init__(self, octrees_output: List[OctreeLevel], dc_meshes: List[DualContouringMesh] = None):
        self.octrees_output = octrees_output
        self.dc_meshes = dc_meshes
        
        # TODO: Probably here is the place to fill the LegacySolution
        self._raw_arrays = LegacySolution.from_gempy_engine_solutions(
            octrees_output=octrees_output,
            meshes=dc_meshes
        )

    def __repr__(self):
        return f"Solutions({len(self.octrees_output)} Octree Levels, {len(self.dc_meshes)} DualContouringMeshes)"

    def _repr_html_(self):
        return f"<b>Solutions:</b> {len(self.octrees_output)} Octree Levels, {len(self.dc_meshes)} DualContouringMeshes"
    
    # def __repr__(self):
    #     return f"{self.__class__.__name__}({self.octrees_output})"
    
    @property
    def raw_arrays(self):
        return self._raw_arrays
