from dataclasses import field
from typing import List

import numpy as np

from gempy_engine.config import SET_RAW_ARRAYS_IN_SOLUTION
from .dual_contouring_mesh import DualContouringMesh
from .octree_level import OctreeLevel
from .raw_arrays_solution import RawArraysSolution


class Solutions:
    octrees_output: List[OctreeLevel]
    dc_meshes: List[DualContouringMesh]
    _raw_arrays: RawArraysSolution = field(init=False)
    # ------
    gravity: np.ndarray = None
    magnetics: np.ndarray = None

    debug_input_data: dict = {}
    
    def __init__(self, octrees_output: List[OctreeLevel], dc_meshes: List[DualContouringMesh] = None):
        self.octrees_output = octrees_output
        self.dc_meshes = dc_meshes
        
        if SET_RAW_ARRAYS_IN_SOLUTION:  # * This can add an unnecessary overhead
            self._raw_arrays = RawArraysSolution.from_gempy_engine_solutions(
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


    def meshes_to_unstruct(self) -> "subsurface.UnstructuredData":
        meshes = self.dc_meshes
        import subsurface
        import pandas as pd
        n_meshes = len(meshes)

        vertex_array = np.concatenate([meshes[i].vertices for i in range(n_meshes)])
        simplex_array = np.concatenate([meshes[i].edges for i in range(n_meshes)])

        # * Prepare the simplex array
        simplex_array = meshes[0].edges
        for i in range(1, n_meshes):
            adder = np.max(meshes[i - 1].edges) + 1
            add_mesh = meshes[i].edges + adder
            simplex_array = np.append(simplex_array, add_mesh, axis=0)

        # * Prepare the cells_attr array
        ids_array = np.ones(simplex_array.shape[0])
        l0 = 0
        id = 1
        for mesh in meshes:
            l1 = l0 + mesh.edges.shape[0]
            ids_array[l0:l1] = id
            l0 = l1
            id += 1

        # * Create the unstructured data
        unstructured_data = subsurface.UnstructuredData.from_array(
            vertex=vertex_array,
            cells=simplex_array,
            cells_attr=pd.DataFrame(ids_array, columns=['id'])
            # TODO: We have to create an array with the shape of simplex array with the id of each simplex
        )

        return unstructured_data

    
