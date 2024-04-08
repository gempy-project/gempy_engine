from dataclasses import field
from typing import List, Optional

import numpy as np

from gempy_engine.config import SET_RAW_ARRAYS_IN_SOLUTION
from gempy_engine.core.backend_tensor import BackendTensor
from .dual_contouring_mesh import DualContouringMesh
from .octree_level import OctreeLevel
from .raw_arrays_solution import RawArraysSolution


class Solutions:
    octrees_output: List[OctreeLevel]
    dc_meshes: List[DualContouringMesh]
    scalar_field_at_surface_points: np.ndarray = np.empty(0)
    _ordered_elements: List[np.ndarray] = []
    _raw_arrays: Optional[RawArraysSolution] = field(init=False)
    # ------
    gravity: np.ndarray = None
    magnetics: np.ndarray = None

    debug_input_data: dict = {}

    def __init__(self, octrees_output: List[OctreeLevel], dc_meshes: List[DualContouringMesh] = None, fw_gravity: np.ndarray = None,
                 block_solution_type: RawArraysSolution.BlockSolutionType = RawArraysSolution.BlockSolutionType.OCTREE):
        self.octrees_output = octrees_output
        self.dc_meshes = dc_meshes
        self.gravity = fw_gravity

        self._set_scalar_field_at_surface_points_and_elements_order(octrees_output)
            
        if SET_RAW_ARRAYS_IN_SOLUTION and octrees_output[0].grid_centers.octree_grid is not None:  # * This can add an unnecessary overhead
            # TODO: Trying to guess if is dense or octree
            self._raw_arrays = RawArraysSolution.from_gempy_engine_solutions(
                octrees_output=octrees_output,
                meshes=dc_meshes,
                block_solution_type=block_solution_type
            )
        else:
            self._raw_arrays = None


    def __repr__(self):
        return f"Solutions({len(self.octrees_output)} Octree Levels, {len(self.dc_meshes)} DualContouringMeshes)"

    def _repr_html_(self):
        return f"<b>Solutions:</b> {len(self.octrees_output)} Octree Levels, {len(self.dc_meshes)} DualContouringMeshes"

    @property
    def raw_arrays(self) -> RawArraysSolution:
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
    
    def _set_scalar_field_at_surface_points_and_elements_order(self, octrees_output):
        self.scalar_field_at_surface_points = np.empty(0)
        self._ordered_elements = []
        
        for structural_group in octrees_output[0].outputs_centers:
            scalar_field_at_surface_points_data = BackendTensor.t.to_numpy(structural_group.scalar_field_at_sp)
            self.scalar_field_at_surface_points = np.append(
                self.scalar_field_at_surface_points,
                scalar_field_at_surface_points_data
            )

            # Order self_scalar_field_at_surface_points
            self._ordered_elements.append(np.argsort(scalar_field_at_surface_points_data)[::-1])
