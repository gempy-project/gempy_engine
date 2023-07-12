from dataclasses import dataclass
from typing import Optional

import numpy as np

from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level


@dataclass(init=False)
class LegacySolution:
    # region Input data results
    weights_vector: np.ndarray
    scalar_field_at_surface_points: np.ndarray
    block_at_surface_points: np.ndarray
    mask_at_surface_points: np.ndarray
    values_at_surface_points: np.ndarray

    # endregion

    # region Regular Grid
    lith_block: np.ndarray
    scalar_field_matrix: np.ndarray
    block_matrix: np.ndarray
    mask_matrix: np.ndarray
    mask_matrix_pad: np.ndarray
    values_matrix: np.ndarray
    gradient: np.ndarray
    # endregion

    # region other grids
    geological_map: Optional[np.ndarray] = None
    sections: Optional[np.ndarray] = None
    custom: Optional[np.ndarray] = None
    # endregion

    # region Geophysics
    fw_gravity: Optional[np.ndarray] = None
    fw_magnetic: Optional[np.ndarray] = None
    # endregion

    # region Mesh
    vertices: list[np.ndarray] = None
    edges: list[np.ndarray] = None
    # endregion

    # region topology
    topology_edges: Optional[np.ndarray] = None  # ? I am not 100% sure the type is correct
    topology_count: Optional[np.ndarray] = None  # ? I am not 100% sure the type is correct

    # endregion

    @classmethod
    def from_gempy_engine_solutions(cls, gempy_engine_solutions: "Solutions") -> "LegacySolution":
        legacy_solution = cls()

        octree_lvl = -1
        octree_output: OctreeLevel = gempy_engine_solutions.octrees_output[octree_lvl]

        # ! I will have to use this as soon as I am using octrees
        regular_grid_scalar: np.ndarray = get_regular_grid_value_for_level(
            octree_list=gempy_engine_solutions.octrees_output
        ).astype("int8")

        legacy_solution._set_block_matrix(octree_output)
        legacy_solution._set_lith_block(octree_output)
        legacy_solution._set_scalar_field(octree_output)

        legacy_solution._set_scalar_field_at_surface_points(octree_output)

        meshes: list[DualContouringMesh] = gempy_engine_solutions.dc_meshes
        if meshes:
            legacy_solution.vertices = [mesh.vertices for mesh in meshes]
            legacy_solution.edges = [mesh.edges for mesh in meshes]
            # TODO: I will have to apply the transform to this one
        
        return legacy_solution

    def _set_block_matrix(self, octree_output: OctreeLevel):
        temp_list = []
        for i in range(octree_output.number_of_outputs):
            temp_list.append(octree_output.outputs_centers[i].values_block)

        block_matrix_stacked = np.vstack(temp_list)
        self.block_matrix = block_matrix_stacked

    def _set_scalar_field(self, octree_output: OctreeLevel):
        temp_list = []
        for i in range(octree_output.number_of_outputs):
            temp_list.append(octree_output.outputs_centers[i].scalar_fields.exported_fields.scalar_field)

        scalar_field_stacked = np.vstack(temp_list)
        self.scalar_field_matrix = scalar_field_stacked

    def _set_scalar_field_at_surface_points(self, octree_output: OctreeLevel):
        temp_list = []
        for i in range(octree_output.number_of_outputs):
            temp_list.append(octree_output.outputs_centers[i].scalar_fields.exported_fields.scalar_field_at_surface_points)

        self.scalar_field_at_surface_points = temp_list

    def _set_lith_block(self, octree_output: OctreeLevel):
        block = octree_output.last_output_center.ids_block
        block[block == 0] = block.max() + 1
        self.lith_block = block
