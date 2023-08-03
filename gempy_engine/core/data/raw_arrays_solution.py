from dataclasses import dataclass
from typing import Optional

import numpy as np

from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level, ValueType


@dataclass(init=True)
class RawArraysSolution:
    # region Input data results
    # ? Do I need these fields?
    # weights_vector: np.ndarray
    # scalar_field_at_surface_points: np.ndarray
    # block_at_surface_points: np.ndarray
    # mask_at_surface_points: np.ndarray
    # values_at_surface_points: np.ndarray

    # endregion

    # region Regular Grid
    lith_block: np.ndarray = np.empty(0)
    scalar_field_matrix: np.ndarray = np.empty(0)
    block_matrix: np.ndarray = np.empty(0)
    mask_matrix: np.ndarray = np.empty(0)
    mask_matrix_pad: np.ndarray = np.empty(0)
    values_matrix: np.ndarray = np.empty(0)
    gradient: np.ndarray = np.empty(0)
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

    # ? TODO: This could be just the init
    @classmethod
    def from_gempy_engine_solutions(cls, octrees_output: list[OctreeLevel], meshes: list[DualContouringMesh]) \
            -> "RawArraysSolution":
        legacy_solution = cls()

        # region Blocks
        last_octree_level: OctreeLevel = octrees_output[(-1)]

        legacy_solution._set_block_matrix(octrees_output)
        legacy_solution._set_lith_block(octrees_output)
        legacy_solution._set_scalar_field(octrees_output)

        legacy_solution._set_scalar_field_at_surface_points(last_octree_level)
        # endregion
        
        # region Grids
        first_level_octree: OctreeLevel = octrees_output[0]
        
        # TODO: Make this more clever to account for the fact that we can have more than one scalar field
        output: InterpOutput = first_level_octree.outputs_centers[-1]  # ! This is the scalar field. Usually we want the last one but not necessarily
        
        legacy_solution.geological_map = output.geological_map
        legacy_solution.sections = output.sections
        legacy_solution.custom = output.custom_grid
        # endregion
        
        # region Meshes
        if meshes:
            legacy_solution.vertices = [mesh.vertices for mesh in meshes]
            legacy_solution.edges = [mesh.edges for mesh in meshes]
            # TODO: I will have to apply the transform to this one
            
        # endregion
        return legacy_solution

    def _set_block_matrix(self, octree_output: list[OctreeLevel]):
        temp_list = []
        for i in range(octree_output[0].number_of_outputs):
            temp_list.append(
                get_regular_grid_value_for_level(
                    octree_list=octree_output,
                    level=None,
                    value_type=ValueType.scalar,
                    scalar_n=i
                ).ravel()
            )
        block_matrix_stacked = np.vstack(temp_list)
        self.block_matrix = block_matrix_stacked

    def _set_scalar_field(self, octree_output:list[OctreeLevel]):
        temp_list = []
        for i in range(octree_output[0].number_of_outputs):
            temp_list.append(
                get_regular_grid_value_for_level(
                    octree_list=octree_output,
                    level=None,
                    value_type=ValueType.scalar,
                    scalar_n=i
                ).ravel()
            )

        scalar_field_stacked = np.vstack(temp_list)
        self.scalar_field_matrix = scalar_field_stacked

    def _set_scalar_field_at_surface_points(self, octree_output: OctreeLevel):
        temp_list = []
        for i in range(octree_output.number_of_outputs):
            temp_list.append(octree_output.outputs_centers[i].scalar_fields.exported_fields.scalar_field_at_surface_points)

        self.scalar_field_at_surface_points = temp_list

    def _set_lith_block(self, octree_output: list[OctreeLevel]):

        block = get_regular_grid_value_for_level(
            octree_list=octree_output,
            level=None,
            value_type=ValueType.ids
        ).astype("int8").ravel()
        
        block[block == 0] = block.max() + 1 # Move basement from first to last
        self.lith_block = block
