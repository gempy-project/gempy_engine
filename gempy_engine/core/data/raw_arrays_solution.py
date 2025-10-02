import enum
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from .transforms import Transform
from ..backend_tensor import BackendTensor
from .interp_output import InterpOutput
from .dual_contouring_mesh import DualContouringMesh
from .octree_level import OctreeLevel

# ? These two imports are suggesting that this class should be splatted and move one half into a gempy.module
from ...modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level
from .output.blocks_value_type import ValueType
from ...optional_dependencies import require_pandas, require_subsurface



@dataclass(init=True)
class RawArraysSolution:
    class BlockSolutionType(enum.Enum):
        NONE = 0
        OCTREE = 1
        DENSE_GRID = 2
        

    # region Regular Grid
    lith_block: np.ndarray = field(default_factory=lambda: np.empty(0))
    fault_block: np.ndarray = field(default_factory=lambda: np.empty(0))
    litho_faults_block: np.ndarray = field(default_factory=lambda: np.empty(0))

    scalar_field_matrix: np.ndarray = field(default_factory=lambda: np.empty(0))
    block_matrix: np.ndarray = field(default_factory=lambda: np.empty(0))
    mask_matrix: np.ndarray = field(default_factory=lambda: np.empty(0))
    mask_matrix_squeezed: np.ndarray = field(default_factory=lambda: np.empty(0))
    values_matrix: np.ndarray = field(default_factory=lambda: np.empty(0))
    gradient: np.ndarray = field(default_factory=lambda: np.empty(0))
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
    vertices: list[np.ndarray] = field(default_factory=lambda: np.empty((0, 3)))
    edges: list[np.ndarray] = None
    # endregion

    # region topology
    topology_edges: Optional[np.ndarray] = None  # ? I am not 100% sure the type is correct
    topology_count: Optional[np.ndarray] = None  # ? I am not 100% sure the type is correct

    # endregion
    scalar_field_at_surface_points: list[np.ndarray] = None
    n_surfaces: int = 0

    # ? TODO: This could be just the init
    @classmethod
    def from_gempy_engine_solutions(cls, octrees_output: list[OctreeLevel], meshes: list[DualContouringMesh],
                                    block_solution_type: BlockSolutionType) -> Union["RawArraysSolution", None]:
        raw_arrays_solution = cls()

        first_level_octree: OctreeLevel = octrees_output[0]
        stacks_output = first_level_octree.outputs_centers
        collapsed_output: InterpOutput = stacks_output[-1]  # ! This is the scalar field. Usually we want the last one but not necessarily
        # ? DEP: last_octree_level: OctreeLevel = octrees_output[(-1)]

        raw_arrays_solution._set_scalar_field_at_surface_points(first_level_octree)

        # Region Blocks
        match block_solution_type:
            case cls.BlockSolutionType.OCTREE:
                _fill_block_solutions_with_octree_output(octrees_output, raw_arrays_solution)
                raw_arrays_solution.dense_ids = BackendTensor.t.to_numpy(collapsed_output.ids_block_octree_grid)
            case cls.BlockSolutionType.DENSE_GRID:
                _fill_block_solutions_with_dense_grid(stacks_output, raw_arrays_solution)
                raw_arrays_solution.dense_ids = BackendTensor.t.to_numpy(collapsed_output.ids_block_dense_grid)

        # Endregion

        # Region Grids

        # TODO: Make this more clever to account for the fact that we can have more than one scalar field
        raw_arrays_solution.geological_map = BackendTensor.t.to_numpy(collapsed_output.geological_map)
        raw_arrays_solution.sections = BackendTensor.t.to_numpy(collapsed_output.sections)
        raw_arrays_solution.custom = BackendTensor.t.to_numpy(collapsed_output.custom_grid_values)
        # End region

        # Region Meshes
        if meshes:
            raw_arrays_solution.vertices = [mesh.vertices for mesh in meshes]
            raw_arrays_solution.edges = [mesh.edges for mesh in meshes]

        # Endregion
        return raw_arrays_solution


    def _set_scalar_field_at_surface_points(self, octree_output: OctreeLevel):
        temp_list = []
        self.n_surfaces = 0
        for i in range(octree_output.number_of_outputs):
            at_surface_points = octree_output.outputs_centers[i].scalar_fields.exported_fields.scalar_field_at_surface_points
            temp_list.append(BackendTensor.t.to_numpy(at_surface_points))
            self.n_surfaces += at_surface_points.shape[0]

        self.scalar_field_at_surface_points = temp_list

    def meshes_to_subsurface(self, input_transform: Transform | None = None):
        ss = require_subsurface()
        pd = require_pandas()

        vertex: list[np.ndarray] = self.vertices
        simplex_list: list[np.ndarray] = self.edges
        
        idx_max = 0
        for i, simplex_array in enumerate(simplex_list):
            simplex_array += idx_max
            idx_max += vertex[i].shape[0]  # Add the number of vertices in this mesh

        vertex_id_array = [np.full(v.shape[0], i + 1) for i, v in enumerate(vertex)]
        cell_id_array = [np.full(v.shape[0], i + 1) for i, v in enumerate(simplex_list)]

        concatenated_id_array = np.concatenate(vertex_id_array)
        concatenated_cell_id_array = np.concatenate(cell_id_array)

        all_vertex = np.concatenate(vertex)
        if input_transform:
            all_vertex = input_transform.apply_inverse(all_vertex)
        
        meshes: ss.UnstructuredData = ss.UnstructuredData.from_array(
            vertex=all_vertex,
            cells=np.concatenate(simplex_list),
            vertex_attr=pd.DataFrame({'id': concatenated_id_array}),
            cells_attr=pd.DataFrame({'id': concatenated_cell_id_array})
        )

        return meshes


def _fill_block_solutions_with_octree_output(octrees_output, raw_arrays_solution: "RawArraysSolution"):

    # Region Local Functions
    def ___get_regular_grid_values_for_all_structural_groups(octree_output: list[OctreeLevel], scalar_type: ValueType):
        temp_list = []
        for i in range(octree_output[0].number_of_outputs):
            dense_valuse_from_octree: np.ndarray = get_regular_grid_value_for_level(
                octree_list=octree_output,
                level=None,
                value_type=scalar_type,
                scalar_n=i
            ).ravel()
            temp_list.append(dense_valuse_from_octree)
        scalar_field_stacked = np.vstack(temp_list)
        return scalar_field_stacked

    # Endregion

    raw_arrays_solution.block_matrix = ___get_regular_grid_values_for_all_structural_groups(
        octree_output=octrees_output,
        scalar_type=ValueType.values_block
    )
    raw_arrays_solution.fault_block = get_regular_grid_value_for_level(
        octree_list=octrees_output,
        level=None,
        value_type=ValueType.faults_block
    ).astype("int8").ravel()
    raw_arrays_solution.litho_faults_block = get_regular_grid_value_for_level(
        octree_list=octrees_output,
        level=None,
        value_type=ValueType.litho_faults_block
    ).astype("int32").ravel()
    raw_arrays_solution.scalar_field_matrix = ___get_regular_grid_values_for_all_structural_groups(
        octree_output=octrees_output,
        scalar_type=ValueType.scalar
    )
    raw_arrays_solution.mask_matrix = ___get_regular_grid_values_for_all_structural_groups(
        octree_output=octrees_output,
        scalar_type=ValueType.mask_component
    )
    raw_arrays_solution.mask_matrix_squeezed = ___get_regular_grid_values_for_all_structural_groups(
        octree_output=octrees_output,
        scalar_type=ValueType.squeeze_mask
    )

    lith_block_temp = get_regular_grid_value_for_level(
        octree_list=octrees_output,
        level=None,
        value_type=ValueType.ids
    ).astype("int8").ravel()
    lith_block_temp[lith_block_temp == 0] = lith_block_temp.max() + 1  # Move basement from first to last
    raw_arrays_solution.lith_block = lith_block_temp


def _fill_block_solutions_with_dense_grid(stacks_output: list[InterpOutput], raw_arrays_solution: "RawArraysSolution"):
    # Region Local Functions
    def ___get_regular_grid_values_for_all_structural_groups(stacks_output: list[InterpOutput], scalar_type: ValueType):
        temp_list = []
        for stacks_output in stacks_output:
            dense_values_from_dense_grid = stacks_output.get_block_from_value_type(
                value_type=scalar_type,
                slice_=stacks_output.grid.dense_grid_slice
            )
            temp_list.append(dense_values_from_dense_grid)
        scalar_field_stacked = np.vstack(temp_list)
        return scalar_field_stacked

    # Endregion

    collapsed_output = stacks_output[-1]

    raw_arrays_solution.block_matrix = ___get_regular_grid_values_for_all_structural_groups(stacks_output, ValueType.values_block)
    raw_arrays_solution.fault_block = collapsed_output.get_block_from_value_type(ValueType.faults_block, slice_=collapsed_output.grid.dense_grid_slice).astype("int8").ravel()
    raw_arrays_solution.litho_faults_block = collapsed_output.get_block_from_value_type(ValueType.litho_faults_block, slice_=collapsed_output.grid.dense_grid_slice).astype("int32").ravel()
    raw_arrays_solution.scalar_field_matrix = ___get_regular_grid_values_for_all_structural_groups(stacks_output, ValueType.scalar)
    raw_arrays_solution.mask_matrix = ___get_regular_grid_values_for_all_structural_groups(stacks_output, ValueType.mask_component)
    raw_arrays_solution.mask_matrix_squeezed = ___get_regular_grid_values_for_all_structural_groups(stacks_output, ValueType.squeeze_mask)

    lith_block_temp = collapsed_output.get_block_from_value_type(ValueType.ids, slice_=collapsed_output.grid.dense_grid_slice).astype("int8").ravel()
    lith_block_temp[lith_block_temp == 0] = lith_block_temp.max() + 1
    raw_arrays_solution.lith_block = lith_block_temp


