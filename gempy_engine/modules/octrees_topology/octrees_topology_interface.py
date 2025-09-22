from typing import List, Optional

import numpy as np

from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.core.data.output.blocks_value_type import ValueType

from ._octree_internals import compute_next_octree_locations
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.engine_grid import EngineGrid
from ...core.backend_tensor import BackendTensor
from ...core.data.options.evaluation_options import EvaluationOptions


# TODO: [ ] Check if fortran order speeds up this function
# TODO: Substitute numpy for b.tfnp
# TODO: Remove all stack to be able to compile TF


def get_next_octree_grid(prev_octree: OctreeLevel, evaluation_options: EvaluationOptions,
                         current_octree_level: int = 9999) -> EngineGrid:

    octree_from_output: EngineGrid = compute_next_octree_locations(
        prev_octree=prev_octree, 
        evaluation_options=evaluation_options,
        current_octree_level=current_octree_level
    )
    return octree_from_output




def get_regular_grid_value_for_level(octree_list: List[OctreeLevel], level: Optional[int] = None,
                                     value_type: ValueType = ValueType.ids, scalar_n=-1) -> np.ndarray:
    # region Internal Functions ==================================================
    def calculate_oct(shape, n_rep: int) -> np.ndarray:

        f1 = shape[2] * shape[1] * 2 ** (n_rep - 1) * 2 ** (n_rep - 1)
        f2 = shape[2] * 2 ** (n_rep - 1)

        e = 2 ** n_rep

        n_voxel_per_dim = np.arange(e)

        d1 = np.repeat(n_voxel_per_dim, e ** 2) * f1
        d2 = np.tile(np.repeat(n_voxel_per_dim, e), e) * f2
        d3 = np.tile(n_voxel_per_dim, e ** 2)

        oct = d1 + d2 + d3
        return oct

    def get_global_anchor(activ, branch_res, n_rep):
        f1 = branch_res[2] * branch_res[1]
        f2 = branch_res[2]

        f1b = f1 * 2 ** (3 * n_rep)
        f2b = f2 * 2 ** (2 * n_rep)
        f3b = 2 ** (1 * n_rep)

        d1 = (activ // f1) * f1b
        d2 = (activ % f1 // f2) * f2b
        d3 = activ % f1 % f2 * f3b

        anchor = d1 + d2 + d3
        return anchor

    def _expand_regular_grid(active_cells_erg: np.ndarray, n_rep: int) -> np.ndarray:
        active_cells_erg = np.repeat(active_cells_erg, 2 ** n_rep, axis=0)
        active_cells_erg = np.repeat(active_cells_erg, 2 ** n_rep, axis=1)
        active_cells_erg = np.repeat(active_cells_erg, 2 ** n_rep, axis=2)
        return active_cells_erg.ravel()

    def _expand_octree(active_cells_eo: np.ndarray, n_rep: int) -> np.ndarray:
        active_cells_eo = np.repeat(active_cells_eo, 2 ** n_rep, axis=1)
        active_cells_eo = np.repeat(active_cells_eo, 2 ** n_rep, axis=2)
        active_cells_eo = np.repeat(active_cells_eo, 2 ** n_rep, axis=3)

        return active_cells_eo.ravel()

    # endregion =================================================================================
    
    if level is None:
        level = len(octree_list) - 1

    if level > (len(octree_list) - 1):
        raise ValueError("Level cannot be larger than the number of octrees.")

    # Octree - Level 0
    root: OctreeLevel = octree_list[0]

    regular_grid_shape = root.grid_centers.octree_grid_shape

    block = _get_block_from_value_type(root, scalar_n, value_type)
    
    regular_grid: np.ndarray = _expand_regular_grid(
        active_cells_erg=block.reshape(regular_grid_shape.tolist()),
        n_rep=level
    )
    
    shape = regular_grid_shape

    active_cells_index: List["np.ndarray[np.int]"] = []
    global_active_cells_index: np.ndarray = np.array([])

    # Octree - Level n
    for e, octree in enumerate(octree_list[1:level + 1]):
        n_rep = (level - e)
        active_cells: np.ndarray = octree.grid_centers.octree_grid.active_cells
        active_cells = BackendTensor.t.to_numpy(active_cells)

        local_active_cells: np.ndarray = np.where(active_cells)[0]
        shape: np.ndarray = octree.grid_centers.octree_grid_shape
        oct: np.ndarray = calculate_oct(shape, n_rep)

        block = _get_block_from_value_type(octree, scalar_n, value_type)

        ids: np.ndarray = _expand_octree(block.reshape((-1, 2, 2, 2)), n_rep - 1)

        is_branch = e > 0

        if is_branch:
            local_shape: "np.ndarray[np.int]" = np.array([2, 2, 2])
            local_anchors = get_global_anchor(local_active_cells, local_shape, n_rep)
            global_anchors = global_active_cells_index[local_anchors]

            global_active_cells_index = (global_anchors.reshape(-1, 1) + oct).ravel()
            regular_grid[global_active_cells_index] = ids  # + (e * 2)
        else:
            local_shape = shape // 2
            local_anchors = get_global_anchor(local_active_cells, local_shape, n_rep)
            global_active_cells_index = (local_anchors.reshape(-1, 1) + oct).ravel()
            regular_grid[global_active_cells_index] = ids  # + (e * 2)

        active_cells_index.append(global_active_cells_index)

    return regular_grid.reshape(shape.tolist())


def _get_block_from_value_type(root: OctreeLevel, scalar_n: int, value_type: ValueType):
    element_output: InterpOutput = root.outputs_centers[scalar_n]
    block = element_output.get_block_from_value_type(value_type, element_output.grid.octree_grid_slice)
    return block
