from typing import List, Optional

import numpy as np

from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.core.data.output.blocks_value_type import ValueType

from ._octree_internals import compute_next_octree_locations
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.engine_grid import EngineGrid
from ...config import AvailableBackends
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


def get_regular_grid_value_for_level(octree_list: List['OctreeLevel'], level: Optional[int] = None,
                                     value_type: 'ValueType' = ValueType.ids, scalar_n=-1) -> "torch.Tensor":
    if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
        return get_regular_grid_value_for_level_torch(octree_list, level, value_type, scalar_n)
    else:
        return get_regular_grid_value_for_level_numpy(octree_list, level, value_type, scalar_n)


def get_regular_grid_value_for_level_torch(octree_list: List['OctreeLevel'], level: Optional[int] = None,
                                           value_type: 'ValueType' = ValueType.ids, scalar_n=-1) -> "torch.Tensor":
    import torch
    # region Internal Functions ==================================================
    def calculate_oct(shape, n_rep: int, device: torch.device) -> torch.Tensor:
        # Replaced repeats/tiles with meshgrid. Much faster and uses less memory.
        f1 = int(shape[2] * shape[1] * (2 ** (n_rep - 1)) * (2 ** (n_rep - 1)))
        f2 = int(shape[2] * (2 ** (n_rep - 1)))
        e = 2 ** n_rep

        n_voxel_per_dim = torch.arange(e, device=device)

        # Create a 3D grid of indices natively
        i, j, k = torch.meshgrid(n_voxel_per_dim, n_voxel_per_dim, n_voxel_per_dim, indexing='ij')

        oct_tensor = (i * f1 + j * f2 + k).flatten()
        return oct_tensor

    def get_global_anchor(activ: torch.Tensor, branch_res, n_rep: int) -> torch.Tensor:
        f1 = int(branch_res[2] * branch_res[1])
        f2 = int(branch_res[2])

        f1b = f1 * (2 ** (3 * n_rep))
        f2b = f2 * (2 ** (2 * n_rep))
        f3b = 2 ** (1 * n_rep)

        # PyTorch requires explicit integer division (truncation)
        d1 = torch.div(activ, f1, rounding_mode='trunc') * f1b
        d2 = torch.div(activ % f1, f2, rounding_mode='trunc') * f2b
        d3 = (activ % f1 % f2) * f3b

        return d1 + d2 + d3

    def _expand_regular_grid(active_cells_erg: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 0:
            return active_cells_erg.flatten()

        f = 2 ** n_rep
        D, H, W = active_cells_erg.shape
        # View/Expand is an O(1) memory trick in PyTorch that avoids allocating 
        # memory until the final reshape forces a copy.
        x = active_cells_erg.view(D, 1, H, 1, W, 1)
        x = x.expand(D, f, H, f, W, f)
        return x.reshape(D * f, H * f, W * f).flatten()

    def _expand_octree(active_cells_eo: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 0:
            return active_cells_eo.flatten()

        f = 2 ** n_rep
        N, D, H, W = active_cells_eo.shape  # Expected shape: (-1, 2, 2, 2)
        x = active_cells_eo.view(N, D, 1, H, 1, W, 1)
        x = x.expand(N, D, f, H, f, W, f)
        return x.reshape(N, D * f, H * f, W * f).flatten()

    # endregion =================================================================================

    if level is None:
        level = len(octree_list) - 1

    if level > (len(octree_list) - 1):
        raise ValueError("Level cannot be larger than the number of octrees.")

    # Octree - Level 0
    root = octree_list[0]
    regular_grid_shape = torch.as_tensor(root.grid.octree_grid_shape)

    # Fetch block as a PyTorch tensor
    block = _get_block_from_value_type(root, scalar_n, value_type)
    if not isinstance(block, torch.Tensor):
        block = torch.tensor(block)

    device = block.device  # Dynamically infer device from the root tensor

    regular_grid = _expand_regular_grid(
        active_cells_erg=block.view(tuple(regular_grid_shape.tolist())),
        n_rep=level
    ).clone()  # Clone ensures it's contiguous and mutable in memory

    shape = regular_grid_shape

    active_cells_index: List[torch.Tensor] = []
    global_active_cells_index = torch.empty(0, dtype=torch.long, device=device)

    # Octree - Level n
    for e, octree in enumerate(octree_list[1:level + 1]):
        n_rep = (level - e)

        # Keep active_cells on the GPU! Do not convert to NumPy.
        active_cells = octree.grid.octree_grid.active_cells
        if not isinstance(active_cells, torch.Tensor):
            active_cells = torch.tensor(active_cells, device=device)

        local_active_cells = torch.where(active_cells)[0]
        shape = torch.as_tensor(octree.grid.octree_grid_shape, device=device)

        oct_tensor = calculate_oct(shape, n_rep, device)

        block = _get_block_from_value_type(octree, scalar_n, value_type)
        if not isinstance(block, torch.Tensor):
            block = torch.tensor(block, device=device)

        ids = _expand_octree(block.view(-1, 2, 2, 2), n_rep - 1)

        is_branch = e > 0

        if is_branch:
            local_shape = torch.tensor([2, 2, 2], device=device)
            local_anchors = get_global_anchor(local_active_cells, local_shape, n_rep)

            # Indexing requires type long
            global_anchors = global_active_cells_index[local_anchors.long()]
            global_active_cells_index = (global_anchors.view(-1, 1) + oct_tensor).flatten()

            regular_grid[global_active_cells_index.long()] = ids
        else:
            local_shape = torch.div(shape, 2, rounding_mode='trunc')
            local_anchors = get_global_anchor(local_active_cells, local_shape, n_rep)

            global_active_cells_index = (local_anchors.view(-1, 1) + oct_tensor).flatten()
            regular_grid[global_active_cells_index.long()] = ids

        active_cells_index.append(global_active_cells_index)

    return regular_grid.view(tuple(shape.tolist()))
    # return regular_grid.view(tuple(regular_grid_shape.tolist()))


def get_regular_grid_value_for_level_numpy(octree_list: List[OctreeLevel], level: Optional[int] = None,
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

    regular_grid_shape = root.grid.octree_grid_shape

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
        active_cells: np.ndarray = octree.grid.octree_grid.active_cells
        active_cells = BackendTensor.t.to_numpy(active_cells)

        local_active_cells: np.ndarray = np.where(active_cells)[0]
        shape: np.ndarray = octree.grid.octree_grid_shape
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
    element_output: InterpOutput = root.outputs[scalar_n]
    block = element_output.get_block_from_value_type(value_type, element_output.grid.octree_grid_slice)
    return block
