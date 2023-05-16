# TODO: Make this methods private
import numpy as np

from gempy_engine.core.data.octree_level import OctreeLevel


def get_left_right_array(octree_list: list[OctreeLevel]) -> np.ndarray:
    # === Local function ===
    def _compute_voxel_binary_code(idx_from_root, dir_idx: int, left_right_all, voxel_select_all):

        # Calculate the voxels from root
        for active_voxels_per_lvl in voxel_select_all:  # * The first level is all True
            idx_from_root = np.repeat(idx_from_root[active_voxels_per_lvl], 8)

        left_right_list = []
        voxel_select_op = list(voxel_select_all[1:])
        voxel_select_op.append(np.ones(left_right_all[-1].shape[0], bool))
        left_right_all = left_right_all[::-1]
        voxel_select_op = voxel_select_op[::-1]

        for e, left_right_per_lvl in enumerate(left_right_all):
            left_right_per_lvl_dir = left_right_per_lvl[:, dir_idx]
            for n_rep in range(e):
                inner = left_right_per_lvl_dir[voxel_select_op[e - n_rep]]
                left_right_per_lvl_dir = np.repeat(inner, 8)  # ? Is it always e?
                # ? Is this repeat wrong?
            left_right_list.append(left_right_per_lvl_dir)

        left_right_list.append(idx_from_root)
        binary_code = np.vstack(left_right_list)
        f = binary_code.T
        return binary_code

    # === Local function ===

    if len(octree_list) == 1:
        # * Not only that, the current implementation only works with pure octree starting at [2,2,2]
        raise ValueError("Octree list must have more than one level")

    voxel_select_all = [octree_iter.grid_centers.regular_grid.active_cells for octree_iter in octree_list[1:]]
    left_right_all = [octree_iter.grid_centers.regular_grid.left_right for octree_iter in octree_list[1:]]

    idx_root_x = np.zeros(8, dtype=bool)
    idx_root_x[4:] = True
    binary_x = _compute_voxel_binary_code(idx_root_x, 0, left_right_all, voxel_select_all)

    idx_root_y = np.zeros(8, dtype=bool)
    idx_root_y[[2, 3, 6, 7]] = True
    binary_y = _compute_voxel_binary_code(idx_root_y, 1, left_right_all, voxel_select_all)

    idx_root_z = np.zeros(8, dtype=bool)
    idx_root_z[1::2] = True
    binary_z = _compute_voxel_binary_code(idx_root_z, 2, left_right_all, voxel_select_all)

    bool_to_int_x = np.packbits(binary_x, axis=0, bitorder="little")
    bool_to_int_y = np.packbits(binary_y, axis=0, bitorder="little")
    bool_to_int_z = np.packbits(binary_z, axis=0, bitorder="little")
    left_right_array = np.vstack((bool_to_int_x, bool_to_int_y, bool_to_int_z)).T

    _StaticTriangulationData.depth = 2
    foo = (left_right_array * _StaticTriangulationData.get_pack_directions_into_bits()).sum(axis=1)

    sorted_indices = np.argsort(foo)
    # left_right_array = left_right_array[sorted_indices]
    return left_right_array


class _StaticTriangulationData:
    depth: int

    @staticmethod
    def get_pack_directions_into_bits() -> np.ndarray:
        base_number = 2 ** _StaticTriangulationData.depth
        # return np.array([1, base_number, base_number ** 2], dtype=np.int64)
        return np.array([base_number ** 2, base_number, 1], dtype=np.int64)

    @staticmethod
    def get_base_array(pack_directions_into_bits: np.ndarray) -> np.ndarray:
        return np.array([pack_directions_into_bits, pack_directions_into_bits * 2, pack_directions_into_bits * 3],
                        dtype=np.int64)

    @staticmethod
    def get_base_number() -> int:
        return 2 ** _StaticTriangulationData.depth


def triangulate(left_right_array: np.ndarray, valid_edges: np.ndarray, tree_depth: int, voxel_normals: np.ndarray):
    # * Variables
    # depending on depth
    _StaticTriangulationData.depth = tree_depth

    edge_vector_a = np.array([0, 0, 0, 0,       -1, -1, 1, 1,       -1,  1, -1, 1])
    edge_vector_b = np.array([-1, -1,-1, 1,      0, 0,0, 0,             -1,  1, -1, 1])
    edge_vector_c = np.array([-1, -1, 1, 1,   -1, -1,1, 1,          0,  0,  0, 0])

    # * Consts
    voxel_code = (left_right_array * _StaticTriangulationData.get_pack_directions_into_bits()).sum(1).reshape(-1, 1)
    # ----------

    indices = []
    all = [0, 3, 4, 7, 8, 11]

    for n in all:
        # TODO: Make sure that changing the edge_vector we do not change
        left_right_array_active_edge = left_right_array[valid_edges[:, n]]
        _ = compute_triangles_for_edge(
            edge_vector_a=edge_vector_a[n],
            edge_vector_b=edge_vector_b[n],
            edge_vector_c=edge_vector_c[n],
            left_right_array_active_edge=left_right_array_active_edge,
            voxel_code=voxel_code,
            voxel_normals=voxel_normals,
            n=n
        )
        indices.append(_)

    return indices


def compute_triangles_for_edge(edge_vector_a, edge_vector_b, edge_vector_c,
                               left_right_array_active_edge, voxel_code, voxel_normals, n):
    """
    Important concepts to understand this triangulation:
    - left_right_array (n_voxels, 3-directions) contains a unique number per direction describing if it is left (even) or right (odd) and the voxel level
    - left_right_array_active_edge (n_voxels - active_voxels_for_given_edge, 3-directions): Subset of left_right for voxels with selected edge active
    - valid_edges (n_voxels - active_voxels_for_given_edge, 1): bool
    - voxel_code (n_voxels, 1): All the compressed codes of the voxels at the leaves
    """

    # region: Removing edges that does not have voxel next to it. Depending on the evaluated edge, the voxels checked are different
    def check_voxels_exist_next_to_edge(coord_col, edge_vector, _left_right_array_active_edge):
        match edge_vector:
            case 0:
                _valid_edges = np.ones(_left_right_array_active_edge.shape[0], dtype=bool)
            case 1:
                _valid_edges = _left_right_array_active_edge[:, coord_col] != _StaticTriangulationData.get_base_number() - 1
            case -1:
                _valid_edges = _left_right_array_active_edge[:, coord_col] != 0
            case _:
                raise ValueError("edge_vector_a must be -1, 0 or 1")
        return _valid_edges

    valid_edges_x = check_voxels_exist_next_to_edge(0, edge_vector_a, left_right_array_active_edge)
    valid_edges_y = check_voxels_exist_next_to_edge(1, edge_vector_b, left_right_array_active_edge)
    valid_edges_z = check_voxels_exist_next_to_edge(2, edge_vector_c, left_right_array_active_edge)

    valid_edges_with_neighbour_voxels = valid_edges_x * valid_edges_y * valid_edges_z  # * In the sense of not being on the side of the model
    left_right_array_active_edge = left_right_array_active_edge[valid_edges_with_neighbour_voxels]
    # * At this point left_right_array_active_edge contains the voxel code of those voxels that have the evaluated
    # * edge cut AND that have nearby voxels
    # endregion

    # region: Compress remaining voxel codes per direction
    # * These are the codes that describe each vertex of the triangle
    edge_vector_0 = np.array([edge_vector_a, 0, 0])
    edge_vector_1 = np.array([0, edge_vector_b, 0])
    edge_vector_2 = np.array([0, 0, edge_vector_c])
    
    binary_idx_0: np.ndarray = left_right_array_active_edge + edge_vector_0  # (n_voxels - active_voxels_for_given_edge - invalid_edges, 3-directions)
    binary_idx_1: np.ndarray = left_right_array_active_edge + edge_vector_1  # (n_voxels - active_voxels_for_given_edge - invalid_edges, 3-directions)
    binary_idx_2: np.ndarray = left_right_array_active_edge + edge_vector_2  # (n_voxels - active_voxels_for_given_edge - invalid_edges, 3-directions)

    compressed_binary_idx_0 = (binary_idx_0 * _StaticTriangulationData.get_pack_directions_into_bits()).sum(axis=1)  # (n_voxels - active_voxels_for_given_edge - invalid_edges, 1)
    compressed_binary_idx_1 = (binary_idx_1 * _StaticTriangulationData.get_pack_directions_into_bits()).sum(axis=1)  # (n_voxels - active_voxels_for_given_edge - invalid_edges, 1)
    compressed_binary_idx_2 = (binary_idx_2 * _StaticTriangulationData.get_pack_directions_into_bits()).sum(axis=1)  # (n_voxels - active_voxels_for_given_edge - invalid_edges, 1)
    # endregion

    # region: Map remaining compressed binary code to all the binary codes at leaves
    mapped_voxel_0 = (voxel_code - compressed_binary_idx_0)  # (n_voxels, n_voxels - active_voxels_for_given_edge - invalid_edges)
    mapped_voxel_1 = (voxel_code - compressed_binary_idx_1)  # (n_voxels, n_voxels - active_voxels_for_given_edge - invalid_edges)
    mapped_voxel_2 = (voxel_code - compressed_binary_idx_2)  # (n_voxels, n_voxels - active_voxels_for_given_edge - invalid_edges)
    # endregion

    # region: Find and remove edges at the border of the extent
    code__a_prod_edge = ~mapped_voxel_0.all(axis=0)  # mapped_voxel_0.prod(axis=0) == 0  # (n_voxels - active_voxels_for_given_edge - invalid_edges, 1)
    code__b_prod_edge = ~mapped_voxel_1.all(axis=0)  # mapped_voxel_1.prod(axis=0) == 0  # (n_voxels - active_voxels_for_given_edge - invalid_edges, 1)
    code__c_prod_edge = ~mapped_voxel_2.all(axis=0)  # mapped_voxel_2.prod(axis=0) == 0  # (n_voxels - active_voxels_for_given_edge - invalid_edges, 1)

    valid_edges_within_extent = code__a_prod_edge * code__b_prod_edge * code__c_prod_edge  # * Valid in the sense that there are valid voxels around

    code__a_p = mapped_voxel_0[:, valid_edges_within_extent] == 0  # (n_voxels, n_voxels - active_voxels_for_given_edge - invalid_edges - edges_at_extent_border)
    code__b_p = mapped_voxel_1[:, valid_edges_within_extent] == 0  # (n_voxels, n_voxels - active_voxels_for_given_edge - invalid_edges - edges_at_extent_border)
    code__c_p = mapped_voxel_2[:, valid_edges_within_extent] == 0  # (n_voxels, n_voxels - active_voxels_for_given_edge - invalid_edges - edges_at_extent_border)

    if False:
        debug_code_p = code__a_p + code__b_p + code__c_p  # (n_voxels, n_voxels - active_voxels_for_given_edge - invalid_edges - edges_at_extent_border)
        # 15 and 17 does not have y
        
    # endregion

    # region Convert remaining compressed binary codes to ints
    indices_array = np.arange(code__a_p.shape[0]).reshape(-1, 1)
    x = (code__a_p * indices_array).T[code__a_p.T]
    y = (code__b_p * indices_array).T[code__b_p.T]
    z = (code__c_p * indices_array).T[code__c_p.T]
    # endregion

    if n < 4:
        normal = (code__a_p * voxel_normals[:, [0]]).T[code__a_p.T]
    elif n < 8:
        normal = (code__b_p * voxel_normals[:, [1]]).T[code__b_p.T]
    elif n < 12:
        normal = (code__c_p * voxel_normals[:, [2]]).T[code__c_p.T]
    else:
        raise ValueError("n must be smaller than 12")

    # flip triangle order if normal is negative
    indices = np.vstack((x[normal >= 0], y[normal >= 0], z[normal >= 0])).T
    flipped_indices = np.vstack((x[normal < 0], y[normal < 0], z[normal < 0])).T[:, [0, 2, 1]]
    indices = np.vstack((indices, flipped_indices))

    return indices
