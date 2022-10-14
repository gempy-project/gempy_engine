# TODO: Make this methods private
import numpy as np

from gempy_engine.core.data.octree_level import OctreeLevel


def get_left_right_array(octree_list: list[OctreeLevel]) -> np.ndarray:
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
                left_right_per_lvl_dir = np.repeat(left_right_per_lvl_dir[voxel_select_op[e-n_rep]], 8)  # ? Is it always e?
            left_right_list.append(left_right_per_lvl_dir)
        
        left_right_list.append(idx_from_root)
        binary_code = np.vstack(left_right_list)
        f = binary_code.T
        return binary_code

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
    stacked = np.vstack((bool_to_int_x, bool_to_int_y, bool_to_int_z)).T

    return stacked


class _StaticTriangulationData:
    depth: int

    @staticmethod
    def get_pack_directions_into_bits() -> np.ndarray:
        base_number = 2 ** _StaticTriangulationData.depth
        return np.array([1, base_number, base_number ** 2], dtype=np.int64)

    @staticmethod
    def get_base_number() -> int:
        return 2 ** _StaticTriangulationData.depth


def triangulate(left_right_array: np.ndarray, valid_edges: np.ndarray, tree_depth: int):
    # * Variables
    # depending on depth
    _StaticTriangulationData.depth = tree_depth  

    # depending on the edge
    edge_vector_a = np.array([0, 0, 0, 0, -1, -1, 1, 1, -1, -1, 1, 1])
    edge_vector_b = np.array([-1, -1, 1, 1, 0, 0, 0, 0, -1, 1, -1, 1])
    edge_vector_c = np.array([-1, 1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0])

    # * Consts
    voxel_code = (left_right_array * _StaticTriangulationData.get_pack_directions_into_bits()).sum(1).reshape(-1, 1)
    # ----------

    indices = []

    all = [0, 3, 4, 7, 8, 11]
    for n in all:
        left_right_array_active_edge = left_right_array[valid_edges[:, n]]
        _ = compute_triangles_for_edge(
            edge_vector_a=edge_vector_a[n],
            edge_vector_b=edge_vector_b[n],
            edge_vector_c=edge_vector_c[n],
            left_right_array_active_edge=left_right_array_active_edge,
            voxel_code=voxel_code)
        indices.append(_)

    return indices


def compute_triangles_for_edge(edge_vector_a, edge_vector_b, edge_vector_c, left_right_array_active_edge, voxel_code):
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
    # endregion. 

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
    code__a_prod_edge = mapped_voxel_0.prod(axis=0) == 0  # (n_voxels - active_voxels_for_given_edge - invalid_edges, 1)
    code__b_prod_edge = mapped_voxel_1.prod(axis=0) == 0  # (n_voxels - active_voxels_for_given_edge - invalid_edges, 1)
    code__c_prod_edge = mapped_voxel_2.prod(axis=0) == 0  # (n_voxels - active_voxels_for_given_edge - invalid_edges, 1)

    valid_edges_within_extent = code__a_prod_edge * code__b_prod_edge * code__c_prod_edge  # * Valid in the sense that there are valid voxels around

    code__a_p = mapped_voxel_0[:, valid_edges_within_extent] == 0  # (n_voxels, n_voxels - active_voxels_for_given_edge - invalid_edges - edges_at_extent_border)
    code__b_p = mapped_voxel_1[:, valid_edges_within_extent] == 0  # (n_voxels, n_voxels - active_voxels_for_given_edge - invalid_edges - edges_at_extent_border)
    code__c_p = mapped_voxel_2[:, valid_edges_within_extent] == 0  # (n_voxels, n_voxels - active_voxels_for_given_edge - invalid_edges - edges_at_extent_border)
    # endregion
    
    # region Convert remaining compressed binary codes to ints
    indices_array = np.arange(code__a_p.shape[0]).reshape(-1, 1)
    x = (code__a_p * indices_array).T[code__a_p.T]
    y = (code__b_p * indices_array).T[code__b_p.T]
    z = (code__c_p * indices_array).T[code__c_p.T]
    # endregion
    indices = np.vstack((x, y, z)).T
    return indices


def triangulate_example(left_right_array: np.ndarray, valid_edges: np.ndarray, valid_voxels: np.ndarray):
    voxel_binary_idx = left_right_array[valid_voxels]
    valid_edges_voxels = valid_edges[valid_voxels]

    # region X edges
    n = 10  # This is the edge we chose
    first_edge = valid_edges_voxels[:, n]
    first_edge_idx = voxel_binary_idx[first_edge]
    idx_2 = first_edge_idx[9, 2]  # * Z idx since the intersection is happening in the Z direction
    idx_0 = first_edge_idx[9, 0] + 1  # * +1 in x due to the specific edge
    idx_1 = first_edge_idx[9, 1] - 1  # * -1 in y due to the specific edge

    # * Compose valid voxeld idx for each voxel that compose the triangle
    idx_a = np.array([2, 1, 1])
    idx_b = np.array([3, 1, 1])
    idx_c = np.array([2, 0, 1])

    # * Find the arg of the 3 vectors of voxels idx to compose the triangle 
    i_0 = np.where((voxel_binary_idx == idx_a).sum(axis=1) == 3)
    i_1 = np.where((voxel_binary_idx == idx_b).sum(axis=1) == 3)
    i_2 = np.where((voxel_binary_idx == idx_c).sum(axis=1) == 3)

    return first_edge_idx, i_0, i_1, i_2, n, voxel_binary_idx


def triangulate_dep(left_right_array: np.ndarray, valid_edges: np.ndarray, valid_voxels: np.ndarray, voxel_code):
    voxel_binary_idx = left_right_array[valid_voxels]
    valid_edges_voxels = valid_edges[valid_voxels]

    n = 10  # This is the edge we chose

    # TODO: Trying to triangulate all the voxels for edge 10
    first_edge = valid_edges_voxels[:, n]
    first_edge_idx = voxel_binary_idx[first_edge]

    # because is a z edge
    direction = 2

    valid_edges_0 = (first_edge_idx[:, 0] != 3) * (first_edge_idx[:, 1] != 0)  # * In the sense of not being on the side of the model
    first_edge_idx = first_edge_idx[valid_edges_0]

    # Add +1 to first column
    aux_x = first_edge_idx + np.array([1, 0, 0])
    aux_y = first_edge_idx + np.array([0, -1, 0])

    # * These are the the codes that describe each vertex of the triangle
    binary_idx_0: np.ndarray = aux_x
    binary_idx_1: np.ndarray = aux_y  # * -1 in y due to the specific edge
    binary_idx_2: np.ndarray = first_edge_idx  # * Z idx since the intersection is happening in the Z direction

    # XYZ to code to full compressed
    code_0 = (binary_idx_0 * np.array([1, 4, 16])).sum(axis=1)
    code__a = (voxel_code[valid_voxels] - code_0)

    code_1 = (binary_idx_1 * np.array([1, 4, 16])).sum(axis=1)
    code__b = (voxel_code[valid_voxels] - code_1)

    code_2 = (binary_idx_2 * np.array([1, 4, 16])).sum(axis=1)
    code__c = (voxel_code[valid_voxels] - code_2)

    # This is the direction of edges
    code__a_prod_edge = code__a.prod(axis=0) == 0
    code__b_prod_edge = code__b.prod(axis=0) == 0
    code__c_prod_edge = code__c.prod(axis=0) == 0

    # TODO: This is unnecessary anymore
    valid_edges = code__a_prod_edge * code__b_prod_edge * code__c_prod_edge  # * Valid in the sense that there are valid voxels around

    # This is the direction of voxels    
    code__a_prod = code__a[:, valid_edges].prod(axis=1)
    code__b_prod = code__b[:, valid_edges].prod(axis=1)
    code__c_prod = code__c[:, valid_edges].prod(axis=1)

    code__a_prod_ = code__a_prod == 0
    code__b_prod_ = code__b_prod == 0
    code__c_prod_ = code__c_prod == 0

    # * Another way
    code__a_p = code__a[:, valid_edges] == 0
    code__b_p = code__b[:, valid_edges] == 0
    code__c_p = code__c[:, valid_edges] == 0

    # code__a_p = code__a == 0
    # code__b_p = code__b == 0
    # code__c_p = code__c == 0

    code__p = code__a_p + code__b_p + code__c_p  # ! This operation screws the triangle order and hence the normals

    x = (code__a_p * np.arange(28).reshape(-1, 1)).T[code__a_p.T]
    y = (code__b_p * np.arange(28).reshape(-1, 1)).T[code__b_p.T]
    z = (code__c_p * np.arange(28).reshape(-1, 1)).T[code__c_p.T]
    foobar = np.vstack((x, y, z)).T

    foo = (code__p * np.arange(28).reshape(-1, 1)).T[code__p.T]
    bar = foo.reshape(-1, 3)
    # Triangle 6 -> [13,16,22] is wrong -> individual code is a(201) b(301) c(1,3,0)

    # ! This sort it
    i_0_a = np.argwhere(code__a_prod_)
    i_1_a = np.argwhere(code__b_prod == 0)
    i_2_a = np.argwhere(code__c_prod == 0)

    indices = np.hstack((i_0_a, i_1_a, i_2_a))

    # TODO: Mapping Binary codes to vertex (which are voxels on the leaf)
    dist = (voxel_binary_idx - binary_idx_0[:, None]).sum(-1)

    # idx_2 = first_edge_idx[9, 2]  # * Z idx since the intersection is happening in the Z direction
    # idx_0 = first_edge_idx[9, 0] + 1  # * +1 in x due to the specific edge
    # idx_1 = first_edge_idx[9, 1] - 1  # * -1 in y due to the specific edge

    # * Compose valid voxeld idx for each voxel that compose the triangle
    idx_a = np.array([2, 1, 1])
    idx_b = np.array([3, 1, 1])
    idx_c = np.array([2, 0, 1])

    # * Find the arg of the 3 vectors of voxels idx to compose the triangle 
    # i_0 = np.where((voxel_binary_idx == idx_a).sum(axis=1) == 3)
    # i_1 = np.where((voxel_binary_idx == idx_b).sum(axis=1) == 3)
    # i_2 = np.where((voxel_binary_idx == idx_c).sum(axis=1) == 3)

    # This triangle is [17,13,15]
    # TODO: Generalize for all edges
    return foobar