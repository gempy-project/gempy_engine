from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.modules.dual_contouring._aux import _calc_mesh_normals, _correct_normals


def get_left_right_array(octree_list: list[OctreeLevel]):
    dtype = bool
    match BackendTensor.engine_backend:
        case BackendTensor.engine_backend.PYTORCH:
            dtype = BackendTensor.tfnp.bool
        case BackendTensor.engine_backend.numpy:
            dtype = bool
        case _:
            raise ValueError("Unsupported backend")

    # === Local function ===
    def _compute_voxel_binary_code(idx_from_root, dir_idx: int, left_right_all, voxel_select_all):

        # Calculate the voxels from root
        for active_voxels_per_lvl in voxel_select_all:  # * The first level is all True
            idx_from_root = BackendTensor.tfnp.repeat(idx_from_root[active_voxels_per_lvl], 8, axis=0)

        left_right_list = []
        voxel_select_op = list(voxel_select_all[1:])
        voxel_select_op.append(BackendTensor.tfnp.ones(
            left_right_all[-1].shape[0],
            dtype=dtype
        )
        )
        left_right_all = left_right_all[::-1]
        voxel_select_op = voxel_select_op[::-1]

        for e, left_right_per_lvl in enumerate(left_right_all):
            left_right_per_lvl_dir = left_right_per_lvl[:, dir_idx]
            for n_rep in range(e):
                inner = left_right_per_lvl_dir[voxel_select_op[e - n_rep]]
                left_right_per_lvl_dir = BackendTensor.tfnp.repeat(inner, 8, axis=0)
            left_right_list.append(left_right_per_lvl_dir)

        left_right_list.append(idx_from_root)
        binary_code = BackendTensor.tfnp.stack(left_right_list)
        return binary_code

    # === Local function ===

    if len(octree_list) == 1:
        # * Not only that, the current implementation only works with pure octree starting at [2,2,2]
        raise ValueError("Octree list must have more than one level")

    voxel_select_all = [octree_iter.grid_centers.octree_grid.active_cells for octree_iter in octree_list[1:]]
    left_right_all = [octree_iter.grid_centers.octree_grid.left_right for octree_iter in octree_list[1:]]

    dtype = bool
    match BackendTensor.engine_backend:
        case BackendTensor.engine_backend.PYTORCH:
            dtype = BackendTensor.tfnp.bool
        case BackendTensor.engine_backend.numpy:
            dtype = bool
        case _:
            raise ValueError("Unsupported backend")

    idx_root_x = BackendTensor.tfnp.zeros(8, dtype=dtype)
    idx_root_x[4:] = True
    binary_x = _compute_voxel_binary_code(idx_root_x, 0, left_right_all, voxel_select_all)

    idx_root_y = BackendTensor.tfnp.zeros(8, dtype=dtype)
    idx_root_y[[2, 3, 6, 7]] = True
    binary_y = _compute_voxel_binary_code(idx_root_y, 1, left_right_all, voxel_select_all)

    idx_root_z = BackendTensor.tfnp.zeros(8, dtype=dtype)
    idx_root_z[1::2] = True
    binary_z = _compute_voxel_binary_code(idx_root_z, 2, left_right_all, voxel_select_all)

    bool_to_int_x = BackendTensor.tfnp.packbits(binary_x, axis=0, bitorder="little")
    bool_to_int_y = BackendTensor.tfnp.packbits(binary_y, axis=0, bitorder="little")
    bool_to_int_z = BackendTensor.tfnp.packbits(binary_z, axis=0, bitorder="little")
    left_right_array = BackendTensor.tfnp.vstack([bool_to_int_x, bool_to_int_y, bool_to_int_z]).T

    _StaticTriangulationData.depth = 2
    foo = (left_right_array * _StaticTriangulationData.get_pack_directions_into_bits()).sum(axis=1)

    sorted_indices = BackendTensor.tfnp.argsort(foo)
    # left_right_array = left_right_array[sorted_indices]
    return left_right_array


class _StaticTriangulationData:
    depth: int

    @staticmethod
    def get_pack_directions_into_bits():
        base_number = 2 ** _StaticTriangulationData.depth
        # return BackendTensor.tfnp.array([1, base_number, base_number ** 2], dtype='int64')
        return BackendTensor.tfnp.array([base_number ** 2, base_number, 1], dtype='int64')

    @staticmethod
    def get_base_array(pack_directions_into_bits):
        return BackendTensor.tfnp.array([pack_directions_into_bits, pack_directions_into_bits * 2, pack_directions_into_bits * 3],
                                        dtype='int64')

    @staticmethod
    def get_base_number() -> int:
        return 2 ** _StaticTriangulationData.depth


def triangulate(left_right_array, valid_edges, tree_depth: int, voxel_normals, vertex):
    # * Variables
    # depending on depth
    _StaticTriangulationData.depth = tree_depth

    edge_vector_a = BackendTensor.tfnp.array([0, 0, 0, 0, -1, -1, 1, 1, -1, 1, -1, 1])
    edge_vector_b = BackendTensor.tfnp.array([-1, -1, -1, 1, 0, 0, 0, 0, -1, 1, -1, 1])
    edge_vector_c = BackendTensor.tfnp.array([-1, -1, 1, 1, -1, -1, 1, 1, 0, 0, 0, 0])

    # * Consts
    voxel_code = (left_right_array * _StaticTriangulationData.get_pack_directions_into_bits()).sum(1).reshape(-1, 1)
    # ----------

    indices = []
    normals = []
    all = [
            0,
            3,
            4,
            7,
            8,
            11
    ]

    for n in all:
        # TODO: Make sure that changing the edge_vector we do not change
        left_right_array_active_edge = left_right_array[valid_edges[:, n]]
        indices_patch, normals_patch = compute_triangles_for_edge(
            edge_vector_a=edge_vector_a[n],
            edge_vector_b=edge_vector_b[n],
            edge_vector_c=edge_vector_c[n],
            left_right_array_active_edge=left_right_array_active_edge,
            voxel_code=voxel_code,
            voxel_normals=voxel_normals,
            n=n
        )


        indices.append(indices_patch)
        normals.append(normals_patch)

    indices = BackendTensor.t.concatenate(indices, axis=0)
    normals_from_edges = BackendTensor.t.concatenate(normals, axis=0)
    # norms_from_mesh = _calc_mesh_normals(vertex, indices)
    indices_corrected, _ = _correct_normals(vertex, indices, normals_from_edges)

    return indices_corrected


def compute_triangles_for_edge(edge_vector_a, edge_vector_b, edge_vector_c,
                               left_right_array_active_edge, voxel_code, voxel_normals, n):
    """
    Important concepts to understand this triangulation:
    - left_right_array (n_voxels, 3-directions) contains a unique number per direction describing if it is left (even) or right (odd) and the voxel level
    - left_right_array_active_edge (n_voxels - active_voxels_for_given_edge, 3-directions): Subset of left_right for voxels with selected edge active
    - valid_edges (n_voxels - active_voxels_for_given_edge, 1): bool
    - voxel_code (n_voxels, 1): All the compressed codes of the voxels at the leaves
    """

    match BackendTensor.engine_backend:
        case BackendTensor.engine_backend.PYTORCH:
            dtype = BackendTensor.tfnp.bool
        case BackendTensor.engine_backend.numpy:
            dtype = bool
        case _:
            raise ValueError("Unsupported backend")

    # Step 1: Filter edges with valid neighbors
    left_right_array_active_edge = _filter_edges_with_neighbors(
        edge_vector_a, edge_vector_b, edge_vector_c,
        left_right_array_active_edge, dtype
    )

    # Step 2: Compress binary indices
    compressed_idx_0, compressed_idx_1, compressed_idx_2 = _compress_binary_indices(
        left_right_array_active_edge,
        edge_vector_a, edge_vector_b, edge_vector_c
    )

    # Step 3: Map voxels and filter by extent
    code__a_p, code__b_p, code__c_p = _map_and_filter_voxels(
        voxel_code,
        compressed_idx_0, compressed_idx_1, compressed_idx_2
    )

    # Step 4: Convert boolean masks to indices
    x, y, z = _convert_masks_to_indices(code__a_p, code__b_p, code__c_p)

    # Step 5: Calculate normals and order triangles
    normals = _calculate_normals_and_order_triangles(
        x, y, z, voxel_normals, n
    )

    indices = BackendTensor.tfnp.stack([x, y, z], axis=1)
    return indices, normals


def _filter_edges_with_neighbors(edge_vector_a, edge_vector_b, edge_vector_c,
                                 left_right_array_active_edge, dtype):
    """Remove edges that don't have voxels next to them."""

    def check_voxels_exist_next_to_edge(coord_col, edge_vector, _left_right_array_active_edge):
        match edge_vector:
            case 0:
                _valid_edges = BackendTensor.tfnp.ones(_left_right_array_active_edge.shape[0], dtype=dtype)
            case 1:
                _valid_edges = _left_right_array_active_edge[:, coord_col] != _StaticTriangulationData.get_base_number() - 1
            case -1:
                _valid_edges = _left_right_array_active_edge[:, coord_col] != 0
            case _:
                raise ValueError("edge_vector must be -1, 0 or 1")
        return _valid_edges

    valid_edges_x = check_voxels_exist_next_to_edge(0, edge_vector_a, left_right_array_active_edge)
    valid_edges_y = check_voxels_exist_next_to_edge(1, edge_vector_b, left_right_array_active_edge)
    valid_edges_z = check_voxels_exist_next_to_edge(2, edge_vector_c, left_right_array_active_edge)

    valid_edges_with_neighbour_voxels = valid_edges_x * valid_edges_y * valid_edges_z
    return left_right_array_active_edge[valid_edges_with_neighbour_voxels]


def _compress_binary_indices(left_right_array_active_edge, edge_vector_a, edge_vector_b, edge_vector_c):
    """Compress voxel codes per direction."""
    edge_vector_0 = BackendTensor.tfnp.array([edge_vector_a, 0, 0])
    edge_vector_1 = BackendTensor.tfnp.array([0, edge_vector_b, 0])
    edge_vector_2 = BackendTensor.tfnp.array([0, 0, edge_vector_c])

    binary_idx_0 = left_right_array_active_edge + edge_vector_0
    binary_idx_1 = left_right_array_active_edge + edge_vector_1
    binary_idx_2 = left_right_array_active_edge + edge_vector_2

    pack_directions = _StaticTriangulationData.get_pack_directions_into_bits()
    compressed_binary_idx_0 = (binary_idx_0 * pack_directions).sum(axis=1)
    compressed_binary_idx_1 = (binary_idx_1 * pack_directions).sum(axis=1)
    compressed_binary_idx_2 = (binary_idx_2 * pack_directions).sum(axis=1)

    return compressed_binary_idx_0, compressed_binary_idx_1, compressed_binary_idx_2


def _map_and_filter_voxels_(voxel_code, compressed_idx_0, compressed_idx_1, compressed_idx_2):
    """Map compressed binary codes to all leaf codes and filter by extent."""
    # Map remaining compressed binary code to all the binary codes at leaves
    mapped_voxel_0 = (voxel_code - compressed_idx_0)
    mapped_voxel_1 = (voxel_code - compressed_idx_1)
    mapped_voxel_2 = (voxel_code - compressed_idx_2)

    # Find and remove edges at the border of the extent
    code__a_prod_edge = ~BackendTensor.tfnp.all(mapped_voxel_0, axis=0)
    code__b_prod_edge = ~BackendTensor.tfnp.all(mapped_voxel_1, axis=0)
    code__c_prod_edge = ~BackendTensor.tfnp.all(mapped_voxel_2, axis=0)

    valid_edges_within_extent = code__a_prod_edge * code__b_prod_edge * code__c_prod_edge

    code__a_p = BackendTensor.tfnp.array(mapped_voxel_0[:, valid_edges_within_extent] == 0)
    code__b_p = BackendTensor.tfnp.array(mapped_voxel_1[:, valid_edges_within_extent] == 0)
    code__c_p = BackendTensor.tfnp.array(mapped_voxel_2[:, valid_edges_within_extent] == 0)

    return code__a_p, code__b_p, code__c_p

def _map_and_filter_voxels(voxel_code, compressed_idx_0, compressed_idx_1, compressed_idx_2):
    """Map compressed binary codes to all leaf codes and filter by extent (optimized)."""

    # Instead of checking .all() on large arrays, we can use .any() on equality checks
    # which is more efficient because:
    # 1. We're looking for matches (== 0) rather than all non-matches (!= 0)
    # 2. .any() can short-circuit on the first True

    # Find which voxels match each compressed index (these ARE the valid edges)
    code__a_prod_edge = BackendTensor.tfnp.any(voxel_code == compressed_idx_0, axis=0)
    code__b_prod_edge = BackendTensor.tfnp.any(voxel_code == compressed_idx_1, axis=0)
    code__c_prod_edge = BackendTensor.tfnp.any(voxel_code == compressed_idx_2, axis=0)

    # Valid edges are those that have all three coordinates matching some voxel
    valid_edges_within_extent = code__a_prod_edge & code__b_prod_edge & code__c_prod_edge

    # Now only compute the expensive equality checks for valid edges
    code__a_p = (voxel_code == compressed_idx_0[ valid_edges_within_extent])
    code__b_p = (voxel_code == compressed_idx_1[ valid_edges_within_extent])
    code__c_p = (voxel_code == compressed_idx_2[ valid_edges_within_extent])

    return code__a_p, code__b_p, code__c_p


def _convert_masks_to_indices(code__a_p, code__b_p, code__c_p):
    """Convert boolean masks to integer indices."""
    indices_array = BackendTensor.tfnp.arange(code__a_p.shape[0]).reshape(-1, 1)
    x = (code__a_p * indices_array).T[code__a_p.T]
    y = (code__b_p * indices_array).T[code__b_p.T]
    z = (code__c_p * indices_array).T[code__c_p.T]
    return x, y, z


def _calculate_normals_and_order_triangles(x, y, z, voxel_normals, n):
    """Calculate normals and order triangles based on normal direction."""
    # Get normals based on edge type
    if n in [8, 11]:
        normal = voxel_normals[z, :, :].sum(1)
    elif n in [0, 3]:
        normal = voxel_normals[x, :, :].sum(1)
    elif n in [4, 7]:
        normal = voxel_normals[y, :, :].sum(1)
    else:
        normal = BackendTensor.tfnp.ones(x.shape[0], dtype=BackendTensor.tfnp.float32)
    return normal