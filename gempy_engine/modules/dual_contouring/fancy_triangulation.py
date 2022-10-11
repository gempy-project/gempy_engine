# TODO: Make this methods private
import numpy as np


def get_left_right_array(voxel_select: np.ndarray, left_right: np.ndarray) -> np.ndarray:
    """
    Args:
        voxel_select (np.ndarray): Array of booleans with the shape (n_voxels, 8) indicating the
            selected voxels.
        left_right (np.ndarray): Array of booleans with the shape (n_voxels, 8) indicating the
            left-right position of the voxel.

    Returns:
        np.ndarray: Array with the shape (n_faces, 3) with the indices of the vertices of the
            triangulated faces.
    """

    # TODO: Check if x_foo_lvl2 is the same as debug_vals_3

    x_foo = np.zeros(8, dtype=bool)
    x_foo[4:] = True
    x_dir_lvl0 = np.repeat(x_foo[voxel_select], 8)
    x_foo_lvl2 = np.tile(x_foo, 8)
    binary_x = np.vstack((left_right[:, 0], x_dir_lvl0))
    y_foo = np.zeros(8, dtype=bool)
    y_foo[[2, 3, 6, 7]] = True
    y_dir_lvl0 = np.tile(np.repeat(y_foo, 4), 2)
    y_foo_lvl2 = np.repeat(np.tile(y_foo, 4), 2)
    y_dir_rep = np.repeat(y_foo[voxel_select], 8)
    binary_y = np.vstack((left_right[:, 1], y_dir_rep))
    z_foo = np.zeros(8, dtype=bool)
    z_foo[1::2] = True
    z_dir_lvl0 = np.tile(z_foo, 8)
    z_foo_lvl2 = np.repeat(z_foo, 8)
    z_dir_rep = np.repeat(z_foo[voxel_select], 8)
    binary_z = np.vstack((left_right[:, 2], z_dir_rep))
    bool_to_int_x = np.packbits(binary_x, axis=0, bitorder="little")
    bool_to_int_y = np.packbits(binary_y, axis=0, bitorder="little")
    bool_to_int_z = np.packbits(binary_z, axis=0, bitorder="little")
    stacked = np.vstack((bool_to_int_x, bool_to_int_y, bool_to_int_z)).T

    return stacked


def triangulate(left_right_array: np.ndarray, valid_edges: np.ndarray, valid_voxels: np.ndarray):
    voxel_binary_idx = left_right_array[valid_voxels]
    voxel_code = (voxel_binary_idx * np.array([1, 4, 16])).sum(1).reshape(-1, 1) 
    
    valid_edges_voxels = valid_edges[valid_voxels]
    n = 10  # This is the edge we chose
    
    first_edge = valid_edges_voxels[:, n]
    first_edge_idx = voxel_binary_idx[first_edge]

    # * Check valid edges
    outer_edge = 0
    inner_edge = 1
    valid_edges_0 = (first_edge_idx[:, outer_edge] != 3) * (first_edge_idx[:, inner_edge] != 0)  # * In the sense of not being on the side of the model
    first_edge_idx = first_edge_idx[valid_edges_0]

    # * These are the the codes that describe each vertex of the triangle
    edge_vector_0 = np.array([1, 0, 0])
    edge_vector_1 = np.array([0,-1, 0])
    edge_vector_2 = np.array([0, 0, 0])
    
    binary_idx_0: np.ndarray = first_edge_idx + edge_vector_0
    binary_idx_1: np.ndarray = first_edge_idx + edge_vector_1  # * -1 in y due to the specific edge
    binary_idx_2: np.ndarray = first_edge_idx + edge_vector_2 # * Z idx since the intersection is happening in the Z direction

    pack_directions_into_bits = np.array([1, 4, 16])
    
    compressed_binary_idx_0 = (binary_idx_0 * pack_directions_into_bits).sum(axis=1)
    compressed_binary_idx_1 = (binary_idx_1 * pack_directions_into_bits).sum(axis=1)
    compressed_binary_idx_2 = (binary_idx_2 * pack_directions_into_bits).sum(axis=1)

    mapped_voxel_0 = (voxel_code - compressed_binary_idx_0)
    mapped_voxel_1 = (voxel_code - compressed_binary_idx_1)
    mapped_voxel_2 = (voxel_code - compressed_binary_idx_2)
            
    # * Another way
    code__a_p = mapped_voxel_0 == 0
    code__b_p = mapped_voxel_1 == 0
    code__c_p = mapped_voxel_2 == 0

    indices_array = np.arange(code__a_p.shape[0]).reshape(-1, 1)
    x = (code__a_p * indices_array).T[code__a_p.T]
    y = (code__b_p * indices_array).T[code__b_p.T]
    z = (code__c_p * indices_array).T[code__c_p.T]
    indices = np.vstack((x, y, z)).T
    return indices


def triangulate_dep(left_right_array: np.ndarray, valid_edges: np.ndarray, valid_voxels: np.ndarray, voxel_code):
    voxel_binary_idx = left_right_array[valid_voxels]
    valid_edges_voxels = valid_edges[valid_voxels]

    n = 10  # This is the edge we chose

    # TODO: Trying to triangulate all the voxels for edge 10
    first_edge = valid_edges_voxels[:, n]
    first_edge_idx = voxel_binary_idx[first_edge]

    # because is a z edge
    direction = 2
    
     
    
    
    valid_edges_0 = (first_edge_idx[:, 0] != 3) * (first_edge_idx[:, 1] != 0) # * In the sense of not being on the side of the model
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
    valid_edges = code__a_prod_edge * code__b_prod_edge * code__c_prod_edge # * Valid in the sense that there are valid voxels around
        
    # This is the direction of voxels    
    code__a_prod = code__a[:, valid_edges].prod(axis=1)
    code__b_prod = code__b[:, valid_edges].prod(axis=1)
    code__c_prod = code__c[:, valid_edges].prod(axis=1)

    code__a_prod_ = code__a_prod == 0
    code__b_prod_ = code__b_prod == 0
    code__c_prod_ = code__c_prod == 0
    
    
    
    # * Another way
    code__a_p= code__a[:, valid_edges] == 0
    code__b_p= code__b[:, valid_edges] == 0
    code__c_p= code__c[:, valid_edges] == 0

    # code__a_p = code__a == 0
    # code__b_p = code__b == 0
    # code__c_p = code__c == 0

    code__p = code__a_p + code__b_p + code__c_p # ! This operation screws the triangle order and hence the normals
    
    
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
