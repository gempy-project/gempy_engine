from typing import Tuple

from gempy_engine.config import AvailableBackends
from ...core.backend_tensor import BackendTensor
from ...core.data.dual_contouring_data import DualContouringData


def find_intersection_on_edge(_xyz_corners, scalar_field_on_corners,
                              scalar_at_sp, masking=None) -> Tuple:
    """This function finds all the intersections for multiple layers per series
    
    - The shape of valid edges is n_surfaces * xyz_corners. Where xyz_corners is 8 * the octree leaf
    - The shape of intersection_xyz really depends on the number of intersections per voxel
    
    
    """
    scalar_8_ = scalar_field_on_corners
    scalar_8 = scalar_8_.reshape((1, -1, 8))
    xyz_8 = _xyz_corners.reshape((-1, 8, 3))

    if masking is not None:
        ma_8 = masking
        xyz_8 = xyz_8[ma_8]
        scalar_8 = scalar_8[:, ma_8]

    scalar_at_sp = scalar_at_sp.reshape((-1, 1, 1))

    n_isosurface = scalar_at_sp.shape[0]
    xyz_8 = BackendTensor.tfnp.tile(xyz_8, (n_isosurface, 1, 1))  # TODO: Generalize

    # Compute distance of scalar field on the corners
    scalar_dx = scalar_8[:, :, :4] - scalar_8[:, :, 4:]
    scalar_d_y = scalar_8[:, :, [0, 1, 4, 5]] - scalar_8[:, :, [2, 3, 6, 7]]
    scalar_d_z = scalar_8[:, :, ::2] - scalar_8[:, :, 1::2]

    # Add a tiny value to avoid division by zero
    scalar_dx += 1e-10
    scalar_d_y += 1e-10
    scalar_d_z += 1e-10

    # Compute the weights
    weight_x = ((scalar_at_sp - scalar_8[:, :, 4:]) / scalar_dx).reshape(-1, 4, 1)
    weight_y = ((scalar_at_sp - scalar_8[:, :, [2, 3, 6, 7]]) / scalar_d_y).reshape(-1, 4, 1)
    weight_z = ((scalar_at_sp - scalar_8[:, :, 1::2]) / scalar_d_z).reshape(-1, 4, 1)

    # Calculate eucledian distance between the corners
    d_x = xyz_8[:, :4] - xyz_8[:, 4:]
    d_y = xyz_8[:, [0, 1, 4, 5]] - xyz_8[:, [2, 3, 6, 7]]
    d_z = xyz_8[:, ::2] - xyz_8[:, 1::2]

    # Compute the weighted distance
    intersect_dx = d_x[:, :, :] * weight_x[:, :, :]
    intersect_dy = d_y[:, :, :] * weight_y[:, :, :]
    intersect_dz = d_z[:, :, :] * weight_z[:, :, :]

    # Mask invalid edges
    valid_edge_x = BackendTensor.tfnp.logical_and(weight_x > -0.01, weight_x < 1.01)
    valid_edge_y = BackendTensor.tfnp.logical_and(weight_y > -0.01, weight_y < 1.01)
    valid_edge_z = BackendTensor.tfnp.logical_and(weight_z > -0.01, weight_z < 1.01)

    # * Note(miguel) From this point on the arrays become sparse
    xyz_8_edges = BackendTensor.tfnp.hstack([xyz_8[:, 4:], xyz_8[:, [2, 3, 6, 7]], xyz_8[:, 1::2]])
    intersect_segment = BackendTensor.tfnp.hstack([intersect_dx, intersect_dy, intersect_dz])
    valid_edges = BackendTensor.tfnp.hstack([valid_edge_x, valid_edge_y, valid_edge_z])[:, :, 0]
    valid_edges = valid_edges > 0

    intersection_xyz = xyz_8_edges[valid_edges] + intersect_segment[valid_edges]

    return intersection_xyz, valid_edges


def generate_dual_contouring_vertices(dc_data_per_stack: DualContouringData, slice_surface: slice, debug: bool = False):
    # @off
    n_edges = dc_data_per_stack.n_edges
    valid_edges = dc_data_per_stack.valid_edges
    valid_voxels = dc_data_per_stack.valid_voxels
    xyz_on_edge = dc_data_per_stack.xyz_on_edge[slice_surface]
    gradients = dc_data_per_stack.gradients[slice_surface]
    # @on

    # * Coordinates for all posible edges (12) and 3 dummy edges_normals in the center
    edges_xyz = BackendTensor.tfnp.zeros((n_edges, 15, 3), dtype=BackendTensor.dtype_obj)
    valid_edges = valid_edges > 0
    edges_xyz[:, :12][valid_edges] = xyz_on_edge

    # Normals
    edges_normals = BackendTensor.tfnp.zeros((n_edges, 15, 3), dtype=BackendTensor.dtype_obj)
    edges_normals[:, :12][valid_edges] = gradients

    if OLD_METHOD := False:
        # ! Moureze model does not seems to work with the new method
        # ! This branch is all nans at least with ch1_1 model
        bias_xyz = BackendTensor.tfnp.copy(edges_xyz[:, :12])
        isclose = BackendTensor.tfnp.isclose(bias_xyz, 0)
        bias_xyz[isclose] = BackendTensor.tfnp.nan  # zero values to nans
        mass_points = BackendTensor.tfnp.nanmean(bias_xyz, axis=1)  # Mean ignoring nans
    else:  # ? This is actually doing something
        bias_xyz = BackendTensor.tfnp.copy(edges_xyz[:, :12])
        if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
            # PyTorch doesn't have masked arrays, so we'll use a different approach
            mask = bias_xyz == 0
            # Replace zeros with NaN for mean calculation
            bias_xyz_masked = BackendTensor.tfnp.where(mask, float('nan'), bias_xyz)
            mass_points = BackendTensor.tfnp.nanmean(bias_xyz_masked, axis=1)
        else:
            # NumPy approach with masked arrays
            bias_xyz = BackendTensor.tfnp.to_numpy(bias_xyz)
            import numpy as np
            mask = bias_xyz == 0
            masked_arr = np.ma.masked_array(bias_xyz, mask)
            mass_points = masked_arr.mean(axis=1)
            mass_points = BackendTensor.tfnp.array(mass_points)

    edges_xyz[:, 12] = mass_points
    edges_xyz[:, 13] = mass_points
    edges_xyz[:, 14] = mass_points

    BIAS_STRENGTH = 1

    bias_x = BackendTensor.tfnp.array([BIAS_STRENGTH, 0, 0], dtype=BackendTensor.dtype_obj)
    bias_y = BackendTensor.tfnp.array([0, BIAS_STRENGTH, 0], dtype=BackendTensor.dtype_obj)
    bias_z = BackendTensor.tfnp.array([0, 0, BIAS_STRENGTH], dtype=BackendTensor.dtype_obj)

    edges_normals[:, 12] = bias_x
    edges_normals[:, 13] = bias_y
    edges_normals[:, 14] = bias_z

    # Remove unused voxels
    edges_xyz = edges_xyz[valid_voxels]
    edges_normals = edges_normals[valid_voxels]

    # Compute LSTSQS in all voxels at the same time
    A = edges_normals
    b = (A * edges_xyz).sum(axis=2)

    if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
        transpose_shape = (2, 1, 0)  # For PyTorch: (batch, dim2, dim1)
    else:
        transpose_shape = (0, 2, 1)  # For NumPy: (batch, dim2, dim1)

    term1 = BackendTensor.tfnp.einsum("ijk, ilj->ikl", A, BackendTensor.tfnp.transpose(A, transpose_shape))
    term2 = BackendTensor.tfnp.linalg.inv(term1)
    term3 = BackendTensor.tfnp.einsum("ijk,ik->ij", BackendTensor.tfnp.transpose(A, transpose_shape), b)
    vertices = BackendTensor.tfnp.einsum("ijk, ij->ik", term2, term3)

    if debug:
        dc_data_per_stack.bias_center_mass = edges_xyz[:, 12:].reshape(-1, 3)
        dc_data_per_stack.bias_normals = edges_normals[:, 12:].reshape(-1, 3)

    return vertices


# NOTE(miguel, July 2021): This class is only used for sanity check
class QEF:
    """Represents and solves the quadratic error function"""

    def __init__(self, A, b, fixed_values):
        self.A = A
        self.b = b
        self.fixed_values = fixed_values

    def evaluate(self, x):
        """Evaluates the function at a given point.
        This is what the solve method is trying to minimize.
        NB: Doesn't work with fixed axes."""
        x = BackendTensor.tfnp.array(x)
        return BackendTensor.tfnp.linalg.norm(BackendTensor.tfnp.matmul(self.A, x) - self.b)

    def eval_with_pos(self, x):
        """Evaluates the QEF at a position, returning the same format solve does."""
        return self.evaluate(x), x

    @staticmethod
    def make_3d(positions, normals):
        """Returns a QEF that measures the the error from a bunch of normals, each emanating
         from given positions"""
        A = BackendTensor.tfnp.array(normals)
        b = [v[0] * n[0] + v[1] * n[1] + v[2] * n[2] for v, n in zip(positions, normals)]
        fixed_values = [None] * A.shape[1]
        return QEF(A, b, fixed_values)

    def solve(self):
        """Finds the point that minimizes the error of this QEF,
        and returns a tuple of the error squared and the point itself"""
        result, residual, rank, s = BackendTensor.tfnp.linalg.lstsq(self.A, self.b)
        if len(residual) == 0:
            residual = self.evaluate(result)
        else:
            residual = residual[0]
        # Result only contains the solution for the unfixed axis,
        # we need to add back all the ones we previously fixed.
        position = []
        i = 0
        for value in self.fixed_values:
            if value is None:
                position.append(result[i])
                i += 1
            else:
                position.append(value)
        return residual, position
