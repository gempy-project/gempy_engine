
from numpy.core.numerictypes import ScalarType
from gempy_engine.core.data.exported_structs import ExportedFields, InterpOutput
import numpy
import numpy as np


def compute_dual_contouring():
    return


def find_intersection_on_edge(_xyz_8: numpy.ndarray, scalar_field: numpy.ndarray, scalar_at_sp: numpy.ndarray):
    # I have to do the topology analysis anyway because is the last octree
    scalar_8_ = scalar_field[:-7] #TODO: Generalize this
    scalar_8 = scalar_8_.reshape(-1, 8, 1)
    xyz_8 = _xyz_8.reshape(-1, 8, 3)

    # Compute distance of scalar field on the corners
    scalar_dx = scalar_8[:, :4] - scalar_8[:, 4:]
    scalar_d_y = scalar_8[:, [0, 1, 4, 5]] - scalar_8[:, [2, 3, 6, 7]]
    scalar_d_z = scalar_8[:, ::2] - scalar_8[:, 1::2]
    
    # Compute the weights
    weight_x =  (scalar_at_sp - scalar_8[:, 4:]) / scalar_dx 
    weight_y =  (scalar_at_sp -  scalar_8[:, [2, 3, 6, 7]]) / scalar_d_y 
    weight_z =  (scalar_at_sp - scalar_8[:, 1::2]) / scalar_d_z 
    
    # Calculate eucledian distance between the corners
    d_x = xyz_8[:, :4]           - xyz_8[:, 4:]
    d_y = xyz_8[:, [0, 1, 4, 5]] - xyz_8[:, [2, 3, 6, 7]]
    d_z = xyz_8[:, ::2]          - xyz_8[:, 1::2]

    # Compute the weighted distance
    intersect_dx = d_x[:, :, :] * weight_x[:,:, [0]]
    intersect_dy = d_y[:, :, :] * weight_y[:,:, [0]]
    intersect_dz = d_z[:, :, :] * weight_z[:,:, [0]]

    # Mask invalid edges
    valid_edge_x = np.logical_and(weight_x > 0, weight_x < 1)[:,:,0]
    valid_edge_y = np.logical_and(weight_y > 0, weight_y < 1)[:,:,0]
    valid_edge_z = np.logical_and(weight_z > 0, weight_z < 1)[:,:,0]

    #valid_edges = np.vstack([valid_edge_x, valid_edge_y,  valid_edge_z]).ravel()


    #Note(miguel) From this point on the arrays become sparse
    
    xyz_8_edges = np.hstack([xyz_8[:, 4:], xyz_8[:, [2, 3, 6, 7]], xyz_8[:, 1::2]])
    intersect_segment = np.hstack([intersect_dx, intersect_dy, intersect_dz])
    valid_edges = np.hstack([valid_edge_x, valid_edge_y, valid_edge_z])

    intersection_xyz = xyz_8_edges[valid_edges] + intersect_segment[valid_edges]


    # interection_x = xyz_8[:, 4:][valid_edge_x] + intersect_dx[valid_edge_x]
    # interection_y = xyz_8[:, [2, 3, 6, 7]][valid_edge_y] + intersect_dy[valid_edge_y]
    # interection_z = xyz_8[:, 1::2][valid_edge_z] + intersect_dz[valid_edge_z]

    # # Stack interections
    # interection_xyz = np.vstack([interection_x, interection_y, interection_z])
    return intersection_xyz, valid_edges




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
        x = numpy.array(x)
        return numpy.linalg.norm(numpy.matmul(self.A, x) - self.b)

    def eval_with_pos(self, x):
        """Evaluates the QEF at a position, returning the same format solve does."""
        return self.evaluate(x), x


    @staticmethod
    def make_3d(positions, normals):
        """Returns a QEF that measures the the error from a bunch of normals, each emanating
         from given positions"""
        A = numpy.array(normals)
        b = [v[0] * n[0] + v[1] * n[1] + v[2] * n[2] for v, n in zip(positions, normals)]
        fixed_values = [None] * A.shape[1]
        return QEF(A, b, fixed_values)



    def solve(self):
        """Finds the point that minimizes the error of this QEF,
        and returns a tuple of the error squared and the point itself"""
        result, residual, rank, s = numpy.linalg.lstsq(self.A, self.b)
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
