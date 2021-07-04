
from numpy.core.numerictypes import ScalarType
from gempy_engine.core.data.exported_structs import ExportedFields, InterpOutput
import numpy
import numpy as np


def compute_dual_contouring():
    return


def find_intersection_on_edge(_xyz_8: numpy.ndarray, scalar_field: numpy.ndarray, scalar_at_sp: numpy.ndarray):
    # I have to do the topology analysis anyway because is the last octree
    scalar_8_ = scalar_field[:-7]
    scalar_8 = scalar_8_.reshape(-1, 8, 1)

    # # TODO: scalar_8_left - scalar_field_at_corners. Find if positive
    # shift_x_l = scalar_8[:, :4]           - scalar_at_sp.reshape(1, 1, -1)
    # shift_y_l = scalar_8[:, [0, 1, 4, 5]] - scalar_at_sp.reshape(1, 1, -1)
    # shift_z_l = scalar_8[:, ::2]          - scalar_at_sp.reshape(1, 1, -1)

    # # TODO: scalar_8_right - scalar_field_at_corners. Find if Positive
    # shift_x_r = scalar_8[:, 4:]           - scalar_at_sp.reshape(1, 1, -1)
    # shift_y_r = scalar_8[:, [2, 3, 6, 7]] - scalar_at_sp.reshape(1, 1, -1)
    # shift_z_r = scalar_8[:, 1::2]         - scalar_at_sp.reshape(1, 1, -1)

    # Find if values are positive
    # shift_x_l_pos = shift_x_l > 0
    # shift_y_l_pos = shift_y_l > 0
    # shift_z_l_pos = shift_z_l > 0
    # shift_x_r_pos = shift_x_r > 0
    # shift_y_r_pos = shift_y_r > 0
    # shift_z_r_pos = shift_z_r > 0

    # Intersection
    # selected_edges_x = np.logical_xor(shift_x_l_pos, shift_x_r_pos)
    # selected_edges_y = np.logical_xor(shift_y_l_pos, shift_y_r_pos)
    # selected_edges_z = np.logical_xor(shift_z_l_pos, shift_z_r_pos)

    # [X] TODO: Compute distance of scalar field on the corners
    d_x = scalar_8[:, :4] - scalar_8[:, 4:]
    d_y = scalar_8[:, [0, 1, 4, 5]] - scalar_8[:, [2, 3, 6, 7]]
    d_z = scalar_8[:, ::2] - scalar_8[:, 1::2]
    

    # TODO: 

    weight_x =  (scalar_at_sp - scalar_8[:, 4:]) / d_x 
    weight_y =  (scalar_at_sp -  scalar_8[:, [2, 3, 6, 7]]) / d_y 
    weight_z =  (scalar_at_sp - scalar_8[:, 1::2]) / d_z 
    
    # weight_x = values_outside_0_1_to_zero(weight_x)
    # weight_y = values_outside_0_1_to_zero(weight_y)
    # weight_z = values_outside_0_1_to_zero(weight_z)
    
    #valid_corners = select_valid_corners(weight_x, weight_y, weight_z)

    # Need to end up with a XYZ list per voxel
    xyz_8 = _xyz_8.reshape(-1, 8, 3)
    
    # TODO: dx, dy, dz are constant in a regular grid.

    d_xx = xyz_8[:, :4]           - xyz_8[:, 4:]
    d_yy = xyz_8[:, [0, 1, 4, 5]] - xyz_8[:, [2, 3, 6, 7]]
    d_zz = xyz_8[:, ::2]          - xyz_8[:, 1::2]

    dxxx = d_xx[:, :, :] * weight_x[:,:, [0]]
    dyyy = d_yy[:, :, :] * weight_y[:,:, [0]]
    dzzz = d_zz[:, :, :] * weight_z[:,:, [0]]

    # xyz_8[:, 4:]           += dxxx
    # xyz_8[:, [2 ,3 ,6 ,7]] += dyyy
    # xyz_8[:, 1::2]          += dzzz
    
    foox =  np.logical_and(weight_x > 0, weight_x < 1)[:,:,0]
    fooy = np.logical_and(weight_y > 0, weight_y < 1)[:,:,0]
    fooz = np.logical_and(weight_z > 0, weight_z < 1)[:,:,0]


 
#     xyz_8[:, 4:] += dxxx
#     xyz_8[:, [2, 3, 6, 7]] += dyyy
#     xyz_8[:, 1::2][fooz] += dzzz[fooz]


#     interection_xyz[:][foox] += xyz_8[:, 4:][foox] + dxxx[foox]
#     interection_xyz[: ][fooy] += xyz_8[:, [2 ,3 ,6 ,7]][fooy] + dyyy[fooy]
#     interection_xyz[: ][fooz] += xyz_8[:, 1::2][fooz] + dzzz[fooz]
# #  #   interection_xyz[foo] = xyz_8[:, 1::2][foo] + dzzz[foo]


# #     interection_xyz = interection_xyz[valid_corners[:,:,0]]
#     return interection_xyz.reshape(-1, 3)
    
    #interection_x = np.zeros_like(xyz_8[:, 4:])
    interection_x = xyz_8[:, 4:][foox] + dxxx[foox]
  #  interection_y = np.zeros_like(xyz_8[:, [2, 3, 6, 7]])
    interection_y = xyz_8[:, [2, 3, 6, 7]][fooy] + dyyy[fooy]
  #  interection_z = np.zeros_like(xyz_8[:, 1::2])
    interection_z = xyz_8[:, 1::2][fooz] + dzzz[fooz]

    # Stack interections
    interection_xyz = np.vstack([interection_x, interection_y, interection_z])
    return interection_xyz.reshape(-1, 3)


def values_outside_0_1_to_zero(values: numpy.ndarray):
    """Set all values outside 0 and 1 to zero"""
    values[values < 0] = 0
    values[values > 1] = 0
    return values

def select_valid_corners(weight_x: np.array, weight_y:np.ndarray, weight_z:np.ndarray):
    """Return  boolean intersection for the values of weight_x, weight_y and 
    weight_z that are True if value is between 0 and 1"""
    # return np.logical_or(#np.logical_and(weight_x > 0, weight_x < 1),
    #                      np.logical_and(weight_y > 0, weight_y < 1),
    #                       #np.logical_and(weight_z > 0, weight_z < 1)
    #                       )

    return np.logical_and(weight_x > 0, weight_x < 1)
   
    
    
    


def solve_qef_3d(x, y, z, positions, normals):
    # The error term we are trying to minimize is sum( dot(y-v[i], n[i]) ^ 2)
    # This should be minimized over the unit square with top left point (x, y)

    # In other words, minimize || A * x - b || ^2 where A and b are a matrix and vector
    # derived from v and n
    # The heavy lifting is done by the QEF class, but this function includes some important
    # tricks to cope with edge cases

    # This is demonstration code and isn't optimized, there are many good C++ implementations
    # out there if you need speed.

    if False:#settings.BIAS:
        # Add extra normals that add extra error the further we go
        # from the cell, this encourages the final result to be
        # inside the cell
        # These normals are shorter than the input normals
        # as that makes the bias weaker,  we want them to only
        # really be important when the input is ambiguous

        # Take a simple average of positions as the point we will
        # pull towards.
        mass_point = numpy.mean(positions, axis=0)

        normals.append([settings.BIAS_STRENGTH, 0, 0])
        positions.append(mass_point)
        normals.append([0, settings.BIAS_STRENGTH, 0])
        positions.append(mass_point)
        normals.append([0, 0, settings.BIAS_STRENGTH])
        positions.append(mass_point)

    qef = QEF.make_3d(positions, normals)

    residual, v = qef.solve()

    if settings.BOUNDARY:
        def inside(r):
            return x <= r[1][0] <= x + 1 and y <= r[1][1] <= y + 1 and z <= r[1][2] <= z + 1

        # It's entirely possible that the best solution to the qef is not actually
        # inside the cell.
        if not inside((residual, v)):
            # If so, we constrain the the qef to the 6
            # planes bordering the cell, and find the best point of those
            r1 = qef.fix_axis(0, x + 0).solve()
            r2 = qef.fix_axis(0, x + 1).solve()
            r3 = qef.fix_axis(1, y + 0).solve()
            r4 = qef.fix_axis(1, y + 1).solve()
            r5 = qef.fix_axis(2, z + 0).solve()
            r6 = qef.fix_axis(2, z + 1).solve()

            rs = list(filter(inside, [r1, r2, r3, r4, r5, r6]))

            if len(rs) == 0:
                # It's still possible that those planes (which are infinite)
                # cause solutions outside the box.
                # So now try the 12 lines bordering the cell
                r1  = qef.fix_axis(1, y + 0).fix_axis(0, x + 0).solve()
                r2  = qef.fix_axis(1, y + 1).fix_axis(0, x + 0).solve()
                r3  = qef.fix_axis(1, y + 0).fix_axis(0, x + 1).solve()
                r4  = qef.fix_axis(1, y + 1).fix_axis(0, x + 1).solve()
                r5  = qef.fix_axis(2, z + 0).fix_axis(0, x + 0).solve()
                r6  = qef.fix_axis(2, z + 1).fix_axis(0, x + 0).solve()
                r7  = qef.fix_axis(2, z + 0).fix_axis(0, x + 1).solve()
                r8  = qef.fix_axis(2, z + 1).fix_axis(0, x + 1).solve()
                r9  = qef.fix_axis(2, z + 0).fix_axis(1, y + 0).solve()
                r10 = qef.fix_axis(2, z + 1).fix_axis(1, y + 0).solve()
                r11 = qef.fix_axis(2, z + 0).fix_axis(1, y + 1).solve()
                r12 = qef.fix_axis(2, z + 1).fix_axis(1, y + 1).solve()

                rs = list(filter(inside, [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12]))

            if len(rs) == 0:
                # So finally, we evaluate which corner
                # of the cell looks best
                r1 = qef.eval_with_pos((x + 0, y + 0, z + 0))
                r2 = qef.eval_with_pos((x + 0, y + 0, z + 1))
                r3 = qef.eval_with_pos((x + 0, y + 1, z + 0))
                r4 = qef.eval_with_pos((x + 0, y + 1, z + 1))
                r5 = qef.eval_with_pos((x + 1, y + 0, z + 0))
                r6 = qef.eval_with_pos((x + 1, y + 0, z + 1))
                r7 = qef.eval_with_pos((x + 1, y + 1, z + 0))
                r8 = qef.eval_with_pos((x + 1, y + 1, z + 1))

                rs = list(filter(inside, [r1, r2, r3, r4, r5, r6, r7, r8]))

            # Pick the best of the available options
            residual, v = min(rs)

    if settings.CLIP:
        # Crudely force v to be inside the cell
        v[0] = numpy.clip(v[0], x, x + 1)
        v[1] = numpy.clip(v[1], y, y + 1)
        v[2] = numpy.clip(v[2], z, z + 1)

    return V3(v[0], v[1], v[2])



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
    def make_2d(positions, normals):
        """Returns a QEF that measures the the error from a bunch of normals, each emanating from given positions"""
        A = numpy.array(normals)
        b = [v[0] * n[0] + v[1] * n[1] for v, n in zip(positions, normals)]
        fixed_values = [None] * A.shape[1]
        return QEF(A, b, fixed_values)

    @staticmethod
    def make_3d(positions, normals):
        """Returns a QEF that measures the the error from a bunch of normals, each emanating
         from given positions"""
        A = numpy.array(normals)
        b = [v[0] * n[0] + v[1] * n[1] + v[2] * n[2] for v, n in zip(positions, normals)]
        fixed_values = [None] * A.shape[1]
        return QEF(A, b, fixed_values)

    def fix_axis(self, axis, value):
        """Returns a new QEF that gives the same values as the old one, only with the position along the given axis
        constrained to be value."""
        # Pre-evaluate the fixed axis, adjusting b
        b = self.b[:] - self.A[:, axis] * value
        # Remove that axis from a
        A = numpy.delete(self.A, axis, 1)
        fixed_values = self.fixed_values[:]
        fixed_values[axis] = value
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
