import numpy as np


def _generate_next_level_centers(xyz_coord, dxdydz, level=1):
    def _expand(slr_x, slr_y, slr_z):
        x_ = np.repeat(slr_x, 4, axis=1)
        x = x_.ravel()
        y_ = np.tile(np.repeat(slr_y, 2, axis=1), (1, 2))
        y = y_.ravel()
        z_ = np.tile(slr_z, (1, 4))
        z = z_.ravel()
        new_xyz = np.stack((x, y, z)).T
        return new_xyz
    # ===================================
    x_coord, y_coord, z_coord = xyz_coord[:, 0], xyz_coord[:, 1], xyz_coord[:, 2]
    dx, dy, dz = dxdydz

    def stack_left_right(a_edg, d_a):
        return np.stack((a_edg - d_a / level / 4, a_edg + d_a / level / 4), axis=1)

    slr_x = stack_left_right(x_coord, dx)
    slr_y = stack_left_right(y_coord, dy)
    slr_z = stack_left_right(z_coord, dz)

    new_xyz = _expand(slr_x, slr_y, slr_z)
    return new_xyz


def _generate_corners(xyz_coord, dxdydz, level=1):
    x_coord, y_coord, z_coord = xyz_coord[:, 0], xyz_coord[:, 1], xyz_coord[:, 2]
    dx, dy, dz = dxdydz

    def stack_left_right(a_edg, d_a):
        return np.stack((a_edg - d_a / level / 2, a_edg + d_a / level / 2), axis=1)

    x_ = np.repeat(stack_left_right(x_coord, dx), 4, axis=1)
    x = x_.ravel()
    y_ = np.tile(np.repeat(stack_left_right(y_coord, dy), 2, axis=1), (1, 2))
    y = y_.ravel()
    z_ = np.tile(stack_left_right(z_coord, dz), (1, 4))
    z = z_.ravel()

    new_xyz = np.stack((x, y, z)).T
    return new_xyz

def _generate_faces(xyz_coord, dxdydz, level=1):
    x_coord, y_coord, z_coord = xyz_coord
    dx, dy, dz = dxdydz

    def stack_left_right(a_edg, d_a):
        return np.stack((a_edg - d_a / level / 2, a_edg, a_edg,  a_edg + d_a / level / 2), axis=1)

    x_ = np.repeat(stack_left_right(x_coord, dx), 3, axis=1)
    x = x_.ravel()
    y_ = np.tile(np.repeat(stack_left_right(y_coord, dy), 2, axis=1), (1, 1))
    y = y_.ravel()
    z_ = np.tile(stack_left_right(z_coord, dz), (1, 3))
    z = z_.ravel()

    new_xyz = np.stack((x, y, z)).T
    return new_xyz