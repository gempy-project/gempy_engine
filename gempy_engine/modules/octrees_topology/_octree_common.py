import numpy as np
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.host_arrays import upload


def _generate_next_level_centers_DEP(xyz_coord, dxdydz, level=1):
    def _expand(slr_x, slr_y, slr_z):
        x_ = np.repeat(slr_x, 4, axis=1)
        x = x_.ravel()
        y_ = np.tile(np.repeat(slr_y, 2, axis=1), (1, 2))
        y = y_.ravel()
        z_ = np.tile(slr_z, (1, 4))
        z = z_.ravel()
        new_xyz = np.stack((x, y, z)).T
        return np.ascontiguousarray(new_xyz)
    # ===================================
    x_coord, y_coord, z_coord = xyz_coord[:, 0], xyz_coord[:, 1], xyz_coord[:, 2]
    dx, dy, dz = dxdydz
    
    def stack_left_right(a_edg, d_a):
        return np.stack((a_edg - d_a / level / 4, a_edg + d_a / level / 4), axis=1)
    
    slr_x = stack_left_right(x_coord, dx)
    slr_y = stack_left_right(y_coord, dy)
    slr_z = stack_left_right(z_coord, dz)
    
    # TODO: Refactor this
    bool_x = np.zeros((x_coord.shape[0], 2), dtype=bool)
    bool_x[:, 1] = True
    bool_y = np.zeros((y_coord.shape[0], 2), dtype=bool)
    bool_y[:, 1] = True
    bool_z = np.zeros((z_coord.shape[0], 2), dtype=bool)
    bool_z[:, 1] = True
    
    new_xyz = _expand(slr_x, slr_y, slr_z)
    bool_idx = _expand(bool_x, bool_y, bool_z)
    return new_xyz, bool_idx





# Per child of a cell, its side along each axis (x slowest, z fastest; True: the upper half) and the sign
# of its offset from the cell center.
_CHILD_SIDES = np.array([[k >> 2 & 1, k >> 1 & 1, k & 1] for k in range(8)], dtype=bool)
_CHILD_SIGNS = np.where(_CHILD_SIDES, 1, -1).astype(np.int8)


def _cell_children(xyz_coord, dxdydz, level, divisor):
    """Each row of ``xyz_coord`` moved by ``± d / level / divisor`` along each axis: 8 rows per row, as
    ``_CHILD_SIDES`` orders them. ``x + (-d)`` is ``x - d`` exactly."""
    t = BackendTensor.t
    xyz_coord = t.array(xyz_coord)
    # The cell size elements may be device scalars: stacked, not read back.
    step = t.stack([t.array(d) for d in dxdydz]) / level / divisor
    # The dtype of ``x - d`` for a scalar ``d``: NumPy scalars promote, 0-d tensors do not.
    dtype = np.result_type(xyz_coord.dtype, step.dtype) if isinstance(xyz_coord, np.ndarray) else xyz_coord.dtype
    offsets = upload(_CHILD_SIGNS) * t.array(step, dtype=dtype)
    return (xyz_coord[:, None, :] + offsets[None]).reshape(-1, 3)


def _generate_next_level_centers(xyz_coord, dxdydz, level=1):
    """The centers of the 8 children of each cell centered at ``xyz_coord`` (cell by cell) and their sides."""
    new_xyz = _cell_children(xyz_coord, dxdydz, level, 4)
    bool_idx = BackendTensor.tfnp.tile(upload(_CHILD_SIDES), (new_xyz.shape[0] // 8, 1))
    return new_xyz, bool_idx
