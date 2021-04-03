from gempy_engine.core.data import SurfacePointsInternals, OrientationsInternals
from gempy_engine.core.backend_tensor import BackendTensor as bt
from gempy_engine.core.data.export_coords import ExportCoordInternals


def export_vector(sp_internals: SurfacePointsInternals,
                  ori_internals: OrientationsInternals,
                  grid, cov_size, n_dim):

    dips_init = bt.tfnp.zeros((cov_size, n_dim))
    ref_init = bt.tfnp.zeros((cov_size, n_dim))
    rest_init = bt.tfnp.zeros((cov_size, n_dim))

    dips = ori_internals.dip_positions_tiled
    dips_init[:ori_internals.n_orientations_tiled] = dips

    dips_i = dips_init[:, None, :]
    grid_j = grid[None, :, :]

    s1 = ori_internals.n_orientations_tiled
    s2 = s1 + sp_internals.n_points

    ref_init[s1:s2] = sp_internals.ref_surface_points
    rest_init[s1:s2] = sp_internals.rest_surface_points

    ref_i = ref_init[:, None, :]
    rest_i = rest_init[:, None, :]

    ei = ExportCoordInternals(dips_i, grid_j, ref_i, rest_i)
    return ei
