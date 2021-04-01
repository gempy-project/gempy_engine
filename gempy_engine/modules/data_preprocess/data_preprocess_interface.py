from gempy_engine.core.data import SurfacePoints, Orientations, SurfacePointsInternals, OrientationsInternals, \
    InterpolationOptions
import numpy as np

from gempy_engine.modules.data_preprocess._grid_preparation import export_vector
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, orientations_preprocess


def prepare_surface_points(surface_points: SurfacePoints, number_points_per_surface: np.ndarray):
    return surface_points_preprocess(surface_points, number_points_per_surface)


def prepare_orientations(orientations: Orientations):
    return orientations_preprocess(orientations)


def prepare_solution_vector(sp_internals: SurfacePointsInternals,
                            ori_internals: OrientationsInternals,
                            grid: np.ndarray,
                            cov_size: int,
                            n_dim: int):
    # TODO: Split export vector in multiple methods
    return export_vector(sp_internals, ori_internals, grid, cov_size, n_dim)
