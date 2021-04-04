from gempy_engine.core.data import SurfacePoints, Orientations, SurfacePointsInternals
import numpy as np
from gempy_engine.core.backend_tensor import BackendTensor as bt

from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, orientations_preprocess


def prepare_surface_points(surface_points: SurfacePoints, number_points_per_surface: np.ndarray):
    return surface_points_preprocess(surface_points, number_points_per_surface)


def prepare_orientations(orientations: Orientations):
    return orientations_preprocess(orientations)

def prepare_grid(grid: np.ndarray, surface_points: SurfacePoints):
    return bt.tfnp.concatenate([grid, surface_points.sp_coords])