from gempy_engine.core.data import SurfacePoints, Orientations
import numpy as np

from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, orientations_preprocess


def prepare_surface_points(surface_points: SurfacePoints, number_points_per_surface: np.ndarray):
    return surface_points_preprocess(surface_points, number_points_per_surface)


def prepare_orientations(orientations: Orientations):
    return orientations_preprocess(orientations)