from typing import Tuple

from numpy import ndarray

from gempy_engine.core.data import SurfacePoints, Orientations, SurfacePointsInternals, TensorsStructure, OrientationsInternals
import numpy as np
from gempy_engine.core.backend_tensor import BackendTensor as bt


from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, orientations_preprocess


def prepare_surface_points(surface_points: SurfacePoints, tensor_structure: TensorsStructure) -> SurfacePointsInternals:
    return surface_points_preprocess(surface_points, tensor_structure)


def prepare_orientations(orientations: Orientations) -> OrientationsInternals:
    return orientations_preprocess(orientations)


def prepare_grid(grid: np.ndarray, surface_points: SurfacePoints) -> np.ndarray:
    concat = bt.tfnp.concatenate([grid, surface_points.sp_coords])
    #concat = np.concatenate([grid, surface_points.sp_coords])    
    return concat


def prepare_faults(faults_values_on_sp: np.ndarray, tensors_structure: TensorsStructure) -> Tuple[ndarray, ndarray]:
    
    partitions_bool = tensors_structure.partitions_bool
    number_repetitions = tensors_structure.number_of_points_per_surface - 1

    ref_points = faults_values_on_sp[:, partitions_bool]

    ref_matrix_val_repeated = np.repeat(ref_points, number_repetitions, 1)
    rest_matrix_val = faults_values_on_sp[:, ~partitions_bool]

    return ref_matrix_val_repeated, rest_matrix_val
