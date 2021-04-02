from gempy_engine.core.data import SurfacePoints, Orientations, TensorsStructure, InterpolationOptions
from gempy_engine.modules.data_preprocess.data_preprocess_interface import prepare_surface_points, prepare_orientations, \
    prepare_solution_vector
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance, yield_b_vector
from gempy_engine.modules.solver.solver_interface import kernel_reduction

import numpy as np

def interpolate_single_scalar(surface_points: SurfacePoints, orientations: Orientations, grid:np.ndarray,
                     options: InterpolationOptions, data_shape: TensorsStructure):

    sp_internal = prepare_surface_points(surface_points, data_shape.number_of_points_per_surface)
    ori_internal = prepare_orientations(orientations)

    A_matrix = yield_covariance(sp_internal, ori_internal, options)
    cov_size = A_matrix.shape[0]
    b_vector = yield_b_vector(ori_internal, cov_size)

    weights = kernel_reduction(A_matrix, b_vector, smooth=0.01) # TODO: Smooth should be taken from options

    # TODO: Obsolete? If I am able to construct the kernel properly
    solution_vector = prepare_solution_vector(sp_internal, ori_internal, grid, cov_size, options.number_dimensions)