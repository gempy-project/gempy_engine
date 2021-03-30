from gempy_engine.core.data import SurfacePoints, Orientations, TensorsStructure, InterpolationOptions
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_kriging_eq
from gempy_engine.modules.solver.solver_interface import kernel_reduction


def interpolate_single_scalar(surface_points: SurfacePoints, orientations: Orientations,
                     options: InterpolationOptions, data_shape: TensorsStructure):

    A_matrix, b_vector = yield_kriging_eq(surface_points, orientations, options, data_shape)
    weights = kernel_reduction(A_matrix, b_vector, smooth=0.01) # TODO: Smooth should be taken from options
