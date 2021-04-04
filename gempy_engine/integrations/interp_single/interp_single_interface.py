from gempy_engine.core.data import SurfacePoints, Orientations, TensorsStructure, InterpolationOptions
from gempy_engine.core.data.exported_structs import ExportedFields, Output
from gempy_engine.modules.data_preprocess.data_preprocess_interface import prepare_surface_points, prepare_orientations, \
    prepare_grid
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance, yield_b_vector, \
    yield_evaluation_kernel, yield_evaluation_grad_kernel
from gempy_engine.modules.solver.solver_interface import kernel_reduction

import numpy as np


def interpolate_single_scalar(surface_points: SurfacePoints, orientations: Orientations, grid: np.ndarray,
                              options: InterpolationOptions, data_shape: TensorsStructure):
    sp_internal = prepare_surface_points(surface_points, data_shape.number_of_points_per_surface)
    ori_internal = prepare_orientations(orientations)
    grid_internal = prepare_grid(grid, surface_points)

    A_matrix = yield_covariance(sp_internal, ori_internal, options)
    b_vector = yield_b_vector(ori_internal, A_matrix.shape[0])

    weights = kernel_reduction(A_matrix, b_vector, smooth=0.01)  # TODO: Smooth should be taken from options

    exported_fields = _evaluate_sys_eq(grid_internal, options, ori_internal, sp_internal, weights)
    scalar_at_surface_points = _get_scalar_field_at_surface_points(exported_fields.scalar_field,
                                                                   data_shape.nspv,
                                                                   surface_points.n_points
                                                                   )
    output = Output(exported_fields, scalar_at_surface_points)

    # -----------------
    # TODO: [ ] Export block

    # TODO: [ ] Topology

    # TODO: [ ] Octree
    # ------------------
    # TODO: [ ] Dual contouring



    # TODO: [ ] Masking OPs

    # ---------------------

    # TODO: [ ] Gravity

    # TODO: [ ] Magnetics
    return output

def _evaluate_sys_eq(grid, options, ori_internal, sp_internal, weights) -> ExportedFields:
    eval_kernel = yield_evaluation_kernel(grid, sp_internal, ori_internal, options)
    eval_gx_kernel = yield_evaluation_grad_kernel(grid, sp_internal, ori_internal, options, axis=0)
    eval_gy_kernel = yield_evaluation_grad_kernel(grid, sp_internal, ori_internal, options, axis=1)

    scalar_field = weights @ eval_kernel
    gx_field = weights @ eval_gx_kernel
    gy_field = weights @ eval_gy_kernel
    if options.number_dimensions == 3:
        eval_gz_kernel = yield_evaluation_grad_kernel(grid, sp_internal, ori_internal, options, axis=2)
        gz_field = weights @ eval_gz_kernel
    elif options.number_dimensions == 2:
        gz_field = None
    else:
        raise ValueError("Number of dimensions have to be 2 or 3")

    # TODO: Add here the logic to extract the scalar field value at surface point

    return ExportedFields(scalar_field, gx_field, gy_field, gz_field)


def _get_scalar_field_at_surface_points(Z_x: np.ndarray, number_of_points_per_surface: np.ndarray, n_surface_points: int):
    npf = number_of_points_per_surface
    scalar_field_at_surface_points_values = Z_x[-n_surface_points:][npf]

    return scalar_field_at_surface_points_values
