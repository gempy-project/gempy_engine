from gempy_engine.core.data import SurfacePoints, Orientations, TensorsStructure, InterpolationOptions
from gempy_engine.core.data.exported_structs import ExportedFields
from gempy_engine.modules.data_preprocess.data_preprocess_interface import prepare_surface_points, prepare_orientations
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance, yield_b_vector, \
    yield_evaluation_kernel, yield_evaluation_grad_kernel
from gempy_engine.modules.solver.solver_interface import kernel_reduction

import numpy as np


def interpolate_single_scalar(surface_points: SurfacePoints, orientations: Orientations, grid: np.ndarray,
                              options: InterpolationOptions, data_shape: TensorsStructure):
    sp_internal = prepare_surface_points(surface_points, data_shape.number_of_points_per_surface)
    ori_internal = prepare_orientations(orientations)

    A_matrix = yield_covariance(sp_internal, ori_internal, options)
    b_vector = yield_b_vector(ori_internal, A_matrix.shape[0])

    weights = kernel_reduction(A_matrix, b_vector, smooth=0.01)  # TODO: Smooth should be taken from options

    # -----------------
    exported_fields = _evaluate_sys_eq(grid, options, ori_internal, sp_internal, weights)
    # TODO: [ ] Topology

    # TODO: [ ] Octree
    # ------------------
    # TODO: [ ] Dual contouring

    # TODO: [ ] Export block

    # TODO: [ ] Masking OPs

    # ---------------------

    # TODO: [ ] Gravity

    # TODO: [ ] Magnetics


def _evaluate_sys_eq(grid, options, ori_internal, sp_internal, weights):
    eval_kernel = yield_evaluation_kernel(grid, sp_internal, ori_internal, options)
    eval_gx_kernel = yield_evaluation_grad_kernel(grid, sp_internal, ori_internal, options, axis=0)
    eval_gy_kernel = yield_evaluation_grad_kernel(grid, sp_internal, ori_internal, options, axis=1)
    eval_gz_kernel = yield_evaluation_grad_kernel(grid, sp_internal, ori_internal, options, axis=2)
    scalar_field = weights @ eval_kernel
    gx_field = weights @ eval_gx_kernel
    gy_field = weights @ eval_gy_kernel
    gz_field = weights @ eval_gz_kernel
    return ExportedFields(scalar_field, gx_field, gy_field, gz_field)
