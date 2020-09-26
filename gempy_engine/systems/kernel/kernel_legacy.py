import numpy as np

from gempy_engine.data_structures.private_structures import SurfacePointsInternals
from gempy_engine.data_structures.public_structures import OrientationsInput, InterpolationOptions
from gempy_engine.systems.generators import squared_euclidean_distances, tensor_transpose
from gempy_engine.systems.kernel.aux_functions import cartesian_distances, compute_perpendicular_matrix, \
    compute_cov_gradients, compute_cov_sp, compute_cov_sp_grad, compute_drift_uni_grad, compute_drift_uni_sp, \
    covariance_assembly


def cov_gradients_f(orientations_input: OrientationsInput,
                    dip_positions_tiled: np.ndarray,
                    kriging_parameters: InterpolationOptions,
                    n_dimensions: int):
    sed_dips_dips = squared_euclidean_distances(
        dip_positions_tiled,
        dip_positions_tiled
    )

    h_u = cartesian_distances(
        #dip_positions_tiled,
        #dip_positions_tiled,
        orientations_input.dip_positions,
                              orientations_input.dip_positions,
                              n_dimensions)
    h_v = tensor_transpose(h_u)
    _perpendicular_matrix = compute_perpendicular_matrix(
        orientations_input.dip_positions.shape[0], n_dimensions
    )

    cov_grad_matrix = compute_cov_gradients(
        sed_dips_dips,
        h_u,
        h_v,
        _perpendicular_matrix,
        kriging_parameters,
        orientations_input.nugget_effect_grad
    )

    return cov_grad_matrix


def cov_sp_f(sp_internal: SurfacePointsInternals,
             kriging_parameters: InterpolationOptions):
    sed_rest_rest = squared_euclidean_distances(
        sp_internal.rest_surface_points,
        sp_internal.rest_surface_points
    )
    sed_ref_rest = squared_euclidean_distances(
        sp_internal.ref_surface_points,
        sp_internal.rest_surface_points
    )
    sed_rest_ref = squared_euclidean_distances(
        sp_internal.rest_surface_points,
        sp_internal.ref_surface_points
    )
    sed_ref_ref = squared_euclidean_distances(
        sp_internal.ref_surface_points,
        sp_internal.ref_surface_points
    )

    cov_sp_matrix = compute_cov_sp(
        sed_rest_rest,
        sed_ref_rest,
        sed_rest_ref,
        sed_ref_ref,
        kriging_parameters,
        sp_internal.nugget_effect_ref_rest
    )

    return cov_sp_matrix


def cov_sp_grad_f(orientations_input: OrientationsInput,
                  dip_positions_tiled: np.ndarray,
                  sp_internal: SurfacePointsInternals,
                  kriging_parameters: InterpolationOptions,
                  n_dimensions: int):
    sed_dips_rest = squared_euclidean_distances(
        dip_positions_tiled,
        sp_internal.rest_surface_points
    )

    sed_dips_ref = squared_euclidean_distances(
        dip_positions_tiled,
        sp_internal.ref_surface_points
    )

    hu_rest = cartesian_distances(
        orientations_input.dip_positions,
        sp_internal.rest_surface_points,
        n_dimensions, cross_variance=True  # This is independent of the number of dimensions
    )
    hu_ref = cartesian_distances(
        orientations_input.dip_positions,
        sp_internal.ref_surface_points,
        n_dimensions, cross_variance=True
    )

    cov_sp_grad_matrix = compute_cov_sp_grad(
        sed_dips_rest, sed_dips_ref, hu_rest, hu_ref, kriging_parameters
    )

    return cov_sp_grad_matrix


def drift_uni_f(dip_positions: np.ndarray, sp_internal: SurfacePointsInternals,
                gi: float, degree: int, n_dim:int):
    drift_uni_grad = compute_drift_uni_grad(dip_positions, n_dim, gi, degree=degree)
    drift_uni_sp = compute_drift_uni_sp(sp_internal, n_dim, gi, degree=degree)
    return drift_uni_grad, drift_uni_sp


def create_covariance_legacy(
        sp_internals: SurfacePointsInternals,
        orientations_input: OrientationsInput,
        dip_positions_tiled: np.ndarray,
        kriging_parameters: InterpolationOptions,

):
    cov_gradients = cov_gradients_f(
        orientations_input,
        dip_positions_tiled,
        kriging_parameters,
        kriging_parameters.number_dimensions
    )

    cov_sp = cov_sp_f(sp_internals, kriging_parameters)

    cov_sp_grad = cov_sp_grad_f(
        orientations_input,
        dip_positions_tiled,
        sp_internals,
        kriging_parameters,
        kriging_parameters.number_dimensions
    )

    drift_uni_grad, dridf_uni_sp = drift_uni_f(
        orientations_input.dip_positions,
        sp_internals,
        kriging_parameters.gi_res,
        kriging_parameters.uni_degree,
        kriging_parameters.number_dimensions
    )

    covariance_matrix = covariance_assembly(
        cov_sp,
        cov_gradients,
        cov_sp_grad,
        drift_uni_grad,
        dridf_uni_sp
    )

    return covariance_matrix