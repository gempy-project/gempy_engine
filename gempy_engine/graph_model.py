import tensorflow as tf
import numpy as np

from gempy_engine.data_structures.private_structures import OrientationsInternals, InterpolationOptions, \
    SurfacePointsInternals
from gempy_engine.data_structures.public_structures import OrientationsInput, KrigingParameters, SurfacePointsInput, \
    TensorsStructure
from gempy_engine.systems.generators import *
from gempy_engine.systems.reductions import solver
from gempy_engine.systems.transformations import dip_to_gradients


def cov_gradients_f(orientations_input: OrientationsInput,
                    dip_positions_tiled: np.ndarray,
                    kriging_parameters: KrigingParameters,
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
             kriging_parameters: KrigingParameters):
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
                  kriging_parameters: KrigingParameters,
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


def create_covariance(
        sp_internals: SurfacePointsInternals,
        orientations_input: OrientationsInput,
        dip_positions_tiled: np.ndarray,
        kriging_parameters: KrigingParameters,
        options: InterpolationOptions
):
    cov_gradients = cov_gradients_f(
        orientations_input,
        dip_positions_tiled,
        kriging_parameters,
        options.number_dimensions
    )

    cov_sp = cov_sp_f(sp_internals, kriging_parameters)

    cov_sp_grad = cov_sp_grad_f(
        orientations_input,
        dip_positions_tiled,
        sp_internals,
        kriging_parameters,
        options.number_dimensions
    )

    drift_uni_grad, dridf_uni_sp = drift_uni_f(
        orientations_input.dip_positions,
        sp_internals,
        kriging_parameters.gi_res,
        kriging_parameters.uni_degree,
        options.number_dimensions
    )

    covariance_matrix = covariance_assembly(
        cov_sp,
        cov_gradients,
        cov_sp_grad,
        drift_uni_grad,
        dridf_uni_sp
    )

    return covariance_matrix


# def tile_dip_positions(dip_positions, n_dimensions):
#     return tile_dip_positions(dip_positions, (n_dimensions, 1))


class GemPyEngine:
    """This class should be backend agnostic, i.e. it should not have any
    trace of tensorflow"""

    def __init__(self):
        """Here we need to initialize the private classes and constants
        """

        # Constants
        # ---------
        self.options = InterpolationOptions()

        # Private
        # -------
        self._orientations_internals = OrientationsInternals()
        self._sp_internals = SurfacePointsInternals()

    def __call__(self, *args, **kwargs):
        """ Here I imagine that we pass the public variables

        Args:
            *args:
            **kwargs:

        Returns:

        """

        return self._call(*args, **kwargs)
    
    def _call(self, *args, **kwargs):
        """This function contains the main logic. It has to be different
        to __call__ so we can later on wrap it has tensor flow function"""

        self.sp_input = SurfacePointsInput()
        self.orientations_input = OrientationsInput()
        self.tensor_structures = TensorsStructure()
        self.kriging_parameters = KrigingParameters()

        # This is per series

        # Used for cov and export
        self._orientations_internals.dip_positions_tiled = tile_dip_positions(
            self.orientations_input.dip_positions,
            n_dimensions=self.options.number_dimensions
        )

        # Used in many places
        (self._sp_internals.ref_surface_points,
         self._sp_internals.rest_surface_points,
         self._sp_internals.nugget_effect_ref_rest) = get_ref_rest(
            self.sp_input,
            self.tensor_structures.number_of_points_per_surface
        )

        self.covariance = create_covariance(
            self._sp_internals,
            self.orientations_input,
            self._orientations_internals.dip_positions_tiled,
            self.kriging_parameters,
            self.options
        )
        # -------------------
        grad = dip_to_gradients(self.orientations_input)
        b_vector = b_scalar_assembly(grad, self.covariance.shape[0])
        weights = solver(self.covariance, b_vector)


class GemPyEngineTF(tf.Module, GemPyEngine):
    @tf.function
    def __call__(self, *args, **kwargs):
        """ Here I imagine that we pass the public variables

        Args:
            *args:
            **kwargs:

        Returns:

        """

        return self._call(*args, **kwargs)
