import tensorflow as tf
import numpy as np

from gempy_engine.data_structures.private_structures import OrientationsInternals, InterpolationOptions, \
    SurfacePointsInternals
from gempy_engine.data_structures.public_structures import OrientationsInput, KrigingParameters, SurfacePointsInput
from gempy_engine.systems.generators import *


def cov_gradients_f(orientations_input: OrientationsInput,
                    dip_positions_tiled: np.ndarray,
                    kriging_parameters: KrigingParameters,
                    n_dimensions: int):

    sed_dips_dips = sed_dips_dips = squared_euclidean_distances(
        dip_positions_tiled,
        dip_positions_tiled
    )

    h_u = cartesian_distances(orientations_input.dip_positions,
                              orientations_input.dip_positions,
                              n_dimensions)
    h_v = tensor_transpose(h_u)
    _perpendicular_matrix = compute_perpendicular_matrix(
        orientations_input.dip_positions.shape[0]
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


def create_covariance(orientations_input: OrientationsInput,
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

    return cov_gradients


def tile_dip_positions(dip_positions, n_dimensions):
    return tile_dip_positions(dip_positions, (n_dimensions, 1))


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
        self.kriging_parameters = KrigingParameters()

        # This is per series

        # Used for cov and export
        self._orientations_internals.dip_positions_tiled = tile_dip_positions(
            self.orientations_input.dip_positions,
            n_dimensions=self.options.number_dimensions
        )

        # Used in many places
        self._sp_internals.ref_surface_points,\
        self._sp_internals.rest_surface_points = get_ref_rest(

        )




        self.covariance = create_covariance(
            self.orientations_input,
            self._orientations_internals.dip_positions_tiled,
            self.kriging_parameters,
            self.options
        )
        # -------------------


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
