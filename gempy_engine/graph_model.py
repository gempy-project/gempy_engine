from gempy_engine.data_structures.private_structures import OrientationsInternals, \
    SurfacePointsInternals
from gempy_engine.data_structures.public_structures import OrientationsInput, \
    InterpolationOptions, TensorsStructure
from gempy_engine.systems.generators import *
from gempy_engine.systems.kernel.aux_functions import b_scalar_assembly
from gempy_engine.systems.kernel.kernel_legacy import create_covariance_legacy
from gempy_engine.systems.reductions import solver
from gempy_engine.systems.transformations import dip_to_gradients
from gempy_engine.config import tfnp, tensorflow_imported, tensor_types


# def tile_dip_positions(dip_positions, n_dimensions):
#     return tile_dip_positions(dip_positions, (n_dimensions, 1))


class GemPyEngineCommon:
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
        self.kriging_parameters = InterpolationOptions()

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

        self.covariance = create_covariance_legacy(
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


if tensorflow_imported:
    inheritance = [tfnp.Module, GemPyEngineCommon]

    class GemPyEngine(*inheritance):
        @tfnp.function
        def __call__(self, *args, **kwargs):
            """ Here I imagine that we pass the public variables

            Args:
                *args:
                **kwargs:

            Returns:

            """

            return self._call(*args, **kwargs)

else:
    inheritance = [GemPyEngineCommon]

    class GemPyEngine(*inheritance):
        def __call__(self, *args, **kwargs):
            """ Here I imagine that we pass the public variables

            Args:
                *args:
                **kwargs:

            Returns:

            """

            return self._call(*args, **kwargs)


