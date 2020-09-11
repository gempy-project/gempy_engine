import tensorflow as tf
import numpy as np

from gempy_engine.data_structures.private_structures import OrientationsInternals
from gempy_engine.data_structures.public_structures import OrientationsInput


class GemPyEngine(tf.Module):
    def __init__(self):
        """Here we need to initialize the private classes
        """
        self.orientations_internals = OrientationsInternals()

    def __call__(self, *args, **kwargs):
        """ Here I imagine that we pass the public variables

        Args:
            *args:
            **kwargs:

        Returns:

        """
        self.orientations_input = OrientationsInput()

        return

    @tf.function
    def tile_dip_positions(self, dip_positions, n_dimensions):
        return tf.tile(dip_positions, (n_dimensions, 1))
