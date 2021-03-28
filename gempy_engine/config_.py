
from dataclasses import dataclass

import numpy as np
from typing import Union

use_tf = False  # Whether using TensorFlow or numpy
use_jax = True

use_pykeops = False # Whether using pykeops for reduction

try:
    import jax.numpy as jnp
    jax_imported = use_jax
except ImportError:
    jax_imported = False

try:
    import tensorflow as tf

    # Set CPU as available physical device
    # my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    # tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')
    # if we import it we check in config
    tensorflow_imported = use_tf
except ImportError:
    tensorflow_imported = False

try:
    import pykeops
    pykeops_imported = use_pykeops
except ImportError:
    pykeops_imported = False

# There are 3 possibles cases for the numpy-TF compatibility
# 1) signature and args are the same -> Nothing needs to be changed
# 2) signature is different but args are the same -> We need to override the
#    name of the function
# 3) signature and args are different -> We need an if statement

if use_jax is True:
    tfnp = jnp
    tfnp.reduce_sum = tfnp.sum
    tfnp.concat = tfnp.concatenate
    tfnp.constant = tfnp.array
elif use_tf is True:
    tfnp = tf
else:
    tfnp = np



# @dataclass
# class GemPyTensor:
#     use_tf = False  # Whether using TensorFlow or numpy
#     use_pykeops = False  # Whether using pykeops for reduction
#     use_jax = False
#
#     @property
#     def tfnp(self):
#         if self.use_jax is True:
#             tfnp = jnp
#             tfnp.reduce_sum = tfnp.sum
#             tfnp.concat = tfnp.concatenate
#             tfnp.constant = tfnp.array
#         elif self.use_tf is True:
#             tfnp = tf
#         else:
#             tfnp = np
#             tfnp.reduce_sum = tfnp.sum
#             tfnp.concat = tfnp.concatenate
#             tfnp.constant = tfnp.array
#         return tfnp


# gempy_tensor = GemPyTensor()



