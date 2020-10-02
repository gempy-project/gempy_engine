import numpy as np
from typing import Union

use_tf = False  # Whether using TensorFlow or numpy
use_pykeops = False # Whether using pykeops for reduction

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

# Case 1)
tfnp = tf if tensorflow_imported else np
tensor_types = Union[np.ndarray, tf.Tensor, tf.Variable]

# Case 2)
if tensorflow_imported is False:
    tfnp.reduce_sum = tfnp.sum
    tfnp.concat = tfnp.concatenate
    tfnp.constant = tfnp.array

