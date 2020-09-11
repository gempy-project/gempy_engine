from dataclasses import dataclass
from typing import Union

import numpy as np
import tensorflow as tf


@dataclass
class OrientationsInternals:
    dip_positions_tiled: Union[np.ndarray, tf.Tensor] = np.empty((0, 3))


