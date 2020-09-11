from dataclasses import dataclass

import numpy as np


@dataclass
class OrientationsInput:
    dip_positions: np.ndarray
    dip_gradients: np.ndarray = None
    dip: np.array = None
    azimuth: np.ndarray = None
    polarity: np.ndarray = None
