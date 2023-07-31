from dataclasses import dataclass

import numpy as np


@dataclass
class GenericGrid:
    values: np.ndarray = np.zeros((0, 3))