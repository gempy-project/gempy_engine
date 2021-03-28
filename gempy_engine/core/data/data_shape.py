from dataclasses import dataclass
from typing import List

import numpy as np

@dataclass
class TensorsStructure:
    number_of_points_per_surface: np.ndarray
    len_c_g: np.int32
    len_c_gi: np.int32
    len_sp: np.int32
    len_faults: np.int32