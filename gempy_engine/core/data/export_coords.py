from dataclasses import dataclass
import numpy as np

@dataclass
class ExportCoordInternals:
    dips_i: np.array = np.empty((0, 1, 3))
    grid_j: np.array = np.empty((0, 1, 3))
    ref_i: np.array = np.empty((0, 1, 3))
    rest_i: np.array = np.empty((0, 1, 3))
