from dataclasses import dataclass

import numpy as np

from gempy_engine.core.data import SurfacePoints, Orientations
from gempy_engine.core.data.grid import Grid


@dataclass
class InterpolationInput:
    surface_points: SurfacePoints
    orientations: Orientations
    grid: Grid
    unit_values: np.ndarray