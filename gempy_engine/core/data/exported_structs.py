from dataclasses import dataclass
import numpy as np

@dataclass
class ExportedFields:
    scalar_field: np.ndarray
    gx_field: np.ndarray
    gy_field: np.ndarray
    gz_field: np.ndarray