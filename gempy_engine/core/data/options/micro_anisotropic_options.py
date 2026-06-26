from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict


class MicroAnisotropicOptions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = False
    points: Optional[np.ndarray] = None          # (N, 3) micro contact points
    residuals: Optional[np.ndarray] = None       # (N,) target residual values
    anisotropy_matrices: Optional[np.ndarray] = None  # (N, 3, 3) per-point anisotropy transforms
    weights: Optional[np.ndarray] = None         # (N,) solved micro weights
    kernel_range: float = 1.0                    # range for the micro exponential kernel
    nugget: float = 0.0                          # diagonal nugget for the micro solve
