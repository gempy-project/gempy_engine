from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

MicroKernelType = Literal["exponential", "matern_3_2", "matern_5_2"]


class MicroAnisotropicOptions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = False
    points: Optional[np.ndarray] = None          # (N, 3) micro constraint points
    residuals: Optional[np.ndarray] = None       # (N,) target residual values
    anisotropy_matrices: Optional[np.ndarray] = None  # (N, 3, 3) per-point anisotropy transforms
    weights: Optional[np.ndarray] = None         # (N,) solved micro weights
    kernel_range: float = 1.0                    # range for the micro kernel
    kernel_type: MicroKernelType = "matern_5_2"  # kernel function for micro solve + eval
    nugget: float = 0.0                          # diagonal nugget for the micro solve
    preserve_macro_points: bool = True           # include macro SP as zero-residual constraints
    strength: float = 1.0                        # global strength multiplier (1.0 = full correction)
