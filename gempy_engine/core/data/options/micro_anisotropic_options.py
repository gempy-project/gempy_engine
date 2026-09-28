from typing import Literal

from pydantic import BaseModel

MicroKernelType = Literal["exponential", "matern_3_2", "matern_5_2"]


class MicroAnisotropicOptions(BaseModel):
    enabled: bool = False
    kernel_range: float = 1.0                    # range for the micro kernel
    kernel_type: MicroKernelType = "matern_5_2"  # kernel function for micro solve + eval
    nugget: float = 0.0                          # diagonal nugget for the micro solve
    preserve_macro_points: bool = True           # include macro SP as zero-residual constraints
    strength: float = 1.0                        # global strength multiplier (1.0 = full correction)
