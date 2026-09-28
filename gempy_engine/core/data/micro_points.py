from dataclasses import dataclass

import numpy as np
from gempy_engine.config import is_pytorch_installed

if is_pytorch_installed:
    import torch
    MicroArray = np.ndarray | torch.Tensor
else:
    MicroArray = np.ndarray


@dataclass
class MicroPoints:
    """Authored contacts indexed by surface in the owning InterpolationInput.

    Root inputs use global structural-frame indices; stack subsets use local indices.
    """
    points: MicroArray
    anisotropy_matrices: MicroArray
    nuggets: MicroArray
    surface_indices: np.ndarray

    def __post_init__(self):
        indices = np.asarray(self.surface_indices)
        n = len(self.points)
        if (self.points.shape != (n, 3) or self.anisotropy_matrices.shape != (n, 3, 3)
                or self.nuggets.shape != (n,) or indices.shape != (n,)
                or not np.issubdtype(indices.dtype, np.integer)):
            raise ValueError("Invalid micro_points shapes or surface_indices dtype")
        for name in ("points", "anisotropy_matrices", "nuggets"):
            values = getattr(self, name)
            finite = (torch.isfinite(values).all() if is_pytorch_installed and isinstance(values, torch.Tensor)
                      else np.isfinite(values).all())
            if not finite:
                raise ValueError("micro_points must be finite")
        if (self.nuggets < 0).any() or (indices < 0).any():
            raise ValueError("micro_points must have nonnegative nuggets and indices")
        self.surface_indices = indices.astype(np.int64)
