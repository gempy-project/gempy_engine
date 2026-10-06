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
    support_to_engine: MicroArray | None = None

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
        if self.support_to_engine is not None:
            transform = self.support_to_engine
            if transform.shape != (3, 3):
                raise ValueError("micro_points.support_to_engine must have shape (3, 3)")
            if is_pytorch_installed and isinstance(transform, torch.Tensor):
                finite = torch.isfinite(transform).all()
                determinant = torch.linalg.det(transform) if finite else None
                valid = finite and torch.isfinite(determinant) and determinant != 0
            else:
                finite = np.isfinite(transform).all()
                determinant = np.linalg.det(transform) if finite else None
                valid = finite and np.isfinite(determinant) and determinant != 0
            if not valid:
                raise ValueError("micro_points.support_to_engine must be finite and invertible")
        self.surface_indices = indices.astype(np.int64)


@dataclass(frozen=True)
class MicroPointResults:
    """Aligned contact rows in the root MicroPoints array (not preservation centers)."""
    source_indices: np.ndarray
    macro_gradients: MicroArray
    anisotropy_matrices: MicroArray
