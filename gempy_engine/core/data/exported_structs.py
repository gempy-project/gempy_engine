from dataclasses import dataclass
from typing import Optional

import numpy as np

from gempy_engine.core.data.exported_fields import ExportedFields


@dataclass()
class MaskMatrices:
    mask_lith: np.ndarray
    mask_fault: Optional[np.ndarray]


@dataclass(init=True)
class CombinedScalarFieldsOutput:
    squeezed_mask_array: np.ndarray
    final_block: np.ndarray  # Masked array containing only the active voxels
    final_exported_fields: ExportedFields  # Masked array containing only the active voxels
