from dataclasses import dataclass
from typing import Optional

import numpy as np

from gempy_engine.core.data.exported_fields import ExportedFields


@dataclass(init=True)
class CombinedScalarFieldsOutput:
    squeezed_mask_array: np.ndarray
    final_block: np.ndarray  # Masked array containing only the active voxels
    faults_block: np.ndarray
    final_exported_fields: ExportedFields  # Masked array containing only the active voxels
    #: Weight of this stack at each point when the stacks are combined: the boolean lithology mask, or its sigmoid
    #: version when ``InterpolationOptions.erosion_sigmoid_slope`` is set. Used to blend unit properties.
    blend_mask_array: Optional[np.ndarray] = None
