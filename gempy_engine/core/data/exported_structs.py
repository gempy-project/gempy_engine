from dataclasses import dataclass
import numpy as np

@dataclass
class ExportedFields:
    scalar_field: np.ndarray
    gx_field: np.ndarray
    gy_field: np.ndarray
    gz_field: np.ndarray = None

@dataclass
class Output:
    exported_fields: ExportedFields
    scalar_field_at_sp: np.ndarray
    values_block: np.ndarray # final values ignoring unconformities
    final_block: np.ndarray # Masked array containing only the active voxels