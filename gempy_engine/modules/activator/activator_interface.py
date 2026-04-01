import numpy as np

from ._soft_segment import soft_segment_unbounded
from ...core.data.exported_fields import ExportedFields


def activate_formation_block(exported_fields: ExportedFields, ids: np.ndarray,
                             sigmoid_slope: float) -> np.ndarray:
    sigm = soft_segment_unbounded(
        Z=exported_fields.scalar_field_everywhere,
        edges=exported_fields.scalar_field_at_surface_points,
        ids=ids,
        sigmoid_slope=sigmoid_slope
    )
    return sigm

