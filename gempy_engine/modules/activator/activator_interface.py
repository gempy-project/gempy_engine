import numpy as np

from ._soft_segment import soft_segment_unbounded
from ...core.data.exported_fields import ExportedFields


def activate_formation_block(exported_fields: ExportedFields, ids: np.ndarray,
                             sigmoid_slope: float, drop_lowest_edge: bool = False) -> np.ndarray:
    """Segment a scalar field into unit ids.

    Args:
        drop_lowest_edge: Leave out the lowest surface of the stack (and the unit below it), so the block has no
            transition there. Used for eroding stacks when the erosion contact is blended by a soft mask instead:
            a second, sharper step at the same contact would carry no gradient.
    """
    edges = exported_fields.scalar_field_at_surface_points
    if drop_lowest_edge:
        edges, ids = edges[:-1], ids[:-1]

    sigm = soft_segment_unbounded(
        Z=exported_fields.scalar_field_everywhere,
        edges=edges,
        ids=ids,
        sigmoid_slope=sigmoid_slope
    )
    return sigm
