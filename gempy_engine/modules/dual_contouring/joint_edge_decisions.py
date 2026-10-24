"""Vectorised per-(surface, minimal edge) decisions of joint extraction.

A surface keeps a dual face on a minimal edge it strictly crosses when every
controller keeps the crossing point. Aligned endpoints, aligned junctions and
crossings without a vertex in every ring leaf emit no face.
"""

import numpy as np

from .joint_xp import namespace


def edge_crossing_decisions(edge_fields, controllers, edge_cells, counts):
    """Retained crossings and their fraction along each minimal edge.

    ``edge_fields`` (S, E, 2) are endpoint fields minus levels, ``edge_cells``
    (E, 4) the cyclic leaf ring (-1 outside the extent) and ``counts`` (S, N)
    the original branch count per leaf. Returns ``(selected, fraction)``:
    interior edges whose dual face is emitted, and the crossing fraction.
    """
    xp = namespace(edge_fields, edge_cells, counts)
    start_fields, end_fields = edge_fields[:, :, 0], edge_fields[:, :, 1]
    aligned = (xp.abs(edge_fields) < 1e-10).any(axis=2)
    crossing = ~aligned & (start_fields*end_fields < 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction = xp.where(crossing, start_fields/(start_fields-end_fields), 0.)
    junction_aligned = xp.zeros(tuple(crossing.shape), dtype=bool)
    retained = xp.copy(crossing)
    for t, rows in controllers.items():
        for c, sign in rows:
            decision = sign*((1-fraction[t])*start_fields[c]+fraction[t]*end_fields[c])
            junction_aligned[t] |= crossing[t] & (xp.abs(decision) < 1e-10)
            retained[t] &= decision > 0
    interior = (edge_cells >= 0).all(axis=1)
    selected = retained & ~junction_aligned & interior[None, :]
    # A crossing without one vertex in every ring leaf has no dual face: skip it.
    ring = xp.maximum(edge_cells, 0)
    for t in range(len(selected)):
        selected[t] &= (counts[t][ring] == 1).all(axis=1)
    return selected, fraction


def crossing_normals(fraction, endpoint_gradients):
    """(S, E, 3) gradients interpolated to each crossing; orient the ring triangles."""
    return (1-fraction)[:, :, None]*endpoint_gradients[:, :, 0] + fraction[:, :, None]*endpoint_gradients[:, :, 1]
