"""Vectorised per-(surface, minimal edge) decisions of joint extraction.

A surface keeps a dual face on a minimal edge it strictly crosses when every
controller keeps the crossing point. Errors keep the edge-major, surface-minor
order of a per-edge sweep.
"""

import numpy as np


def edge_crossing_decisions(edge_fields, controllers, edge_cells, counts):
    """Retained crossings and their fraction along each minimal edge.

    ``edge_fields`` (S, E, 2) are endpoint fields minus levels, ``edge_cells``
    (E, 4) the cyclic leaf ring (-1 outside the extent) and ``counts`` (S, N)
    the original branch count per leaf. Returns ``(selected, fraction)``:
    interior edges whose dual face is emitted, and the crossing fraction.
    """
    start_fields, end_fields = edge_fields[:, :, 0], edge_fields[:, :, 1]
    aligned = np.any(np.abs(edge_fields) < 1e-10, axis=2)
    crossing = ~aligned & (start_fields*end_fields < 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction = np.where(crossing, start_fields/(start_fields-end_fields), 0.)
    junction_aligned = np.zeros_like(crossing)
    retained = crossing.copy()
    for t, rows in controllers.items():
        for c, sign in rows:
            decision = sign*((1-fraction[t])*start_fields[c]+fraction[t]*end_fields[c])
            junction_aligned[t] |= crossing[t] & (np.abs(decision) < 1e-10)
            retained[t] &= decision > 0
    interior = np.all(edge_cells >= 0, axis=1)
    selected = retained & ~junction_aligned & interior[None, :]
    hanging = np.zeros_like(selected)
    for t in range(len(selected)):
        hanging[t] = selected[t] & np.any(counts[t][np.maximum(edge_cells, 0)] != 1, axis=1)
    errors = ((aligned, 'sample_aligned_interface: minimal edge endpoint lies on surface'),
              (junction_aligned, 'grid_edge_junction: geological ownership is aligned'),
              (hanging, 'unsupported_hanging_branch: crossing lacks original leaf branch'))
    failing = np.flatnonzero(np.any([mask.T for mask, _ in errors], axis=0).reshape(-1))
    if len(failing):
        e, t = divmod(int(failing[0]), len(selected))
        raise ValueError(next(message for mask, message in errors if mask[t, e]))
    return selected, fraction


def crossing_normals(fraction, endpoint_gradients):
    """(S, E, 3) gradients interpolated to each crossing; orient the ring triangles."""
    return (1-fraction)[:, :, None]*endpoint_gradients[:, :, 0] + fraction[:, :, None]*endpoint_gradients[:, :, 1]
