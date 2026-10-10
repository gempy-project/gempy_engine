"""Leaf corners, canonical face tiles and minimal edges on the finest dyadic lattice.

Nodes are integer lattice coordinates; physical points are ``bounds[::2] +
nodes * spacing``. Every tile and minimal-edge node of a balanced octree is a
leaf corner, so their field values can be read from the leaf corner samples.
"""

import numpy as np


def lattice_frame(extent, domain_shape):
    """Validated physical bounds (6,) and finest-lattice spacing (3,)."""
    bounds = np.asarray(extent, dtype=float)
    if bounds.shape != (6,) or not np.isfinite(bounds).all() or np.any(bounds[1::2] <= bounds[::2]):
        raise ValueError('invalid_extent: increasing finite physical bounds required')
    return bounds, (bounds[1::2]-bounds[::2])/np.asarray(domain_shape)


def leaf_corner_nodes(origins, spans, corner_offsets):
    """(N, 8, 3) integer corners of every leaf, in ``corner_offsets`` order."""
    return np.asarray(origins)[:, None, :] + np.asarray(spans)[:, None, None]*corner_offsets


def tile_nodes(faces):
    """(F, 4, 3) tile corners ordered 00, 10, 01, 11 in the two non-normal axes."""
    nodes = np.repeat(np.array([f['origin'] for f in faces], dtype=np.int64).reshape(-1, 1, 3), 4, axis=1)
    axes = np.array([f['axis'] for f in faces], dtype=np.int64)
    spans = np.array([f['span'] for f in faces], dtype=np.int64)
    for axis in range(3):
        rows = axes == axis
        other = [a for a in range(3) if a != axis]
        nodes[np.ix_(rows, range(4), other)] += np.array([[0, 0], [1, 0], [0, 1], [1, 1]])*spans[rows, None, None]
    return nodes


def minimal_edge_nodes(edges):
    """(E, 2, 3) start and end node of every minimal edge."""
    nodes = np.repeat(np.array([e['origin'] for e in edges], dtype=np.int64).reshape(-1, 1, 3), 2, axis=1)
    nodes[np.arange(len(edges)), 1, [e['axis'] for e in edges]] += [e['span'] for e in edges]
    return nodes


def field_tolerances(samples, levels):
    """Per-surface agreement tolerance: relative to the field range plus a few raw ULPs."""
    fields = samples-levels[:, None, None]
    scales = np.ptp(fields, axis=(1, 2))
    # Permit a few representable raw-value ULPs, not a relative raw-offset error.
    roundoff = 4*np.maximum(np.max(np.abs(np.spacing(samples)), axis=(1, 2)), np.abs(np.spacing(levels)))
    return 1e-10*(1+scales)+roundoff


def shared_node_fields(corners, fields, domain_shape, tolerances):
    """One field value per distinct leaf-corner node; leaves sharing a node must agree.

    Returns ``(codes, node_fields, weights)`` for ``lookup_node_fields``.
    """
    n = len(fields)
    stride = np.asarray(domain_shape).astype(np.int64)+1
    weights = np.array([stride[1]*stride[2], stride[2], 1])
    codes = (corners.reshape(-1, 3).astype(np.int64)*weights).sum(axis=1)
    unique, first, inverse = np.unique(codes, return_index=True, return_inverse=True)
    flat = fields.reshape(n, -1)
    low = np.full((n, len(unique)), np.inf)
    high = np.full((n, len(unique)), -np.inf)
    for s in range(n):
        np.minimum.at(low[s], inverse.reshape(-1), flat[s])
        np.maximum.at(high[s], inverse.reshape(-1), flat[s])
    if np.any(high-low > tolerances[:, None]) or np.any((low < 0) & (high > 0)):
        raise ValueError('inconsistent_corner_samples: leaves sharing a lattice node disagree')
    return unique, flat[:, first], weights


def lookup_node_fields(nodes, codes, node_fields, weights):
    """(S, K) fields at ``nodes`` (..., 3), which must all be leaf corners."""
    node_codes = (nodes.reshape(-1, 3).astype(np.int64)*weights).sum(axis=1)
    found = np.minimum(np.searchsorted(codes, node_codes), len(codes)-1)
    if np.any(codes[found] != node_codes):
        raise ValueError('unsupported_adaptive_node: tile or edge node is not a leaf corner')
    return node_fields[:, found]

