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


def precision_tolerance(tolerance, dtype=np.float64):
    """A float64 agreement bound widened to what ``dtype`` production values resolve.

    Float64 bounds are returned unchanged.
    """
    eps = np.finfo(dtype).eps
    return tolerance if eps <= np.finfo(np.float64).eps else max(tolerance, 2**14*eps)


def _node_codes(corners, domain_shape):
    stride = np.asarray(domain_shape).astype(np.int64)+1
    weights = np.array([stride[1]*stride[2], stride[2], 1])
    codes = (corners.reshape(-1, 3).astype(np.int64)*weights).sum(axis=1)
    return np.unique(codes, return_index=True, return_inverse=True), weights


def snap_straddling_nodes(corners, samples, levels, domain_shape, aligned=1e-10):
    """Raw samples where leaves sharing a node straddle a surface take one value.

    Leaves evaluating a node separately (other octree level, low precision)
    can put it on both sides of a surface, or exactly on it in one leaf only
    (quantized float32 fields). Such corners take the node's first sample (the
    ``shared_node_fields`` value), so every leaf sees one side. A node then
    within ``aligned`` of a surface counts as positive (the usual ``f >= 0``
    sided rule): aligned endpoints emit no dual face, which would open a hole
    around the node. All other samples are returned unchanged.
    """
    n = len(samples)
    (_, first, inverse), _ = _node_codes(corners, domain_shape)
    flat = samples.reshape(n, -1)
    canonical = flat[:, first][:, inverse.reshape(-1)]
    fields, canonical_fields = flat-levels[:, None], canonical-levels[:, None]
    straddle = np.sign(fields) != np.sign(canonical_fields)
    snapped = np.where(straddle, canonical, flat)
    # Just past the aligned bound and resolvable at the isovalue's magnitude.
    sided = levels[:, None] + np.maximum(10*aligned, 4*np.spacing(np.abs(levels[:, None])))
    return np.where(np.abs(snapped-levels[:, None]) < aligned, sided, snapped).reshape(samples.shape)


def shared_node_fields(corners, fields, domain_shape):
    """One field value per distinct leaf-corner node: the first leaf's.

    Returns ``(codes, node_fields, weights)`` for ``lookup_node_fields``.
    """
    (unique, first, _), weights = _node_codes(corners, domain_shape)
    return unique, fields.reshape(len(fields), -1)[:, first], weights


def lookup_node_fields(nodes, codes, node_fields, weights):
    """(S, K) fields at ``nodes`` (..., 3), which must all be leaf corners."""
    node_codes = (nodes.reshape(-1, 3).astype(np.int64)*weights).sum(axis=1)
    found = np.minimum(np.searchsorted(codes, node_codes), len(codes)-1)
    if np.any(codes[found] != node_codes):
        raise ValueError('unsupported_adaptive_node: tile or edge node is not a leaf corner')
    return node_fields[:, found]

