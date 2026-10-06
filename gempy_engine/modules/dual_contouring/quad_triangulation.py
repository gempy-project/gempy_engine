"""Same-level dual quads keyed by (lower integer edge endpoint, direction)."""

from ...config import AvailableBackends
from ...core.backend_tensor import BackendTensor
from ._aux import _correct_normals
from .fancy_triangulation import _get_pack_factors


def triangulate_quads(coordinates, valid_edges, voxel_normals, vertices, domain_shape,
                      generated_coordinates=None, report=None):
    """Emit two triangles per complete quad, never partial triangles.

    Coordinates include retained non-crossing cells; vertices include only cells
    with any valid edge, in the same order. Crossing flags come from the existing
    tolerant intersection rule, not a new sign test. Conflicting flags on shared
    edges are rejected. Missing-cell counts are incidences; edge categories can
    overlap. Without pre-mask coordinates, interior omissions are 'unknown'.
    This does not stitch different octree levels or resolve ambiguous topology.
    """
    t = BackendTensor.t
    report = {} if report is None else report
    report.clear()
    report.update(crossing_edge_count=0, quad_count=0, missing_incident_cell_count=0,
                  physical_boundary_edge_count=0, mask_boundary_edge_count=0,
                  internal_refinement_boundary_edge_count=0, unknown_boundary_edge_count=0)
    empty = t.zeros((0, 3), dtype='int64')
    factors = _get_pack_factors(*domain_shape)
    bounds = t.array(domain_shape, dtype='int64')
    if coordinates.shape != (len(valid_edges), 3) or valid_edges.shape != (len(coordinates), 12):
        raise ValueError("Expected cell coordinates (N, 3) and crossing flags (N, 12)")
    if t.any((coordinates < 0) | (coordinates >= bounds)):
        raise ValueError("Cell coordinates must lie inside the theoretical domain")
    codes = (coordinates * factors).sum(axis=1)
    order = t.argsort(codes)
    sorted_codes = codes[order]
    if t.any(sorted_codes[1:] == sorted_codes[:-1]):
        raise ValueError("Duplicate cell coordinates are not supported")
    active = t.any(valid_edges, axis=1)
    if len(vertices) != int(active.sum()):
        raise ValueError("Expected one vertex per crossing cell")
    if len(coordinates) == 0:
        return empty

    # Edge numbering matches find_intersection_on_edge: x, then y, then z.
    offsets = t.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                       [0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1],
                       [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype='int64')
    directions = t.array([0] * 4 + [1] * 4 + [2] * 4, dtype='int64')
    origins = coordinates[:, None, :] + offsets[None, :, :]
    ids = t.concatenate((origins, t.zeros(origins.shape[:2] + (1,), dtype='int64') +
                         directions[None, :, None]), axis=2).reshape(-1, 4)
    if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
        unique, inverse, counts = t.unique(ids, dim=0, sorted=True, return_inverse=True, return_counts=True)
    else:
        unique, inverse, counts = t.unique(ids, axis=0, return_inverse=True, return_counts=True)
    crossings = t.bincount(inverse[valid_edges.reshape(-1)], minlength=len(unique))
    if t.any((crossings != 0) & (crossings != counts)):
        raise ValueError("Inconsistent crossing flags for duplicate canonical edges")
    edges = unique[crossings > 0]
    report['crossing_edge_count'] = len(edges)
    if len(edges) == 0:
        return empty

    # Cyclic incident cells; the fixed 1--3 diagonal matches legacy complete quads.
    incident_offsets = t.array([
        [[0, -1, -1], [0, -1, 0], [0, 0, 0], [0, 0, -1]],
        [[-1, 0, -1], [-1, 0, 0], [0, 0, 0], [0, 0, -1]],
        [[-1, -1, 0], [-1, 0, 0], [0, 0, 0], [0, -1, 0]]
    ], dtype='int64')
    incident = edges[:, None, :3] + incident_offsets[edges[:, 3]]
    inside = t.all((incident >= 0) & (incident < bounds), axis=2)
    keys = (incident * factors).sum(axis=2)
    positions = t.clip(t.searchsorted(sorted_codes, keys.reshape(-1)), 0, len(codes) - 1).reshape(-1, 4)
    found = inside & (sorted_codes[positions] == keys)
    cells = order[positions]
    complete = t.all(found, axis=1)
    report['quad_count'] = int(complete.sum())
    report['missing_incident_cell_count'] = int((~found).sum())
    report['physical_boundary_edge_count'] = int(t.any(~inside, axis=1).sum())
    missing_inside = inside & ~found
    if generated_coordinates is None:
        report['unknown_boundary_edge_count'] = int(t.any(missing_inside, axis=1).sum())
    else:
        generated_codes = (generated_coordinates * factors).sum(axis=1)
        generated = t.isin(keys, generated_codes)
        report['mask_boundary_edge_count'] = int(t.any(missing_inside & generated, axis=1).sum())
        report['internal_refinement_boundary_edge_count'] = int(t.any(missing_inside & ~generated, axis=1).sum())
    if not t.any(complete):
        return empty

    # All four cells of a consistent crossing edge have a QEF vertex.
    vertex_ids = t.cumsum(active, axis=0) - 1
    quads = vertex_ids[cells[complete]]
    split = t.array([[0, 1, 3], [2, 3, 1]], dtype='int64')
    triangles = quads[:, split].reshape(-1, 3)
    local_edges = t.array([[3, 2, 0, 1], [7, 6, 4, 5], [11, 10, 8, 9]], dtype='int64')
    reference = voxel_normals[cells[complete], local_edges[edges[complete, 3]]].sum(axis=1)
    reference = t.repeat(reference, 2, axis=0)
    return _correct_normals(vertices, triangles, reference)[0]
