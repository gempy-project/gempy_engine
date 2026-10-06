"""CPU-only topology diagnostics, independent of triangle-candidate filtering."""
from itertools import product

import numpy as np

from ...core.backend_tensor import BackendTensor


def mesh_support_report(coordinates, scalar_corners, isovalue, domain_shape,
                        mask=None, surface_index=0, ancestor_coordinates=()):
    """Classify missing incident cells for unique sampled sign-changing edges.

    This diagnoses sampled crossings, not unsampled components or later triangle
    removal by overlap/fault processing. Counts of missing cells are incidences.
    """
    to_numpy = BackendTensor.t.to_numpy
    coords = np.asarray(to_numpy(coordinates), dtype=np.int64)
    scalar = np.asarray(to_numpy(scalar_corners)).reshape(-1, 8)
    iso = float(isovalue)
    retained = np.ones(len(coords), dtype=bool) if mask is None else np.asarray(to_numpy(mask), dtype=bool)
    generated = set(map(tuple, coords))
    kept = set(map(tuple, coords[retained]))
    ancestors = [set(map(tuple, to_numpy(c))) for c in ancestor_coordinates]
    bounds = tuple(int(n) for n in domain_shape)
    corners = np.array(list(product((0, 1), repeat=3)), dtype=np.int64)
    edges = set()
    for direction, pairs in enumerate((
        ((0, 4), (1, 5), (2, 6), (3, 7)),
        ((0, 2), (1, 3), (4, 6), (5, 7)),
        ((0, 1), (2, 3), (4, 5), (6, 7)),
    )):
        for a, b in pairs:
            crossing = (scalar[:, a] >= iso) != (scalar[:, b] >= iso)
            edges.update((direction, *p) for p in coords[crossing] + corners[a])
    report = dict(surface_index=surface_index, crossing_edge_count=len(edges),
                  missing_incident_cell_count=0, physical_boundary_edge_count=0,
                  mask_boundary_edge_count=0, internal_refinement_boundary_edge_count=0,
                  violations=[])
    for edge in sorted(edges):
        direction, *origin = edge
        transverse = [i for i in range(3) if i != direction]
        missing = []
        kinds = set()
        for offsets in product((-1, 0), repeat=2):
            cell = list(origin)
            for axis, offset in zip(transverse, offsets):
                cell[axis] += offset
            cell = tuple(cell)
            if cell in kept:
                continue
            outside = any(c < 0 or c >= n for c, n in zip(cell, bounds))
            kind = 'physical' if outside else 'mask' if cell in generated else 'refinement'
            kinds.add(kind)
            stopped = None
            if kind == 'refinement':
                for level, existing in enumerate(ancestors):
                    shift = len(ancestors) - level
                    if tuple(c >> shift for c in cell) in existing:
                        stopped = level
            missing.append(dict(coordinate=cell, kind=kind, outside_extent=outside,
                                ancestor_stop_level=stopped))
        if missing:
            report['missing_incident_cell_count'] += len(missing)
            for kind, key in (('physical', 'physical_boundary_edge_count'),
                              ('mask', 'mask_boundary_edge_count'),
                              ('refinement', 'internal_refinement_boundary_edge_count')):
                report[key] += kind in kinds
            report['violations'].append(dict(direction=direction, coordinate=tuple(origin), missing=missing))
    return report
