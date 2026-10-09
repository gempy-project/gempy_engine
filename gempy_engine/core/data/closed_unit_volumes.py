from dataclasses import dataclass, field

import numpy as np

OUTSIDE = ('outside',)


@dataclass
class ClosedUnitVolumes:
    """Closed, outward-oriented unit shells from one shared triangle complex.

    Every triangle separates ``unit_sides[i] = (negative-side unit, positive-side
    unit)`` with respect to its winding normal; units are ``(partition, stack,
    interval)`` (partition = fault bank, or 0 without faults) and ``OUTSIDE``.
    ``patch_kinds`` is 'interface', 'fault_skin' or 'extent'. ``shell(unit)``
    returns that unit's triangles oriented outward. ``vertex_keys`` are the
    canonical identities (partition-namespaced, never welded across banks).
    """
    vertices: np.ndarray
    triangles: np.ndarray
    unit_sides: list
    patch_kinds: list
    vertex_keys: list
    report: dict = field(default_factory=dict)

    @property
    def units(self):
        return sorted({u for sides in self.unit_sides for u in sides if u != OUTSIDE})

    def shell(self, unit):
        """(vertices, outward triangles) of one unit, indexing the shared vertices."""
        rows = []
        for triangle, (negative, positive) in zip(self.triangles, self.unit_sides):
            if negative == unit:
                rows.append(triangle)
            elif positive == unit:
                rows.append(triangle[[0, 2, 1]])
        return self.vertices, np.asarray(rows, dtype=np.int64).reshape(-1, 3)

    @property
    def volumes(self):
        return dict(self.report.get('volumes', {}))
