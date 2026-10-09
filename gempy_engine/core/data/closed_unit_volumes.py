from dataclasses import dataclass, field

import numpy as np

OUTSIDE = ('outside',)


@dataclass
class ClosedUnitVolumes:
    """Closed, outward-oriented unit shells from one shared triangle complex.

    Every triangle separates ``unit_sides[i] = (negative-side unit, positive-side
    unit)`` with respect to its winding normal; units are ``(partition, stack,
    interval)`` (partition = fault bank, or 0 without faults) and ``OUTSIDE``.
    ``patch_kinds`` is 'interface', 'fault_skin' or 'extent'; with a fault, the
    'fault_skin' triangles are one shared interface between bank-0 and bank-1
    units. ``shell(unit)`` returns that unit's triangles oriented outward.
    ``vertex_keys`` are the canonical identities (partition-namespaced; shared
    fault vertices keep their bank-0 key, crossings are ('fault_crossing', i)).
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

    def collapse_short_edges(self, ratio=.2):
        """A copy with needle edges collapsed, keeping features and closure (see ``collapse_short_edges``)."""
        from ...modules.dual_contouring.closed_volume_cleanup import collapse_short_edges
        return collapse_short_edges(self, ratio)

    def tetrahedralize(self, **kwargs):
        """One conforming tetrahedral mesh of all units (needs the optional ``tetgen``).

        See ``tetrahedralize_closed_volumes`` for the options.
        """
        from ...modules.dual_contouring.closed_volume_tetmesh import tetrahedralize_closed_volumes
        return tetrahedralize_closed_volumes(self, **kwargs)


@dataclass
class UnitTetMesh:
    """Conforming tetrahedral mesh of closed unit volumes.

    ``tet_units[i]`` indexes ``units`` for tetrahedron ``i`` (positively
    oriented). ``boundary_faces`` are the faces lying on the closed-volume
    triangles; ``boundary_patches`` gives, for each, the index of the source
    triangle in ``surface.triangles`` (so its kind and side units), where
    ``surface`` is the ``ClosedUnitVolumes`` actually meshed (after any
    short-edge collapse).
    """
    nodes: np.ndarray
    tetrahedra: np.ndarray
    tet_units: np.ndarray
    units: list
    boundary_faces: np.ndarray
    boundary_patches: np.ndarray
    surface: 'ClosedUnitVolumes' = None
    report: dict = field(default_factory=dict)

    @property
    def unit_volumes(self):
        p = self.nodes[self.tetrahedra]
        volumes = np.einsum('ij,ij->i', p[:, 1]-p[:, 0], np.cross(p[:, 2]-p[:, 0], p[:, 3]-p[:, 0]))/6
        totals = np.bincount(self.tet_units, weights=volumes, minlength=len(self.units))
        return {unit: float(v) for unit, v in zip(self.units, totals)}
