"""Tetrahedral unit meshes from closed unit volumes, through TetGen.

The closed-volume complex is already what TetGen needs: one conforming set of
triangles (each stored once) that touch only along shared edges and vertices,
with the unit on each side known. It is passed as a piecewise-linear complex.
Every connected piece of every unit shell gets a region seed, so each
tetrahedron carries its unit. Each input triangle's index is its facet marker,
so boundary faces map back to their patch (kind and side units) even after
TetGen splits them. TetGen rejects self-intersecting input with exact
predicates; that error is reported as ``closed_volume_self_intersection``.

TetGen is an optional dependency (``pip install tetgen``, AGPL-3.0 licensed).
"""

import contextlib
import os
import tempfile

import numpy as np

from ...core.data.closed_unit_volumes import UnitTetMesh


def tetrahedralize_closed_volumes(volumes, *, collapse_ratio=.2, quality=True, min_ratio=1.5, min_dihedral=10.,
                                  max_volume=None, keep_surface=False, switches=''):
    """One conforming tetrahedral mesh of every unit of ``volumes`` (a ``ClosedUnitVolumes``).

    ``collapse_ratio`` first collapses needle edges (``collapse_short_edges``;
    None keeps the triangles as they are). The mesh then fills the collapsed
    complex, which ``mesh.surface`` holds.

    ``quality`` refines to radius-edge ratio ``min_ratio`` and minimum dihedral
    angle ``min_dihedral`` (degrees); ``max_volume`` bounds tetrahedron volume.
    ``keep_surface`` forbids Steiner points on the input triangles, so boundary
    faces are exactly the closed-volume triangles. Extra TetGen ``switches``
    are appended verbatim.
    """
    try:
        import tetgen
    except ImportError as error:
        raise ImportError('tetrahedralize_closed_volumes needs the optional tetgen package '
                          '(pip install tetgen)') from error
    if collapse_ratio is not None:
        from .closed_volume_cleanup import collapse_short_edges
        volumes = collapse_short_edges(volumes, collapse_ratio)
    vertices = np.asarray(volumes.vertices, dtype=np.float64)
    triangles = np.asarray(volumes.triangles, dtype=np.int64)
    units = list(volumes.units)
    mesher = tetgen.TetGen(vertices, triangles.astype(np.int32), np.arange(1, len(triangles)+1, dtype=np.int32))
    seeds = []
    for index, unit in enumerate(units):
        for seed in _region_seeds(*volumes.shell(unit)):
            mesher.add_region(index+1, seed)
            seeds.append((index, seed))
    flags = 'pAQ' + ('Y' if keep_surface else '')
    if quality:
        flags += 'q{:g}/{:g}'.format(min_ratio, min_dihedral)
    if max_volume is not None:
        flags += 'a{:g}'.format(max_volume)
    try:
        with _scratch_directory():  # TetGen writes skipped-facet files to the working directory
            nodes, tetrahedra, attributes, _ = mesher.tetrahedralize(switches=flags+switches)
    except RuntimeError as error:
        if 'self-intersection' in str(error):
            raise ValueError('closed_volume_self_intersection: TetGen found intersecting input triangles') \
                from error
        raise
    nodes = np.asarray(nodes, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    regions = np.rint(np.asarray(attributes, dtype=float).ravel()).astype(np.int64)
    if np.any((regions < 1) | (regions > len(units))):
        raise ValueError('unseeded_tetrahedral_region: a closed region received no unit seed')
    signed = _signed_volumes(nodes, tetrahedra)
    tetrahedra[signed < 0] = tetrahedra[signed < 0][:, [0, 2, 1, 3]]
    faces = np.asarray(mesher.trifaces, dtype=np.int64).reshape(-1, 3)
    markers = np.asarray(mesher.triface_markers, dtype=np.int64).ravel()
    on_input = markers > 0
    mesh = UnitTetMesh(nodes=nodes, tetrahedra=tetrahedra, tet_units=regions-1, units=units,
                       boundary_faces=faces[on_input], boundary_patches=markers[on_input]-1, surface=volumes,
                       report=dict(switches=flags+switches, seeds=len(seeds),
                                   collapse=volumes.report.get('collapse')))
    expected, obtained = _shell_volumes(volumes), mesh.unit_volumes
    total = sum(expected.values())
    errors = {u: abs(obtained.get(u, 0.)-v) for u, v in expected.items()}
    mesh.report.update(volume_errors=errors, min_dihedral=float(_min_dihedral(nodes, tetrahedra).min()))
    if max(errors.values()) > 1e-9*total:
        raise ValueError('tetrahedral_volume_mismatch: unit tetrahedra do not fill their closed shells')
    return mesh


@contextlib.contextmanager
def _scratch_directory():
    previous = os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)
        try:
            yield
        finally:
            os.chdir(previous)


def _shell_volumes(volumes):
    """Signed volume of each unit shell, from the triangles themselves."""
    result = {}
    for unit in volumes.units:
        vertices, faces = volumes.shell(unit)
        a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        result[unit] = float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum()/6)
    return result


def _region_seeds(vertices, faces, candidates=8):
    """One point inside each edge-connected piece of an outward-oriented unit shell."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    parent = np.arange(len(faces))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for t, face in enumerate(faces.tolist()):
        for i in range(3):
            edge = tuple(sorted((face[i], face[(i+1) % 3])))
            if edge in owner:
                parent[find(t)] = find(owner[edge])
            else:
                owner[edge] = t
    roots = np.array([find(t) for t in range(len(faces))])
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normals = np.cross(b-a, c-a)
    areas = np.linalg.norm(normals, axis=1)/2
    perimeters = np.linalg.norm(b-a, axis=1)+np.linalg.norm(c-b, axis=1)+np.linalg.norm(a-c, axis=1)
    seeds = []
    for root in np.unique(roots):
        piece = np.flatnonzero(roots == root)
        for t in piece[np.argsort(-areas[piece], kind='stable')][:candidates]:
            inradius = 2*areas[t]/perimeters[t]
            point = (a[t]+b[t]+c[t])/3-.1*inradius*normals[t]/(2*areas[t])
            if abs(_winding_number(point, a[piece], b[piece], c[piece])-1) < .5:
                seeds.append(point)
                break
        else:
            raise ValueError('unseeded_tetrahedral_region: no interior seed found for a shell piece')
    return seeds


def _winding_number(point, a, b, c):
    """Generalised winding number of ``point`` w.r.t. triangles (a, b, c) (outward = +1 inside)."""
    a, b, c = a-point, b-point, c-point
    la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
    numerator = np.einsum('ij,ij->i', a, np.cross(b, c))
    denominator = la*lb*lc+np.einsum('ij,ij->i', a, b)*lc+np.einsum('ij,ij->i', b, c)*la + \
        np.einsum('ij,ij->i', c, a)*lb
    return float(np.sum(2*np.arctan2(numerator, denominator))/(4*np.pi))


def _signed_volumes(nodes, tetrahedra):
    p = nodes[tetrahedra]
    return np.einsum('ij,ij->i', p[:, 1]-p[:, 0], np.cross(p[:, 2]-p[:, 0], p[:, 3]-p[:, 0]))/6


def _min_dihedral(nodes, tetrahedra):
    """Smallest dihedral angle (degrees) of each tetrahedron."""
    p = nodes[tetrahedra]
    faces = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]
    normals = []
    for i, j, k in faces:
        n = np.cross(p[:, j]-p[:, i], p[:, k]-p[:, i])
        normals.append(n/np.linalg.norm(n, axis=1, keepdims=True))
    angles = []
    for f in range(4):
        for g in range(f+1, 4):
            cosine = np.clip(-np.einsum('ij,ij->i', normals[f], normals[g]), -1, 1)
            angles.append(np.degrees(np.arccos(cosine)))
    return np.min(angles, axis=0)
