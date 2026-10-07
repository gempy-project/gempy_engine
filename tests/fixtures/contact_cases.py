"""Small analytic contact catalogue; no pytest registration or interpolation.

Each call returns fresh numpy extraction inputs in the unit cube. Plane equations
are ``xyz @ normals[i] = levels[i]``. Contacts list (controller, truncated,
retained_sign): retain points where sign * controller residual >= 0. Seams are
independent analytic line segments, not legacy mesh-derived expectations.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from gempy_engine.core.data.dual_contouring_data import DualContouringData
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge


@dataclass
class ContactCase:
    name: str
    resolution: tuple[int, int, int]
    normals: np.ndarray
    levels: np.ndarray
    surface_to_stack: tuple[int, ...]
    relations: tuple[StackRelationType, ...]
    contacts: tuple[tuple[int, int, int], ...]
    seams: dict[tuple[int, int], np.ndarray]
    junction: np.ndarray | None
    dc_data: list[DualContouringData]

    @property
    def active_cells(self):
        return [data.left_right_codes[data.valid_voxels] for data in self.dc_data]


def build_contact_case(name: str, resolution=6) -> ContactCase:
    """Build planar_erosion, onlap, parallel_false_overlap or three_way_junction.

    ``resolution`` is an integer or three positive integers (cells per axis).
    Geometry is fixed as resolution changes; false overlap is resolved at fine
    resolutions. Inputs use legacy edge crossings and legacy triangulation,
    with RAW (unmasked) support. Ownership metadata does not clip the inputs.
    """
    shape = np.asarray(resolution)
    if shape.ndim == 0:
        shape = np.repeat(shape, 3)
    if shape.shape != (3,) or not np.issubdtype(shape.dtype, np.integer) or np.any(shape < 1):
        raise ValueError("resolution must be an integer or three positive integers")
    shape = tuple(int(n) for n in shape)
    erode, onlap, basement = StackRelationType.ERODE, StackRelationType.ONLAP, StackRelationType.BASEMENT
    junction = None
    if name == "planar_erosion":
        normals, levels = [(0, 0, 1), (1, 0, 0)], [.47, .43]
        relations, contacts = (erode, basement), ((0, 1, -1),)
        seams = {(0, 1): [[.43, 0, .47], [.43, 1, .47]]}
    elif name == "onlap":
        normals, levels = [(1, 0, 0), (0, 0, 1)], [.43, .47]
        relations, contacts = (onlap, basement), ((1, 0, 1),)
        seams = {(0, 1): [[.43, 0, .47], [.43, 1, .47]]}
    elif name == "parallel_false_overlap":
        normals, levels = [(1, 0, 0), (1, 0, 0)], [.43, .46]
        relations, contacts, seams = (erode, basement), (), {}
    elif name == "three_way_junction":
        normals, levels = [(1, 0, 0), (0, 1, 0), (0, 0, 1)], [.43, .46, .47]
        relations, contacts = (erode, erode, basement), ((0, 1, -1), (0, 2, -1), (1, 2, -1))
        junction = np.array([.43, .46, .47])
        seams = {(0, 1): [[.43, .46, 0], [.43, .46, 1]],
                 (0, 2): [[.43, 0, .47], [.43, 1, .47]],
                 (1, 2): [[0, .46, .47], [1, .46, .47]]}
    else:
        raise ValueError(f"Unknown contact case: {name}")
    normals, levels = np.asarray(normals, dtype=float), np.asarray(levels, dtype=float)
    cells = np.array(list(product(*(range(n) for n in shape))), dtype=int)
    offsets = np.array(list(product((0, 1), repeat=3)))
    corners = ((cells[:, None, :] + offsets) / shape).reshape(-1, 3)
    centers = (cells + .5) / shape
    data = []
    for normal, level in zip(normals, levels):
        xyz, valid = find_intersection_on_edge(corners, corners @ normal, np.array([level]))
        data.append(DualContouringData(
            xyz_on_edge=xyz, valid_edges=valid, xyz_on_centers=centers.copy(),
            dxdydz=1 / np.asarray(shape), n_surfaces_to_export=1,
            left_right_codes=cells.copy(), gradients=np.tile(normal, (len(xyz), 1)),
            tree_depth=1, base_number=shape,
        ))
    return ContactCase(name, shape, normals, levels, tuple(range(len(levels))),
                       relations, contacts, {pair: np.asarray(line) for pair, line in seams.items()},
                       junction, data)
