from types import SimpleNamespace

import numpy as np

from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.solutions import Solutions


def _make_stub_octree_level():
    # Minimal stand-in for an OctreeLevel: no scalar fields and no octree
    # grid, which is what Solutions sees when meshes are not extracted.
    return SimpleNamespace(outputs=[], grid=SimpleNamespace(octree_grid=None))


def test_solutions_repr_with_meshes_none():
    # With mesh_extraction=False (e.g. dense-grid interpolation options)
    # dc_meshes is None; repr must not raise.
    solutions = Solutions(octrees_output=[_make_stub_octree_level()], dc_meshes=None)

    assert repr(solutions) == "Solutions(1 Octree Levels, 0 DualContouringMeshes)"
    assert solutions._repr_html_() == "<b>Solutions:</b> 1 Octree Levels, 0 DualContouringMeshes"


def test_solutions_repr_with_meshes():
    solutions = Solutions(octrees_output=[_make_stub_octree_level()], dc_meshes=["mesh_a", "mesh_b"])

    assert repr(solutions) == "Solutions(1 Octree Levels, 2 DualContouringMeshes)"
    assert solutions._repr_html_() == "<b>Solutions:</b> 1 Octree Levels, 2 DualContouringMeshes"
