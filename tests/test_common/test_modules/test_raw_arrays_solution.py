import pytest

from gempy_engine.core.data.raw_arrays_solution import RawArraysSolution


def test_meshes_to_subsurface_without_meshes():
    """A compute with mesh_extraction=False leaves no vertices, and the server still exports them."""
    pytest.importorskip("subsurface")
    meshes = RawArraysSolution().meshes_to_subsurface()
    assert meshes.vertex.shape == (0, 3)
    assert meshes.cells.shape == (0, 3)
