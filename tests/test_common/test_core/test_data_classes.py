from types import SimpleNamespace

import numpy as np

from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.core.data.scalar_field_output import ScalarFieldOutput
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.core.data.stack_relation_type import StackRelationType


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


def test_exported_fields_dense_grid_preserves_gradients():
    exported_fields = ExportedFields(
        _scalar_field=np.arange(5.0),
        _gx_field=np.arange(10.0, 15.0),
        _gy_field=np.arange(20.0, 25.0),
        _gz_field=np.arange(30.0, 35.0),
        _grid_size=5,
    )
    output = ScalarFieldOutput(
        weights=None,
        grid=SimpleNamespace(dense_grid_slice=slice(2, 5)),
        exported_fields=exported_fields,
        stack_relation=StackRelationType.ERODE,
        values_block=None,
    )

    dense_fields = output.exported_fields_dense_grid

    assert np.array_equal(dense_fields.scalar_field, np.arange(2.0, 5.0))
    assert np.array_equal(dense_fields.gx_field, np.arange(12.0, 15.0))
    assert np.array_equal(dense_fields.gy_field, np.arange(22.0, 25.0))
    assert np.array_equal(dense_fields.gz_field, np.arange(32.0, 35.0))


def test_exported_fields_dense_grid_preserves_missing_gradients():
    exported_fields = ExportedFields(_scalar_field=np.arange(5.0), _grid_size=5)
    output = ScalarFieldOutput(
        weights=None,
        grid=SimpleNamespace(dense_grid_slice=slice(2, 5)),
        exported_fields=exported_fields,
        stack_relation=StackRelationType.ERODE,
        values_block=None,
    )

    dense_fields = output.exported_fields_dense_grid

    assert dense_fields.gx_field is None
    assert dense_fields.gy_field is None
    assert dense_fields.gz_field is None
