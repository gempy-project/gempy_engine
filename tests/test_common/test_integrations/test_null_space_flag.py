"""Null space as a per-stack flag, orthogonal to ERODE/ONLAP, and unit ids of null and point-free stacks."""

import numpy as np
import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.generic_grid import GenericGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.core.backend_tensor import BackendTensor

EXTENT = [0, 10., 0, 2., 0, 5.]


def _plane(height):
    """External field ``z - height``, isovalue 0: positive above the plane."""
    return CustomInterpolationFunctions(
        scalar_field_at_surface_points=np.array([0.0]),
        implicit_function=lambda xyz: xyz[:, 2] - height,
        gx_function=lambda xyz: BackendTensor.t.zeros(len(xyz), dtype=BackendTensor.dtype_obj),
        gy_function=lambda xyz: BackendTensor.t.zeros(len(xyz), dtype=BackendTensor.dtype_obj),
        gz_function=lambda xyz: BackendTensor.t.ones(len(xyz), dtype=BackendTensor.dtype_obj),
    )


def _layers(*heights):
    """Two points per horizontal surface and one upward orientation per stack."""
    return np.array([[x, .5, z] for z in heights for x in (2., 8.)])


COLUMN = np.column_stack([np.full(50, 5.), np.full(50, 1.), np.arange(.05, 5, .1)])


def _compute(stacks, null_space=None, custom=None, ignored=None, elements=None, mesh=False, legacy_null=False,
             overlap=None):
    """``stacks``: (relation, external function or None, surface heights) from youngest to basement.

    The custom grid defaults to a vertical column, ``COLUMN``, to read the layer cake from.
    """
    relations = [relation for relation, _, _ in stacks]
    functions = [function for _, function, _ in stacks]
    heights = [h for _, _, h in stacks]
    if legacy_null:
        relations = [R.NULL_SPACE if flag else relation for relation, flag in zip(relations, null_space)]
        null_space = None
    points = np.concatenate([_layers(*h) for h in heights if h])
    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([2*len(h) for h in heights]),
        number_of_orientations_per_stack=np.array([1 if h else 0 for h in heights]),
        number_of_surfaces_per_stack=np.array([len(h) for h in heights]),
        masking_descriptor=relations,
        interp_functions_per_stack=functions if any(f is not None for f in functions) else None,
        ignored_grid_types_per_stack=ignored,
        null_space_per_stack=null_space,
        number_of_elements_per_stack=None if elements is None else np.array(elements),
    )
    tensors = TensorsStructure(number_of_points_per_surface=np.full(sum(len(h) for h in heights), 2))
    orientations = Orientations(np.array([[5., 1., h[0]] for h in heights if h]),
                                np.array([[0., 0., 1.] for h in heights if h]))
    grid = EngineGrid(octree_grid=RegularGrid(EXTENT, [10, 2, 20]),
                      custom_grid=GenericGrid(values=np.asarray(COLUMN if custom is None else custom, dtype=float)))
    options = InterpolationOptions.from_args(86.6, 3571.4, uni_degree=0, i_res=4, gi_res=2, number_dimensions=3,
                                             kernel_function=AvailableKernelFunctions.cubic)
    options.number_octree_levels = 2
    options.evaluation_options.mesh_extraction = mesh
    if overlap is not None:
        options.evaluation_options.mesh_extraction_overlap = overlap
    return compute_model(InterpolationInput(SurfacePoints(points), orientations, grid, np.arange(1, 10)), options,
                         InputDataDescriptor(tensors, stack_structure))


def _cake(solutions):
    """``(height, id below, id above)`` wherever the id along ``COLUMN`` changes."""
    ids = np.rint(np.asarray(BackendTensor.t.to_numpy(solutions.raw_arrays.custom))).astype(int)
    z = COLUMN[:, 2]
    return [((z[i] + z[i + 1])/2, ids[i], ids[i + 1]) for i in range(len(ids) - 1) if ids[i] != ids[i + 1]]


def _assert_cake(solutions, expected):
    """``expected``: (surface height, id below, id above) from the bottom up."""
    cake = _cake(solutions)
    assert [(i, j) for _, i, j in cake] == [(i, j) for _, i, j in expected], cake
    np.testing.assert_allclose([h for h, _, _ in cake], [h for h, _, _ in expected], atol=.051)


def test_the_flag_computes_like_a_legacy_null_space_relation():
    stacks = [(R.ERODE, _plane(4.), []), (R.ERODE, None, [2.5]), (R.BASEMENT, None, [1.])]
    legacy = _compute(stacks, null_space=[True, False, False], legacy_null=True, mesh=True)
    flagged = _compute(stacks, null_space=[True, False, False], mesh=True)
    np.testing.assert_array_equal(legacy.raw_arrays.lith_block, flagged.raw_arrays.lith_block)
    np.testing.assert_array_equal(legacy.raw_arrays.scalar_field_matrix, flagged.raw_arrays.scalar_field_matrix)
    assert len(legacy.dc_meshes) == len(flagged.dc_meshes) == 2  # the null boundary truncates but is not exported
    for a, b in zip(legacy.dc_meshes, flagged.dc_meshes):
        np.testing.assert_array_equal(a.vertices, b.vertices)
    _assert_cake(flagged, [(1., 3, 2), (2.5, 2, 1), (4., 1, -1)])


def test_an_onlapping_null_stack_voids_above_the_older_top_whatever_its_own_surface():
    older = (R.ERODE, None, [3., 2.])
    erode = _compute([(R.ERODE, _plane(4.), []), older, (R.BASEMENT, None, [1.])], null_space=[True, False, False])
    onlap = _compute([(R.ONLAP, _plane(4.), []), older, (R.BASEMENT, None, [1.])], null_space=[True, False, False])
    # Eroding: void above its own plane (4), so the older stack's top unit shows between 3 and 4.
    _assert_cake(erode, [(1., 4, 3), (2., 3, 2), (3., 2, 1), (4., 1, -1)])
    # Onlapping: void from the older stack's top (3), as for any onlapping stack; its plane at 4 bounds nothing.
    _assert_cake(onlap, [(1., 4, 3), (2., 3, 2), (3., 2, -1)])


def test_a_younger_onlapping_stack_rests_on_the_null_field():
    stacks = [(R.ONLAP, None, [4.5]), (R.ONLAP, _plane(3.8), []), (R.ERODE, None, [3., 2.]), (R.BASEMENT, None, [1.])]
    solutions = _compute(stacks, null_space=[False, True, False, False])
    # The void fills from the older top (3) up to the null plane (3.8), where the young stack starts.
    _assert_cake(solutions, [(1., 5, 4), (2., 4, 3), (3., 3, -1), (3.8, -1, 2), (4.5, 2, 1)])


@pytest.mark.parametrize("relation", [R.ERODE, R.ONLAP])
def test_ignored_custom_points_keep_the_geology_below(relation):
    custom = [[5., 1., 4.6], [5., 1., 4.7], [5., 1., 4.8], [5., 1., 4.9]]
    stacks = [(relation, _plane(4.), []), (R.ERODE, None, [3., 2.]), (R.BASEMENT, None, [1.])]
    nothing = _compute(stacks, null_space=[True, False, False], custom=custom)
    whole = _compute(stacks, null_space=[True, False, False], custom=custom, ignored=[("custom_grid",), (), ()])
    part = _compute(stacks, null_space=[True, False, False], custom=custom, ignored=[(("custom_grid", 0, 2),), (), ()])
    custom_ids = lambda solutions: np.rint(BackendTensor.t.to_numpy(solutions.raw_arrays.custom)).astype(int)
    np.testing.assert_array_equal(custom_ids(nothing), [-1, -1, -1, -1])
    np.testing.assert_array_equal(custom_ids(whole), [1, 1, 1, 1])
    np.testing.assert_array_equal(custom_ids(part), [1, 1, -1, -1])


def test_a_stack_without_points_owns_its_unit():
    stacks = [(R.ERODE, _plane(4.), []), (R.ERODE, None, [2.5]), (R.BASEMENT, None, [1.])]
    # Without element counts the plane's unit merges into the next stack's top unit ...
    _assert_cake(_compute(stacks), [(1., 3, 2), (2.5, 2, 1)])
    # ... with them it is the first unit, and every other id moves down by one.
    _assert_cake(_compute(stacks, elements=[1, 1, 1]), [(1., 4, 3), (2.5, 3, 2), (4., 2, 1)])


def test_a_kriged_null_stack_takes_no_ids():
    stacks = [(R.ERODE, None, [4.]), (R.ERODE, None, [2.5]), (R.BASEMENT, None, [1.])]
    _assert_cake(_compute(stacks, null_space=[True, False, False], elements=[1, 1, 1]),
                 [(1., 3, 2), (2.5, 2, 1), (4., 1, -1)])


def test_only_eroding_onlapping_or_bottom_stacks_can_be_null():
    with pytest.raises(ValueError, match="must erode or onlap"):
        StacksStructure(np.array([2, 2]), np.array([1, 1]), np.array([1, 1]), [R.FAULT, R.BASEMENT],
                        null_space_per_stack=[True, False])
    bottom = StacksStructure(np.array([2, 2]), np.array([1, 1]), np.array([1, 1]), [R.ERODE, R.BASEMENT],
                             null_space_per_stack=[False, True])
    assert bottom.is_bottom_null_space(1) and not bottom.is_bottom_null_space(0)
    legacy = StacksStructure(np.array([0, 2]), np.array([0, 1]), np.array([0, 1]), [R.NULL_SPACE, R.BASEMENT])
    assert legacy.masking_descriptor == [R.ERODE, R.BASEMENT] and legacy.null_space_per_stack == [True, False]


@pytest.mark.parametrize("overlap", ["none", "pretty", "joint", "joint_contacts"])
@pytest.mark.parametrize("relation, void_bottom", [(R.ERODE, 4.), (R.ONLAP, 3.)])
def test_null_stacks_mesh_like_lithology_stacks_without_their_own(relation, overlap, void_bottom):
    """The other stacks' meshes are those of the same model with the null stack as a lithology.

    (In none and pretty, the older top under any onlapping stack comes out empty; the joint modes export it.)
    """
    stacks = [(relation, _plane(4.), []), (R.ERODE, None, [3., 2.]), (R.BASEMENT, None, [1.])]
    null = _compute(stacks, null_space=[True, False, False], mesh=True, overlap=overlap)
    lithology = _compute(stacks, mesh=True, overlap=overlap, elements=[1, 2, 1])
    others = [mesh for mesh in lithology.dc_meshes if mesh.stack_index != 0]
    assert [m.stack_index for m in null.dc_meshes] == [m.stack_index for m in others] == [1, 1, 2]
    for a, b in zip(null.dc_meshes, others):
        np.testing.assert_array_equal(a.vertices, b.vertices)
        np.testing.assert_array_equal(a.edges, b.edges)
        assert not len(a.vertices) or np.asarray(a.vertices)[:, 2].max() <= void_bottom + 1e-6


@pytest.mark.parametrize("bottom", [(R.BASEMENT, None, [2., 1.]), (R.BASEMENT, _plane(1.), [])], ids=["kriged", "external"])
def test_a_null_bottom_stack_voids_above_its_lowest_surface_and_leaves_the_basement_below(bottom):
    stacks = [(R.ERODE, None, [3.]), bottom]
    # The basement takes the unit after the other stacks' (2); the bottom stack's elements own none.
    _assert_cake(_compute(stacks, null_space=[False, True], elements=[1, len(bottom[2]) or 1]),
                 [(1., 2, -1), (3., -1, 1)])


def test_custom_points_a_null_bottom_stack_ignores_are_the_basement():
    custom = [[5., 1., 1.6], [5., 1., 1.7], [5., 1., 1.8], [5., 1., 1.9]]
    stacks = [(R.ERODE, None, [3.]), (R.BASEMENT, _plane(1.), [])]
    custom_ids = lambda **kwargs: np.rint(BackendTensor.t.to_numpy(_compute(
        stacks, null_space=[False, True], elements=[1, 1], custom=custom, **kwargs).raw_arrays.custom)).astype(int)
    np.testing.assert_array_equal(custom_ids(), [-1, -1, -1, -1])
    np.testing.assert_array_equal(custom_ids(ignored=[(), ("custom_grid",)]), [2, 2, 2, 2])
    np.testing.assert_array_equal(custom_ids(ignored=[(), (("custom_grid", 0, 2),)]), [2, 2, -1, -1])


@pytest.mark.parametrize("overlap", ["none", "pretty", "joint", "joint_contacts"])
def test_a_null_bottom_stack_meshes_the_others_like_a_lithology_bottom(overlap):
    stacks = [(R.ERODE, None, [3.]), (R.ERODE, None, [2.]), (R.BASEMENT, _plane(1.), [])]
    null = _compute(stacks, null_space=[False, False, True], mesh=True, overlap=overlap, elements=[1, 1, 1])
    lithology = _compute(stacks, mesh=True, overlap=overlap, elements=[1, 1, 1])
    others = [mesh for mesh in lithology.dc_meshes if mesh.stack_index != 2]
    assert [m.stack_index for m in null.dc_meshes] == [m.stack_index for m in others] == [0, 1]
    for a, b in zip(null.dc_meshes, others):
        np.testing.assert_array_equal(a.vertices, b.vertices)
        np.testing.assert_array_equal(a.edges, b.edges)

