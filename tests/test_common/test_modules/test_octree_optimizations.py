from unittest.mock import patch

import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.generic_grid import GenericGrid
from gempy_engine.core.data.internal_structs import SolverInput, EvaluatorInput
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.API.interp_single._octree_generation import _generate_corners
from gempy_engine.API.interp_single._interp_scalar_field import _deduplicate_corners, _evaluate_sys_eq, _solve_interpolation
from gempy_engine.API.interp_single._interp_single_feature import input_preprocess
from gempy_engine.modules.octrees_topology._octree_common import _generate_next_level_centers
from tests.fixtures.simple_models import simple_model_interpolation_input_factory


@pytest.fixture(params=['numpy', 'PYTORCH'])
def backend(request):
    if request.param == 'PYTORCH':
        pytest.importorskip('torch')
    old = (BackendTensor.engine_backend, BackendTensor.use_gpu, BackendTensor.dtype, BackendTensor.use_pykeops,
           BackendTensor.COMPUTE_GRADS)
    BackendTensor._change_backend(engine_backend=AvailableBackends[request.param], use_gpu=False,
                                 dtype='float64', use_pykeops=False, grads=True)
    yield request.param
    BackendTensor._change_backend(engine_backend=old[0], use_gpu=old[1], dtype=old[2], use_pykeops=old[3], grads=old[4])


def corner_grid(sparse=False):
    t = BackendTensor.t
    root = RegularGrid([10, 13, -4, -2, 20, 24], [3, 2, 2])
    if sparse:
        active = t.array([True, True] + [False] * 10, dtype=bool)
        xyz, bits = _generate_next_level_centers(root.values[active], root.dxdydz)
        root = RegularGrid.from_octree_level(xyz, root, active, bits)
    corners = _generate_corners(root)
    return EngineGrid(octree_grid=root, corners_grid=GenericGrid(corners),
                      custom_grid=GenericGrid(corners[:2]))


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('faulted', [False, True])
def test_corner_mapping(backend, sparse, faulted):
    t = BackendTensor.t
    grid = corner_grid(sparse)
    xyz = t.concatenate((grid.values, grid.corners_grid.values[:3]))
    faults = FaultsData(xyz[:, 0][None, :], xyz[-3:, 0][None, :]) if faulted else None
    original = SolverInput(None, None, xyz, faults)
    reduced, inverse = _deduplicate_corners(original, grid)
    assert inverse is not None
    assert len(reduced.xyz_to_interpolate) < len(xyz)
    np.testing.assert_allclose(t.to_numpy(reduced.xyz_to_interpolate[inverse]), t.to_numpy(xyz), atol=1e-13, rtol=0)
    start = grid.corners_grid_slice.start
    np.testing.assert_array_equal(t.to_numpy(reduced.xyz_to_interpolate[:start]), t.to_numpy(xyz[:start]))
    np.testing.assert_array_equal(t.to_numpy(reduced.xyz_to_interpolate[-3:]), t.to_numpy(xyz[-3:]))
    assert original.xyz_to_interpolate is xyz
    if faulted:
        np.testing.assert_array_equal(t.to_numpy(reduced.fault_internal.fault_values_everywhere[:, inverse]),
                                      t.to_numpy(faults.fault_values_everywhere))
        assert original.fault_internal is faults
        # Different fault values at shared corners must not be merged.
        faults.fault_values_everywhere = BackendTensor.arange(len(xyz), dtype='float64')[None, :]
        assert _deduplicate_corners(original, grid)[1] is None


def test_corner_fallbacks(backend):
    t = BackendTensor.t
    grid = corner_grid()
    original = SolverInput(None, None, grid.values)
    assert _deduplicate_corners(original, None)[1] is None
    grid.corners_grid.values = grid.corners_grid.values[:0]
    assert _deduplicate_corners(original, grid)[1] is None
    grid = corner_grid()
    grid.corners_grid.values[0] += 0.1
    original.xyz_to_interpolate = grid.values
    # The first corner is unique; alter a corner that is shared instead.
    grid.corners_grid.values[7] += 0.1
    original.xyz_to_interpolate = grid.values
    assert _deduplicate_corners(original, grid)[1] is None
    if backend == 'PYTORCH':
        grid = corner_grid()
        grid.corners_grid.values.requires_grad_()
        assert _deduplicate_corners(original, grid)[1] is None
        grid.corners_grid.values = grid.corners_grid.values.detach()
        original.fault_internal = FaultsData(t.ones((1, len(original.xyz_to_interpolate))).requires_grad_(), t.ones((1, 3)))
        assert _deduplicate_corners(original, grid)[1] is None


@pytest.mark.parametrize('flat_input', [False, True])
@pytest.mark.parametrize('symbolic', [False, True])
def test_evaluation_and_gradient_parity(backend, flat_input, symbolic, monkeypatch):
    if symbolic:
        pytest.importorskip('pykeops')
        monkeypatch.setattr(BackendTensor, 'use_pykeops', True)
    t = BackendTensor.t
    interp, options, descriptor = simple_model_interpolation_input_factory()
    grid = interp.grid
    grid.corners_grid = GenericGrid(_generate_corners(grid.octree_grid))
    options.evaluation_options.compute_scalar_gradient = True
    options.evaluation_options.evaluation_chunk_size = 300
    if backend == 'PYTORCH':
        interp.surface_points.sp_coords.requires_grad_()
    solver = input_preprocess(descriptor.tensors_structure, interp)
    weights = _solve_interpolation(solver, options.kernel_options)
    eval_input = EvaluatorInput(solver, interp, descriptor.tensors_structure) if flat_input else solver
    original_xyz = eval_input.xyz_to_interpolate
    legacy = _evaluate_sys_eq(eval_input, weights, options, grid)
    options.evaluation_options.deduplicate_octree_corners = True
    reduced, inverse = _deduplicate_corners(eval_input, grid)
    assert inverse is not None
    optimized = _evaluate_sys_eq(eval_input, weights, options, grid)
    assert eval_input.xyz_to_interpolate is original_xyz
    for name in ('_scalar_field', '_gx_field', '_gy_field', '_gz_field'):
        a, b = getattr(legacy, name), getattr(optimized, name)
        np.testing.assert_allclose(t.to_numpy(a), t.to_numpy(b), atol=1e-10, rtol=1e-10)
    for result in (legacy, optimized):
        result.set_structure_values(descriptor.tensors_structure.reference_sp_position, interp.slice_feature, grid.len_all_grids)
    np.testing.assert_allclose(t.to_numpy(legacy.scalar_field_at_surface_points), t.to_numpy(optimized.scalar_field_at_surface_points))
    if backend == 'PYTORCH':
        import torch
        coefficients = torch.arange(len(original_xyz), dtype=weights.dtype) + 1
        def loss(result):
            return sum((getattr(result, name) * coefficients).sum()
                       for name in ('_scalar_field', '_gx_field', '_gy_field', '_gz_field'))
        targets = (weights, interp.surface_points.sp_coords)
        a = torch.autograd.grad(loss(legacy), targets, retain_graph=True)
        b = torch.autograd.grad(loss(optimized), targets)
        for old, new in zip(a, b):
            torch.testing.assert_close(old, new, atol=1e-7, rtol=1e-8)


@pytest.mark.parametrize('flat', [False, True])
def test_model_parity(backend, flat, monkeypatch):
    from gempy_engine.API.model.model_api import compute_model
    from gempy_engine.API.interp_single import _multi_scalar_field_manager as manager

    monkeypatch.setenv('GEMPY_FLAT_STACKS', 'False')
    interp, options, descriptor = simple_model_interpolation_input_factory()
    options.evaluation_options.number_octree_levels = 2
    legacy = compute_model(interp, options, descriptor)
    interp, options, descriptor = simple_model_interpolation_input_factory()
    options.evaluation_options.number_octree_levels = 2
    options.evaluation_options.deduplicate_octree_corners = True
    monkeypatch.setenv('GEMPY_FLAT_STACKS', str(flat))
    if flat:
        # Public flat dispatch requires PyKeOps; exercise the same stack manager
        # with dense per-stack evaluation here without requiring the JIT compiler.
        monkeypatch.setattr(manager, '_interpolate_stack', manager._interpolate_stack_flat)
    optimized = compute_model(interp, options, descriptor)
    t = BackendTensor.t
    for old_level, new_level in zip(legacy.octrees_output, optimized.octrees_output):
        np.testing.assert_allclose(t.to_numpy(old_level.grid.values), t.to_numpy(new_level.grid.values))
        for old, new in zip(old_level.outputs, new_level.outputs):
            np.testing.assert_allclose(t.to_numpy(old.exported_fields.scalar_field_everywhere),
                                       t.to_numpy(new.exported_fields.scalar_field_everywhere), atol=1e-9)
    assert len(legacy.dc_meshes) == len(optimized.dc_meshes)
    for old, new in zip(legacy.dc_meshes, optimized.dc_meshes):
        np.testing.assert_allclose(old.vertices, new.vertices, atol=1e-8)
        np.testing.assert_array_equal(old.edges, new.edges)


@pytest.mark.parametrize('flat', [False, True])
def test_fault_model_parity(backend, flat, monkeypatch):
    from gempy_engine.API.interp_single import _multi_scalar_field_manager as manager
    from gempy_engine.API.model.model_api import compute_model
    from tests.fixtures.complex_geometries import graben_fault_model

    monkeypatch.setenv('GEMPY_FLAT_STACKS', 'False')
    results = []
    for optimized in (False, True):
        interp, descriptor, options = graben_fault_model.__wrapped__()
        options.evaluation_options.number_octree_levels = 2
        options.evaluation_options.mesh_extraction = False
        options.evaluation_options.deduplicate_octree_corners = optimized
        if flat and optimized:
            monkeypatch.setattr(manager, '_interpolate_stack', manager._interpolate_stack_flat)
        results.append(compute_model(interp, options, descriptor))
    t = BackendTensor.t
    for old_level, new_level in zip(results[0].octrees_output, results[1].octrees_output):
        for old, new in zip(old_level.outputs, new_level.outputs):
            np.testing.assert_allclose(t.to_numpy(old.exported_fields.scalar_field_everywhere),
                                       t.to_numpy(new.exported_fields.scalar_field_everywhere), atol=1e-8)
            np.testing.assert_allclose(t.to_numpy(old.scalar_fields.values_block),
                                       t.to_numpy(new.scalar_fields.values_block), atol=1e-8)


def test_empty_evaluation_and_isolated_cell(backend):
    t = BackendTensor.t
    options = InterpolationOptions.from_args(range=1., c_o=1.)
    options.evaluation_options.compute_scalar_gradient = True
    options.evaluation_options.deduplicate_octree_corners = True
    empty = SolverInput(None, None, t.zeros((0, 3)))
    result = _evaluate_sys_eq(empty, t.ones(1), options)
    assert result.scalar_field_everywhere.shape == (0,)
    assert result.gx_field_everywhere.shape == (0,)
    grid = corner_grid()
    grid.octree_grid._integer_coordinates = grid.octree_grid.integer_coordinates[:1]
    grid.octree_grid.values = grid.octree_grid.values[:1]
    grid.corners_grid.values = grid.corners_grid.values[:8]
    original = SolverInput(None, None, grid.values)
    assert _deduplicate_corners(original, grid)[1] is None


def test_public_pykeops_flat_parity(backend, monkeypatch):
    pytest.importorskip('pykeops')
    from gempy_engine.API.model.model_api import compute_model
    from gempy_engine.API.interp_single import _stack_ops

    monkeypatch.setattr(BackendTensor, 'use_pykeops', True)
    monkeypatch.setenv('GEMPY_FLAT_STACKS', 'True')
    results = []
    with patch.object(_stack_ops, '_evaluate_optimized', wraps=_stack_ops._evaluate_optimized) as evaluate:
        for optimized in (False, True):
            interp, options, descriptor = simple_model_interpolation_input_factory()
            options.evaluation_options.number_octree_levels = 2
            options.evaluation_options.mesh_extraction = False
            options.evaluation_options.deduplicate_octree_corners = optimized
            results.append(compute_model(interp, options, descriptor))
        assert evaluate.call_count == 4
    t = BackendTensor.t
    for old, new in zip(results[0].octrees_output, results[1].octrees_output):
        np.testing.assert_allclose(t.to_numpy(old.outputs[0].exported_fields.scalar_field_everywhere),
                                   t.to_numpy(new.outputs[0].exported_fields.scalar_field_everywhere), atol=1e-9)


def test_selectors_serialization():
    options = InterpolationOptions.from_args(range=1., c_o=1.)
    assert not options.evaluation_options.deduplicate_octree_corners
    options.evaluation_options.deduplicate_octree_corners = True
    restored = InterpolationOptions.model_validate_json(options.model_dump_json())
    assert restored.evaluation_options.deduplicate_octree_corners
