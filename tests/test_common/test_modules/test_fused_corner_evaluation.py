from unittest.mock import patch

import numpy as np
import pytest

from gempy_engine.API.interp_single import _stack_ops
from gempy_engine.API.interp_single._octree_generation import _generate_corners
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.generic_grid import GenericGrid
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.modules.evaluator import symbolic_evaluator as symbolic
from tests.fixtures.simple_models import simple_model_interpolation_input_factory
from tests.test_common.test_modules.test_octree_optimizations import backend


FIELDS = ('_scalar_field', '_gx_field', '_gy_field', '_gz_field')


@pytest.mark.parametrize('case', ['plain', 'faults', 'unequal_faults', 'corner_grad', 'fault_grad'])
@pytest.mark.parametrize('mode', ['scalar', 'gradient', 'combined'])
def test_fused_corner_evaluation(backend, case, mode, monkeypatch):
    pytest.importorskip('pykeops')
    if case in ('corner_grad', 'fault_grad') and backend != 'PYTORCH':
        pytest.skip('Autograd safeguards require torch')
    monkeypatch.setattr(BackendTensor, 'use_pykeops', True)
    t = BackendTensor.t
    inputs, solvers, structs, options = [], [], [], []
    targets = []
    for i in range(3):
        interp, opt, descriptor = simple_model_interpolation_input_factory()
        root = RegularGrid([0, 1, 0, 1, 0, 1], [i + 2, 2, 2])
        corners = _generate_corners(root)
        if case == 'corner_grad' and i == 2:
            corners.requires_grad_()
            targets.append(corners)
        grid = EngineGrid(octree_grid=root, corners_grid=GenericGrid(corners),
                          custom_grid=GenericGrid(t.array([[0.2, 0.3, 0.4]])))
        interp.set_temp_grid(grid)
        if backend == 'PYTORCH':
            interp.surface_points.sp_coords.requires_grad_()
            targets.append(interp.surface_points.sp_coords)
        if case in ('faults', 'unequal_faults', 'fault_grad'):
            xyz = t.concatenate((grid.values, interp.all_surface_points.sp_coords))
            values = xyz[:, 0][None, :] ** 2
            if backend == 'PYTORCH':
                values = values.detach()
            if case == 'unequal_faults' and i == 2:
                values = BackendTensor.arange(len(xyz), dtype='float64')[None, :] / len(xyz)
            if case == 'fault_grad' and i == 2:
                values.requires_grad_()
                # Spatial-gradient kernels do not depend on the fault values.
                if mode != 'gradient':
                    targets.append(values)
            interp.fault_values = FaultsData(values, values[:, -interp.surface_points.n_points:])
        # Third stack is ineligible in the plain case, independently of its option.
        if case == 'plain' and i == 2:
            grid.corners_grid.values[7] += 0.01
        opt.evaluation_options.compute_scalar_gradient = mode != 'scalar'
        opt.evaluation_options.compute_scalar = mode != 'gradient'
        solver = _stack_ops.input_preprocess_v2(descriptor.tensors_structure, interp)
        n = (solver.ori_internal.n_orientations_tiled + solver.sp_internal.n_points
             + opt.kernel_options.n_uni_eq + solver.fault_internal.n_faults)
        solver.weights_x0 = t.array(np.linspace(0.1 + i, 1.1 + i, n))
        if backend == 'PYTORCH':
            solver.weights_x0.requires_grad_()
            targets.append(solver.weights_x0)
        inputs.append(interp)
        solvers.append(solver)
        structs.append(descriptor.tensors_structure)
        options.append(opt)

    args = dict(interpolation_inputs=inputs, options=options[0], solver_inputs=solvers,
                stack_structure=descriptor.stack_structure, tensor_structs=structs,
                stack_indices=[0, 1, 2], options_per_stack=options)
    _, expected = _stack_ops._evaluate_optimized(**args)
    _, serial = _stack_ops._evaluate(**args)
    fields = FIELDS if mode == 'combined' else FIELDS[1:] if mode == 'gradient' else FIELDS[:1]
    for fused_result, serial_result in zip(expected, serial):
        for name in fields:
            np.testing.assert_allclose(t.to_numpy(getattr(fused_result, name)),
                                       t.to_numpy(getattr(serial_result, name)), atol=1e-9, rtol=1e-9)
    for i, opt in enumerate(options):
        opt.evaluation_options.deduplicate_octree_corners = i != 1

    if backend == 'numpy':
        from pykeops.numpy import LazyTensor
    else:
        from pykeops.torch import LazyTensor
    reductions = []
    lazy_sum = LazyTensor.sum

    def record_sum(self, *args, **kwargs):
        if kwargs.get('axis') == 0:
            reductions.append(kwargs)
        return lazy_sum(self, *args, **kwargs)

    with patch.object(symbolic, 'symbolic_evaluator_optimized_stacked',
                      wraps=symbolic.symbolic_evaluator_optimized_stacked) as fused, \
            patch.object(LazyTensor, 'sum', new=record_sum), \
            patch.object(_stack_ops, '_evaluate', side_effect=AssertionError('Must stay fused')):
        originals, actual = _stack_ops._evaluate_optimized(**args)
    assert fused.call_count == 1
    assert len(reductions) == 1
    assert reductions[0]['backend'] == 'CPU'
    assert reductions[0]['ranges'] is not None
    assert fused.call_args.kwargs['options_list'] is options
    reduced = fused.call_args.kwargs['eval_inputs']
    assert len({len(e.xyz_to_interpolate) for e in reduced}) == 3
    for i, (original, view, result, reference) in enumerate(zip(originals, reduced, actual, expected)):
        eligible = i == 0 or (i == 2 and case == 'faults')
        assert (len(view.xyz_to_interpolate) < len(original.xyz_to_interpolate)) == eligible
        assert (view is not original) == eligible
        assert original.solver_input is solvers[i]
        assert fused.call_args.kwargs['weights_list'][i] is solvers[i].weights_x0
        assert view.sp_internal is original.sp_internal
        assert view.ori_internal is original.ori_internal
        if original.fault_internal.n_faults:
            assert original.fault_internal is inputs[i].fault_values
            assert original.fault_internal.fault_values_everywhere.shape[1] == len(original.xyz_to_interpolate)
            assert view.fault_internal.fault_values_everywhere.shape[1] == len(view.xyz_to_interpolate)
            assert (view.fault_internal is not original.fault_internal) == eligible
        np.testing.assert_array_equal(t.to_numpy(original.xyz_to_interpolate),
                                      t.to_numpy(t.concatenate((inputs[i].grid.values,
                                                               inputs[i].all_surface_points.sp_coords))))
        assert result.grid_size == inputs[i].grid.len_all_grids
        assert result.n_points_per_surface is original._n_points_per_surface
        assert result.slice_feature is original._slice_feature
        assert result.debug is solvers[i].debug
        for name in FIELDS:
            a, b = getattr(result, name), getattr(reference, name)
            if b is None:
                assert a is None
            else:
                np.testing.assert_allclose(t.to_numpy(a), t.to_numpy(b), atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(t.to_numpy(result.scalar_field_at_surface_points),
                                   t.to_numpy(reference.scalar_field_at_surface_points), atol=1e-9, rtol=1e-9)

    if backend == 'PYTORCH':
        import torch

        def loss(results):
            return sum((getattr(result, name) * (torch.arange(len(result._scalar_field),
                                                             dtype=torch.float64) + 1)).sum()
                       for result in results for name in fields)

        old_grads = torch.autograd.grad(loss(expected), targets, retain_graph=True)
        serial_grads = torch.autograd.grad(loss(serial), targets, retain_graph=True)
        new_grads = torch.autograd.grad(loss(actual), targets)
        for old, new, reference in zip(old_grads, new_grads, serial_grads):
            torch.testing.assert_close(old, new, atol=1e-7, rtol=1e-8)
            torch.testing.assert_close(new, reference, atol=1e-7, rtol=1e-8)


@pytest.mark.parametrize('deduplicate', [False, True])
def test_non_pykeops_dispatch(backend, deduplicate):
    interp, options, descriptor = simple_model_interpolation_input_factory()
    options.evaluation_options.deduplicate_octree_corners = deduplicate
    with patch.object(_stack_ops, '_evaluate', return_value=([], [])) as evaluate, \
            patch.object(symbolic, 'symbolic_evaluator_optimized_stacked',
                         side_effect=AssertionError('PyKeOps is disabled')):
        assert _stack_ops._evaluate_optimized([interp], options, [], descriptor.stack_structure,
                                              [], [0]) == ([], [])
    assert evaluate.call_count == 1
