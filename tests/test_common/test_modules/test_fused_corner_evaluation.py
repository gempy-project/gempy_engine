from unittest.mock import patch

import numpy as np
import pytest

from gempy_engine.API.interp_single import _stack_ops
from gempy_engine.API.interp_single._octree_generation import _generate_corners
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.generic_grid import GenericGrid
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.modules.evaluator import symbolic_evaluator as symbolic
from tests.fixtures.simple_models import simple_model_interpolation_input_factory
from tests.test_common.test_modules.test_octree_optimizations import backend


def test_fused_corner_evaluation_uses_reduced_views(backend, monkeypatch):
    pytest.importorskip('pykeops')
    monkeypatch.setattr(BackendTensor, 'use_pykeops', True)
    t = BackendTensor.t
    inputs, solvers, structs, options = [], [], [], []
    for i in range(3):
        interp, opt, descriptor = simple_model_interpolation_input_factory()
        root = RegularGrid([10, 13, -4, -2, 20, 24], [i + 2, 2, 2])
        corners = _generate_corners(root)
        grid = EngineGrid(octree_grid=root, corners_grid=GenericGrid(corners),
                          custom_grid=GenericGrid(corners[:2]))
        interp.set_temp_grid(grid)
        opt.evaluation_options.compute_scalar = True
        opt.evaluation_options.compute_scalar_gradient = False
        opt.evaluation_options.deduplicate_octree_corners = True
        solver = _stack_ops.input_preprocess_v2(descriptor.tensors_structure, interp)
        n = (solver.ori_internal.n_orientations_tiled + solver.sp_internal.n_points
             + opt.kernel_options.n_uni_eq + solver.fault_internal.n_faults)
        solver.weights_x0 = t.array(np.linspace(0.1 + i, 1.1 + i, n))
        inputs.append(interp)
        solvers.append(solver)
        structs.append(descriptor.tensors_structure)
        options.append(opt)

    args = dict(interpolation_inputs=inputs, options=options[0], solver_inputs=solvers,
                stack_structure=descriptor.stack_structure, tensor_structs=structs,
                stack_indices=[0, 1, 2], options_per_stack=options)

    with patch.object(symbolic, 'symbolic_evaluator_optimized_stacked',
                      wraps=symbolic.symbolic_evaluator_optimized_stacked) as fused, \
            patch.object(_stack_ops, '_evaluate', side_effect=AssertionError('Must stay fused')):
        originals, actual = _stack_ops._evaluate_optimized(**args)
    assert fused.call_count == 1
    assert fused.call_args.kwargs['options_list'] is options
    reduced = fused.call_args.kwargs['eval_inputs']
    assert len({len(e.xyz_to_interpolate) for e in reduced}) == 3
    for i, (original, view, result) in enumerate(zip(originals, reduced, actual)):
        assert len(view.xyz_to_interpolate) < len(original.xyz_to_interpolate)
        assert view is not original
        assert original.solver_input is solvers[i]
        assert fused.call_args.kwargs['weights_list'][i] is solvers[i].weights_x0
        assert view.sp_internal is original.sp_internal
        assert view.ori_internal is original.ori_internal
        np.testing.assert_array_equal(t.to_numpy(original.xyz_to_interpolate),
                                      t.to_numpy(t.concatenate((inputs[i].grid.values,
                                                               inputs[i].all_surface_points.sp_coords))))
        assert result.grid_size == inputs[i].grid.len_all_grids
        assert result.n_points_per_surface is original._n_points_per_surface
        assert result.slice_feature is original._slice_feature
        assert result.debug is solvers[i].debug
        assert len(result._scalar_field) == len(original.xyz_to_interpolate)
        assert result._gx_field is None
        assert result._gy_field is None
        assert result._gz_field is None


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
