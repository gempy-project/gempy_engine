from types import SimpleNamespace

import numpy as np
import pytest

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.modules.evaluator import symbolic_evaluator as symbolic
from gempy_engine.modules.kernel_constructor._kernels_assembler import create_scalar_kernel
from gempy_engine.modules.kernel_constructor._structs import (
    CartesianSelector, DriftMatrixSelector, FaultDrift, KernelInput,
    OrientationSurfacePointsCoords, OrientationsDrift, PointsDrift,
)
from tests.test_common.test_modules.test_octree_optimizations import backend


def kernel_input(n_constraints, n_grid, seed, fault=False):
    t = BackendTensor.t
    rng = np.random.default_rng(seed)
    x = t.array(rng.random((n_constraints, 3)))
    grid = t.array(rng.random((n_grid, 3)))
    i_selector = t.ones((n_constraints, 3))
    j_selector = t.ones((n_grid, 3))
    return KernelInput(
        ori_sp_matrices=OrientationSurfacePointsCoords(x, grid, x + 0.1, grid),
        cartesian_selector=CartesianSelector(
            i_selector, j_selector, i_selector, j_selector,
            i_selector, j_selector, i_selector, j_selector,
        ),
        nugget_scalar=None,
        nugget_grad=None,
        ori_drift=OrientationsDrift(x, grid, x, grid, x, grid, i_selector),
        ref_drift=PointsDrift(x, grid, x, grid, x, grid),
        rest_drift=PointsDrift(x, grid, x, grid, x, grid),
        drift_matrix_selector=DriftMatrixSelector(
            n_constraints, n_grid, 1, n_constraints - 1, n_grid,
        ),
        ref_fault=FaultDrift(x[:, :1], grid[:, :1]) if fault else None,
        rest_fault=None,
    )


@pytest.mark.parametrize('order', [(0, 1, 2), (1, 2, 0), (2, 0, 1)])
def test_stacked_singletons_match_independent_evaluation(backend, monkeypatch, order):
    pytest.importorskip('pykeops')
    t = BackendTensor.t
    sizes = [(1, 4), (3, 1), (2, 5)]
    kernels = [kernel_input(n, m, i) for i, (n, m) in enumerate(sizes)]
    weights = [t.array(np.linspace(0.2, 0.8, n)) for n, _ in sizes]
    options = InterpolationOptions.from_args(range=2, c_o=1)
    options.evaluation_options.compute_scalar = True
    options.evaluation_options.compute_scalar_gradient = False
    expected = [t.to_numpy(t.sum(create_scalar_kernel(ki, options.kernel_options) * w[:, None], axis=0))
                for ki, w in zip(kernels, weights)]

    monkeypatch.setattr(symbolic, 'evaluation_vectors_preparations',
                        lambda ei, *args, **kwargs: ei.kernel_data)
    monkeypatch.setattr(symbolic, '_stacked_failure_details',
                        lambda *args: pytest.fail('Successful evaluations must not collect diagnostics'))
    inputs = [SimpleNamespace(kernel_data=kernels[i], xyz_to_interpolate=t.zeros((sizes[i][1], 3)))
              for i in order]
    actual = symbolic.symbolic_evaluator_optimized_stacked(
        inputs, [weights[i] for i in order], [options] * len(order),
    )
    for result, i in zip(actual, order):
        np.testing.assert_allclose(t.to_numpy(result._scalar_field), expected[i], rtol=1e-10, atol=1e-10)

    # Chunk boundaries must not change the result either.
    split = symbolic.symbolic_evaluator_optimized_stacked(inputs[:1], [weights[order[0]]], [options])
    split += symbolic.symbolic_evaluator_optimized_stacked(
        inputs[1:], [weights[i] for i in order[1:]], [options] * 2,
    )
    for result, i in zip(split, order):
        np.testing.assert_allclose(t.to_numpy(result._scalar_field), expected[i], rtol=1e-10, atol=1e-10)


def test_stacked_fault_singletons_keep_their_axes(backend):
    pytest.importorskip('pykeops')
    stacked = symbolic._build_stacked_kernel_data([
        kernel_input(1, 4, 0, fault=True), kernel_input(3, 1, 1, fault=True),
    ])
    assert stacked.ref_fault.faults_i.ni == 4
    assert stacked.ref_fault.faults_i.nj is None
    assert stacked.ref_fault.faults_j.nj == 5
    assert stacked.ref_fault.faults_j.ni is None
    assert stacked.drift_matrix_selector.sel_ui.ni == 4
    assert stacked.drift_matrix_selector.sel_vj.nj == 5


def test_stacked_empty_grid_keeps_the_j_axis(backend):
    pytest.importorskip('pykeops')
    stacked = symbolic._build_stacked_kernel_data([
        kernel_input(2, 0, 0), kernel_input(3, 5, 1),
    ])
    assert stacked.ori_sp_matrices.dip_ref_i.ni == 5
    assert stacked.ori_sp_matrices.dip_ref_j.nj == 5
    assert stacked.cartesian_selector.hu_sel_j.nj == 5
    assert stacked.ref_drift.dipsPoints_ui_aj.nj == 5
    assert stacked.drift_matrix_selector.sel_vj.nj == 5


def test_stacked_mismatched_grid_field_fails_before_kernel_construction(backend, monkeypatch):
    pytest.importorskip('pykeops')
    kernels = [kernel_input(2, 4, 0), kernel_input(3, 5, 1)]
    kernels[1].ref_drift.dipsPoints_ui_aj = kernels[1].ref_drift.dipsPoints_ui_aj[:, :2, :]
    monkeypatch.setattr(symbolic, 'evaluation_vectors_preparations',
                        lambda ei, *args, **kwargs: ei.kernel_data)
    monkeypatch.setattr(symbolic, 'create_scalar_kernel',
                        lambda *args, **kwargs: pytest.fail('Invalid fields must fail before kernel construction'))
    t = BackendTensor.t
    inputs = [SimpleNamespace(kernel_data=ki, xyz_to_interpolate=t.zeros((m, 3)))
              for ki, m in zip(kernels, (4, 5))]
    options = InterpolationOptions.from_args(range=2, c_o=1)
    with pytest.raises(RuntimeError, match=r'ref_drift\.dipsPoints_ui_aj in group 1:.*expected size=5'):
        symbolic.symbolic_evaluator_optimized_stacked(
            inputs, [t.ones(2), t.ones(3)], [options] * 2,
        )


@pytest.mark.parametrize('fault_first', [False, True])
def test_stacked_mixed_fault_presence_is_rejected(backend, fault_first):
    pytest.importorskip('pykeops')
    kernels = [kernel_input(2, 4, 0, fault=fault_first),
               kernel_input(3, 5, 1, fault=not fault_first)]
    with pytest.raises(RuntimeError, match='ref_fault: present in only some groups'):
        symbolic._build_stacked_kernel_data(kernels)


def test_lazy_nj_failure_captures_raw_and_stacked_metadata(backend, monkeypatch):
    pytest.importorskip('pykeops')
    if backend == 'numpy':
        from pykeops.numpy import LazyTensor
    else:
        from pykeops.torch import LazyTensor
    t = BackendTensor.t
    kernels = [kernel_input(2, 4, 0), kernel_input(3, 5, 1)]
    inputs = [SimpleNamespace(kernel_data=ki, xyz_to_interpolate=t.ones((m, 3)) * 9876543)
              for ki, m in zip(kernels, (4, 5))]
    options = InterpolationOptions.from_args(range=2, c_o=1)
    monkeypatch.setattr(symbolic, 'evaluation_vectors_preparations',
                        lambda ei, *args, **kwargs: ei.kernel_data)
    original_build = symbolic._build_stacked_kernel_data

    def mismatched_lazy_field(items):
        stacked = original_build(items)
        stacked.ref_drift.dipsPoints_ui_aj = LazyTensor(t.array(np.ones((1, 3, 3))))
        return stacked

    monkeypatch.setattr(symbolic, '_build_stacked_kernel_data', mismatched_lazy_field)
    with pytest.raises(RuntimeError) as error:
        symbolic.symbolic_evaluator_optimized_stacked(inputs, [t.ones(2), t.ones(3)], [options] * 2)
    message = str(error.value)
    assert 'stage=create_scalar_kernel' in message
    assert 'Incompatible values for attribute nj' in message
    assert 'M_sizes=[4, 5]' in message
    assert 'N_sizes=[2, 3]' in message
    assert 'group 1: weights=' in message
    assert 'prepared[1].ref_drift.dipsPoints_ui_aj:' in message
    assert 'stacked[0].ref_drift.dipsPoints_ui_aj:' in message
    assert "'nj': 3" in message
    assert 'kernel_function' in message
    assert 'pykeops_enabled=' in message
    assert 'pykeops=' in message
    assert '9876543' not in message
    assert isinstance(error.value.__cause__, ValueError)


def test_threaded_preparation_failure_identifies_group(backend, monkeypatch):
    t = BackendTensor.t
    inputs = [SimpleNamespace(xyz_to_interpolate=t.ones((m, 3))) for m in (4, 5)]
    failure = ValueError('intermittent preparation failure')

    def fail(ei, *args, **kwargs):
        if ei is inputs[1]:
            raise failure
        return kernel_input(2, 4, 0)

    monkeypatch.setattr(symbolic, 'evaluation_vectors_preparations', fail)
    options = InterpolationOptions.from_args(range=2, c_o=1)
    with pytest.raises(RuntimeError) as error:
        symbolic.symbolic_evaluator_optimized_stacked(inputs, [t.ones(2), t.ones(3)], [options] * 2)
    assert 'stage=prepare_scalar_inputs' in str(error.value)
    assert 'group=1, gradient_axis=None' in str(error.value)
    assert error.value.__cause__.__cause__ is failure


def test_reduction_failure_captures_ranges_and_weights(backend, monkeypatch):
    pytest.importorskip('pykeops')
    if backend == 'numpy':
        from pykeops.numpy import LazyTensor
    else:
        from pykeops.torch import LazyTensor
    t = BackendTensor.t
    ki = kernel_input(2, 4, 0)
    inputs = [SimpleNamespace(kernel_data=ki, xyz_to_interpolate=t.ones((4, 3)))]
    monkeypatch.setattr(symbolic, 'evaluation_vectors_preparations',
                        lambda ei, *args, **kwargs: ei.kernel_data)
    failure = ValueError('intermittent reduction failure')
    original_sum = LazyTensor.sum

    def fail_reduction(self, *args, **kwargs):
        if 'ranges' in kwargs:
            raise failure
        return original_sum(self, *args, **kwargs)

    monkeypatch.setattr(LazyTensor, 'sum', fail_reduction)
    options = InterpolationOptions.from_args(range=2, c_o=1)
    with pytest.raises(RuntimeError) as error:
        symbolic.symbolic_evaluator_optimized_stacked(inputs, [t.array(np.ones(2))], [options])
    message = str(error.value)
    assert 'stage=weighted_reduction' in message
    assert 'eval_kernel:' in message
    assert 'lazy_weights:' in message
    assert 'all_weights:' in message
    assert 'ranges:' in message
    assert 'tile_factor=1' in message
    assert error.value.__cause__ is failure


def test_diagnostic_failure_does_not_hide_original_error(backend, monkeypatch):
    t = BackendTensor.t
    failure = ValueError('original shape mismatch')

    def fail(*args, **kwargs):
        raise failure

    def fail_diagnostics(*args, **kwargs):
        raise RuntimeError('snapshot unavailable')

    monkeypatch.setattr(symbolic, 'evaluation_vectors_preparations', fail)
    monkeypatch.setattr(symbolic, '_stacked_failure_details', fail_diagnostics)
    options = InterpolationOptions.from_args(range=2, c_o=1)
    with pytest.raises(RuntimeError) as error:
        symbolic.symbolic_evaluator_optimized_stacked(
            [SimpleNamespace(xyz_to_interpolate=t.ones((4, 3)))], [t.ones(2)], [options],
        )
    message = str(error.value)
    assert 'original shape mismatch' in message
    assert 'Diagnostic collection failed: RuntimeError: snapshot unavailable' in message
    assert error.value.__cause__.__cause__ is failure


@pytest.mark.parametrize('phase', ['prepare', 'stack'])
def test_gradient_failure_does_not_report_stale_scalar_inputs(backend, monkeypatch, phase):
    pytest.importorskip('pykeops')
    t = BackendTensor.t
    ki = kernel_input(2, 4, 0)
    inputs = [SimpleNamespace(kernel_data=ki, xyz_to_interpolate=t.ones((4, 3)))]
    failure = ValueError('gradient failure')

    def prep(ei, *args, **kwargs):
        if phase == 'prepare' and kwargs['axis'] is not None:
            raise failure
        return ei.kernel_data

    original_build = symbolic._build_stacked_kernel_data

    def stack(items):
        if phase == 'stack' and len(items) == 3:
            raise failure
        return original_build(items)

    monkeypatch.setattr(symbolic, 'evaluation_vectors_preparations', prep)
    monkeypatch.setattr(symbolic, '_build_stacked_kernel_data', stack)
    options = InterpolationOptions.from_args(range=2, c_o=1)
    options.evaluation_options.compute_scalar_gradient = True
    with pytest.raises(RuntimeError) as error:
        symbolic.symbolic_evaluator_optimized_stacked(inputs, [t.array(np.ones(2))], [options])
    message = str(error.value)
    assert f'stage={phase}_gradient_inputs' in message
    assert 'prep_layout=[(0, 0), (0, 1), (0, 2)]' in message
    assert 'eval_kernel_scalar:' in message
    assert 'stacked[0]' not in message
    if phase == 'prepare':
        assert 'prepared[0]' not in message
    else:
        assert 'prepared[2]' in message


def test_tensor_valued_options_are_described_without_values(backend, monkeypatch):
    t = BackendTensor.t
    options = InterpolationOptions.from_args(range=2, c_o=1)
    options.kernel_options.range = t.array([9876543.0])

    def fail(*args, **kwargs):
        raise ValueError('preparation failure')

    monkeypatch.setattr(symbolic, 'evaluation_vectors_preparations', fail)
    with pytest.raises(RuntimeError) as error:
        symbolic.symbolic_evaluator_optimized_stacked(
            [SimpleNamespace(xyz_to_interpolate=t.ones((4, 3)))], [t.ones(2)], [options],
        )
    message = str(error.value)
    assert "'range':" in message
    assert '9876543' not in message
    assert 'Diagnostic collection failed' not in message


def test_unavailable_cuda_metadata_keeps_other_diagnostics(backend, monkeypatch):
    if backend != 'PYTORCH':
        pytest.skip('CUDA metadata applies to the PyTorch backend')
    import torch
    t = BackendTensor.t
    inputs = [SimpleNamespace(xyz_to_interpolate=t.ones((4, 3)))]
    weights = [t.ones(2)]
    options = InterpolationOptions.from_args(range=2, c_o=1)

    def unavailable():
        raise RuntimeError('CUDA context unavailable')

    def forbid_tensor_repr(self):
        pytest.fail('Diagnostics must not format tensor values')

    monkeypatch.setattr(BackendTensor, 'use_gpu', True)
    monkeypatch.setattr(torch.cuda, 'current_device', unavailable)
    monkeypatch.setattr(torch.Tensor, '__repr__', forbid_tensor_repr)
    message = symbolic._stacked_failure_details(inputs, weights, [options], {
        'backend_at_entry': ('PYTORCH', True, 'float64', True, True, False),
        'M_sizes': [4], 'N_sizes': [2],
    })
    assert 'CUDA memory metadata unavailable: CUDA context unavailable' in message
    assert 'group 0: weights=' in message
    assert 'M_sizes=[4]' in message
    assert 'N_sizes=[2]' in message
