"""FLAT stacks: fused KeOps evaluation must equal per-stack evaluation.

Regressions covered:
- the scalar kernel marked fault-drift rows as the trailing rows of the whole
  stacked covariance, so every faulted stack but the last lost its fault drift
  (Model 7 stack 1 was off by 13% of its range);
- FLAT chunks mixing stacks with different fault counts (now rejected by the
  stacked builder) and with different per-stack kernel options (silently
  evaluated with the first stack's kernel) are split into fusable groups.
"""

import copy
import os
from types import SimpleNamespace

import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as BT


@pytest.fixture(autouse=True)
def restore_backend():
    saved = dict(engine_backend=BT.engine_backend, use_gpu=BT.use_gpu,
                 use_pykeops=BT.use_pykeops, dtype=BT.dtype, grads=BT.COMPUTE_GRADS)
    yield
    BT._change_backend(**saved)


def _keops_gpu():
    torch = pytest.importorskip('torch')
    pytest.importorskip('pykeops')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    BT._change_backend(AvailableBackends.PYTORCH, use_gpu=True, use_pykeops=True, dtype='float64', grads=False)
    return torch


def _restore(torch):
    BT._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype='float64', grads=False)
    torch.set_default_device('cpu')


def _model7_scalar_fields(flat):
    from gempy_engine.API.model.model_api import compute_model
    from tests.fixtures.model7_combination import model7_combination_factory

    previous = os.environ.get('GEMPY_FLAT_STACKS')
    os.environ['GEMPY_FLAT_STACKS'] = 'True' if flat else 'False'
    try:
        inputs, options, descriptor = model7_combination_factory(number_octree_levels=2, mesh_extraction=False)
        options.evaluation_options.compute_scalar_gradient = False
        solution = compute_model(inputs, options, descriptor)
    finally:
        if previous is None:
            os.environ.pop('GEMPY_FLAT_STACKS')
        else:
            os.environ['GEMPY_FLAT_STACKS'] = previous
    return [BT.t.to_numpy(output.exported_fields.scalar_field) for output in solution.octrees_output[0].outputs]


def test_flat_keops_scalar_fields_match_per_stack_evaluation():
    torch = _keops_gpu()
    try:
        reference = _model7_scalar_fields(flat=False)
        flat = _model7_scalar_fields(flat=True)
    finally:
        _restore(torch)
    for stack, (expected, actual) in enumerate(zip(reference, flat)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10, err_msg=f'stack {stack}')


def test_fused_evaluation_groups_split_by_kernel_flags_and_fault_rows():
    from gempy_engine.API.interp_single._stack_ops import _fused_evaluation_groups

    def options(range_, gradient=False):
        return SimpleNamespace(kernel_options=SimpleNamespace(range=range_),
                               evaluation_options=SimpleNamespace(compute_scalar=True,
                                                                  compute_scalar_gradient=gradient))

    def inputs(n_faults):
        return SimpleNamespace(micro_points=None, fault_values=SimpleNamespace(n_faults=n_faults))

    stacks = [(options(4.), inputs(1)), (options(4.), inputs(2)), (options(2.5), inputs(1)),
              (options(4.), inputs(1)), (options(4., gradient=True), inputs(1))]
    assert _fused_evaluation_groups([o for o, _ in stacks], [i for _, i in stacks]) == [[0, 3], [1], [2], [4]]


def _two_fault_model(per_stack_range):
    """F1, F2 independent faults; A (eroding) cut by F1, B (two surfaces) by both.

    FLAT's second chunk [A, B] mixes 1 and 2 fault rows; with ``per_stack_range``
    F2 has its own kernel range (and curved data, so the range matters), making
    the first chunk [F1, F2] mix kernel options.
    """
    from gempy_engine.core.data import TensorsStructure
    from gempy_engine.core.data.engine_grid import EngineGrid
    from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
    from gempy_engine.core.data.interpolation_input import InterpolationInput
    from gempy_engine.core.data.kernel_classes.orientations import Orientations
    from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
    from gempy_engine.core.data.options import InterpolationOptions
    from gempy_engine.core.data.regular_grid import RegularGrid
    from gempy_engine.core.data.stack_relation_type import StackRelationType as R
    from gempy_engine.core.data.stacks_structure import StacksStructure

    uu, vv = (axis.ravel() for axis in np.meshgrid([-.6, 0., .6], [-.6, 0., .6]))
    surfaces = [np.column_stack((.2 + .1*vv, vv, uu)),
                np.column_stack((uu, -.3 + .1*uu + .25*vv**2, vv)),
                np.column_stack((uu, vv, .3 + .1*uu)),
                np.column_stack((uu, vv, -.2 + .05*vv)),
                np.column_stack((uu, vv, -.5 + .05*vv))]
    gradients = np.array([[1, -.1, 0], [-.1, 1, 0], [-.1, 0, 1], [0, -.05, 1]], dtype=float)
    gradients /= np.linalg.norm(gradients, axis=1, keepdims=True)
    matrix = np.zeros((4, 4), dtype=bool)
    matrix[0, 2] = matrix[0, 3] = matrix[1, 3] = True
    stacks = StacksStructure(np.array([9, 9, 9, 18]), np.array([1, 1, 1, 1]), np.array([1, 1, 1, 2]),
                             [R.FAULT, R.FAULT, R.ERODE, R.BASEMENT], faults_relations=matrix)
    options = InterpolationOptions.from_args(4., 1.)
    options.evaluation_options.number_octree_levels = 2
    options.evaluation_options.mesh_extraction = False
    options.evaluation_options.compute_scalar_gradient = False
    if per_stack_range:
        other = copy.deepcopy(options)
        other.kernel_options.range = 2.5
        stacks.interpolation_options_per_stack = [None, other, None, None]
    descriptor = InputDataDescriptor(TensorsStructure(np.array([9] * 5)), stacks)
    grid = EngineGrid(octree_grid=RegularGrid([-1., 1., -1., 1., -1., 1.], [4, 4, 4]))
    inputs = InterpolationInput(SurfacePoints(np.vstack(surfaces)),
                                Orientations(np.array([[.2, 0, 0], [0, -.3, 0], [0, 0, .3], [0, 0, -.35]]),
                                             gradients), grid)
    return inputs, options, descriptor


def _flat_fields(model, flat):
    from gempy_engine.API.model.model_api import compute_model

    previous = os.environ.get('GEMPY_FLAT_STACKS')
    os.environ['GEMPY_FLAT_STACKS'] = 'True' if flat else 'False'
    try:
        solution = compute_model(*model)
    finally:
        if previous is None:
            os.environ.pop('GEMPY_FLAT_STACKS')
        else:
            os.environ['GEMPY_FLAT_STACKS'] = previous
    return [BT.t.to_numpy(output.exported_fields.scalar_field) for output in solution.octrees_output[0].outputs]


@pytest.mark.parametrize('per_stack_range', [False, True])
def test_flat_keops_mixed_fault_counts_and_kernel_options(per_stack_range):
    torch = _keops_gpu()
    try:
        reference = _flat_fields(_two_fault_model(per_stack_range), flat=False)
        flat = _flat_fields(_two_fault_model(per_stack_range), flat=True)
    finally:
        _restore(torch)
    for stack, (expected, actual) in enumerate(zip(reference, flat)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10, err_msg=f'stack {stack}')
