"""evaluate_fields / prepare_field_evaluator after compute_model, and classify_units."""

import numpy as np
import pytest

from gempy_engine import classify_units, evaluate_fields, prepare_field_evaluator
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as BT
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine.modules.activator.unit_classification import unit_ids
from tests.fixtures.model7_combination import model7_combination_factory


@pytest.fixture(autouse=True)
def restore_backend():
    saved = dict(engine_backend=BT.engine_backend, use_gpu=BT.use_gpu,
                 use_pykeops=BT.use_pykeops, dtype=BT.dtype, grads=BT.COMPUTE_GRADS)
    yield
    BT._change_backend(**saved)
    try:
        import torch
        torch.set_default_device('cpu')
    except ImportError:
        pass


def _backend(kind):
    if kind == 'numpy':
        BT._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype='float64', grads=False)
        return
    torch = pytest.importorskip('torch')
    if kind != 'torch_cpu' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if kind == 'keops_gpu':
        pytest.importorskip('pykeops')
    BT._change_backend(AvailableBackends.PYTORCH, use_gpu=kind != 'torch_cpu', use_pykeops=kind == 'keops_gpu',
                       dtype='float64', grads=False)


def _host(values):
    return values.detach().cpu().numpy() if hasattr(values, 'detach') else np.asarray(values)


def _model7(levels=3):
    inputs, options, descriptor = model7_combination_factory(number_octree_levels=levels, mesh_extraction=False)
    solutions = compute_model(inputs, options, descriptor)
    return solutions, inputs, options, descriptor


@pytest.mark.parametrize('kind', ['numpy', 'torch_cpu', 'keops_gpu'])
def test_fields_equal_production_scalars_on_every_octree_level(kind):
    _backend(kind)
    solutions, inputs, options, descriptor = _model7()
    evaluator = prepare_field_evaluator(solutions, inputs, descriptor, options)
    assert evaluator.fault_stacks == (0,) and evaluator.affected_by == {0: (), 1: (0,), 2: (0,)}
    # Production KeOps reductions are not bitwise reproducible across batch layouts.
    atol = 1e-8 if kind == 'keops_gpu' else 1e-10
    for level in solutions.octrees_output:
        points = _host(level.grid.values)
        values = evaluator(points)
        assert values.shape == (3, len(points))
        for stack, output in enumerate(level.outputs):
            np.testing.assert_allclose(values[stack], _host(output.exported_fields.scalar_field)[:len(points)],
                                       rtol=0, atol=atol, err_msg=f'stack {stack}')
    subset = evaluator(points, stacks=(2, 0))
    np.testing.assert_allclose(subset, values[[2, 0]], rtol=0, atol=atol)
    one_shot = evaluate_fields(solutions, inputs, descriptor, options, points[:10], stacks=(1,))
    np.testing.assert_allclose(one_shot, values[[1], :10], rtol=0, atol=atol)


@pytest.mark.parametrize('kind', ['numpy', 'keops_gpu'])
def test_side_fixed_fields_match_production_away_from_the_fault_and_throw_across_it(kind):
    _backend(kind)
    solutions, inputs, options, descriptor = _model7()
    evaluator = prepare_field_evaluator(solutions, inputs, descriptor, options)
    points = np.random.default_rng(1).uniform(*_host(solutions.octrees_output[0].grid.octree_grid
                                                      .orthogonal_extent).reshape(3, 2).T, (400, 3))
    offset = evaluator.fault_values(points)
    sides = np.where(offset > 0, 1, -1)
    production = evaluator(points)
    fixed = evaluator(points, fault_sides=sides)
    far = np.abs(offset[0]) > .5*np.abs(offset[0]).max()
    assert far.sum() > 20
    atol = 1e-6 if kind == 'keops_gpu' else 1e-8
    # The fault stack and the stack it does not affect are side-independent.
    np.testing.assert_allclose(fixed[0], production[0], rtol=0, atol=atol)
    # Production drift saturates away from the fault: one-sided values agree there.
    np.testing.assert_allclose(fixed[1:, far], production[1:, far], rtol=0, atol=atol)
    # Across the fault the other side's plateau gives the thrown field.
    other = evaluator(points, fault_sides=-sides)
    assert np.abs(other[1:, far]-fixed[1:, far]).max() > 1e-3
    with pytest.raises(ValueError, match='invalid_fault_sides'):
        evaluator(points, fault_sides=np.zeros_like(sides))
    with pytest.raises(ValueError, match='invalid_fault_sides'):
        evaluator(points, fault_sides=sides[:, :5])


def test_classified_units_match_the_production_lithology_block():
    _backend('numpy')
    solutions, inputs, options, descriptor = _model7()
    evaluator = prepare_field_evaluator(solutions, inputs, descriptor, options)
    last = solutions.octrees_output[-1]
    points = _host(last.grid.values)
    values = evaluator(points)
    stack, interval = classify_units(values, evaluator.stack_relations, evaluator.stack_isovalues)
    assert set(np.unique(stack)) == {1, 2}
    ids = unit_ids(stack, interval, descriptor.stack_structure.number_of_surfaces_per_stack, inputs.unit_values)
    production = np.rint(_host(last.outputs[-1].final_block)[:len(points)]).astype(int)
    # The production block is a sigmoid: compare away from every interface.
    clear = np.all([np.min(np.abs(values[g][:, None]-levels[None]), axis=1) > 1e-3
                    for g, levels in enumerate(evaluator.stack_isovalues)], axis=0)
    assert clear.mean() > .9
    np.testing.assert_array_equal(ids[clear], production[clear])


def test_classify_units_masking_rules():
    isovalues = [np.array([.5]), np.array([.2, .4]), np.array([.3])]
    values = np.array([[.6, .1, .1, .1],
                       [.0, .5, .3, .1],
                       [.0, .0, .0, .9]])
    stack, interval = classify_units(values, [R.ERODE, R.ERODE, R.BASEMENT], isovalues)
    # Above the first eroder; above the second's top; between its surfaces; below it, the basement.
    np.testing.assert_array_equal(stack, [0, 1, 1, 2])
    np.testing.assert_array_equal(interval, [0, 0, 1, 0])
    # Production slices: each stack starts after the surfaces of the stacks above it.
    np.testing.assert_array_equal(unit_ids(stack, interval, [1, 2, 1], np.arange(1, 6)), [1, 2, 3, 4])
    # An onlapping stack is kept where the next (older) stack is above its top surface.
    stack, _ = classify_units(np.array([[.6, .6], [.9, .1]]), [R.ONLAP, R.BASEMENT], [np.array([.5]), np.array([.3])])
    np.testing.assert_array_equal(stack, [0, 1])
    with pytest.raises(ValueError, match='unsupported_relation'):
        classify_units(values, [R.NULL_SPACE, R.ERODE, R.BASEMENT], isovalues)


def test_fault_free_model_evaluates_without_drift():
    from gempy_engine.core.data import TensorsStructure
    from gempy_engine.core.data.engine_grid import EngineGrid
    from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
    from gempy_engine.core.data.interpolation_input import InterpolationInput
    from gempy_engine.core.data.kernel_classes.orientations import Orientations
    from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
    from gempy_engine.core.data.options import InterpolationOptions
    from gempy_engine.core.data.regular_grid import RegularGrid
    from gempy_engine.core.data.stacks_structure import StacksStructure

    _backend('numpy')
    stacks = StacksStructure(np.array([3]), np.array([1]), np.array([1]), [R.BASEMENT])
    descriptor = InputDataDescriptor(TensorsStructure(np.array([3])), stacks)
    grid = EngineGrid(octree_grid=RegularGrid([0., 1., 0., 1., 0., 1.], [4, 4, 4]))
    inputs = InterpolationInput(SurfacePoints(np.array([[.1, .1, .37], [.9, .1, .37], [.1, .9, .37]])),
                                Orientations(np.array([[.5, .5, .37]]), np.array([[0., 0., 1.]])), grid)
    options = InterpolationOptions.from_args(range=2., c_o=1.)
    options.evaluation_options.number_octree_levels = 2
    options.evaluation_options.mesh_extraction = False
    solutions = compute_model(inputs, options, descriptor)
    evaluator = prepare_field_evaluator(solutions, inputs, descriptor, options)
    assert evaluator.fault_stacks == () and evaluator.fault_values(np.zeros((4, 3))).shape == (0, 4)
    points = np.array([[.5, .5, .37], [.2, .7, .9]])
    values = evaluator(points)
    np.testing.assert_allclose(values[0, 0], evaluator.stack_isovalues[0][0], atol=1e-8)
    np.testing.assert_array_equal(evaluator(points, fault_sides=np.empty((0, 2))), values)
