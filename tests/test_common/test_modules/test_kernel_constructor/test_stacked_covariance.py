"""One covariance build for several stacks gives exactly the per-stack matrices (stacked_covariance)."""
import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import Orientations, SurfacePoints, TensorsStructure
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.options import KernelOptions, NuggetImplementation
from gempy_engine.modules.data_preprocess._input_preparation import orientations_preprocess, surface_points_preprocess
from gempy_engine.modules.kernel_constructor._kernels_assembler import create_cov_kernel
from gempy_engine.modules.kernel_constructor._vectors_preparation import cov_vectors_preparation
from gempy_engine.modules.kernel_constructor.stacked_covariance import stacked_covariances


@pytest.fixture(autouse=True)
def torch_backend(restore_backend):
    pytest.importorskip("torch")
    BackendTensor._change_backend(AvailableBackends.PYTORCH)


def _stack(rng, points_per_surface, n_orientations, n_faults):
    n_points = sum(points_per_surface)
    surface_points = SurfacePoints(rng.random((n_points, 3)), nugget_effect_scalar=rng.random(n_points) * 1e-2 + 1e-4)
    orientations = Orientations(rng.random((n_orientations, 3)), rng.random((n_orientations, 3)) - 0.5,
                                nugget_effect_grad=rng.random(n_orientations) * 1e-2 + 1e-3)
    structure = TensorsStructure(np.array(points_per_surface))
    sp_internal = surface_points_preprocess(surface_points, structure)
    faults = None
    if n_faults:
        n_rest, n_surfaces = n_points - len(points_per_surface), len(points_per_surface)
        faults = FaultsData(fault_values_on_sp=BackendTensor.t.array(rng.random((n_faults, n_points))))
        faults.fault_values_ref = BackendTensor.t.array(rng.random((n_faults, n_rest)), dtype=BackendTensor.dtype_obj)
        faults.fault_values_rest = BackendTensor.t.array(rng.random((n_faults, n_rest)), dtype=BackendTensor.dtype_obj)
        assert n_surfaces <= n_rest
    return SolverInput(sp_internal, orientations_preprocess(orientations), fault_internal=faults)


def _per_stack(solver_input, options):
    return create_cov_kernel(cov_vectors_preparation(solver_input, options), options)


@pytest.mark.parametrize("nugget", list(NuggetImplementation))
@pytest.mark.parametrize("kernel", [AvailableKernelFunctions.cubic, AvailableKernelFunctions.exponential])
@pytest.mark.parametrize("n_faults", [0, 2])
def test_stacked_blocks_equal_per_stack_matrices(nugget, kernel, n_faults):
    rng = np.random.default_rng(7)
    options = KernelOptions(range=1.7, c_o=0.9, uni_degree=1, kernel_function=kernel, nugget_implementation=nugget)
    stacks = [_stack(rng, [3, 4], 2, n_faults), _stack(rng, [5], 3, n_faults), _stack(rng, [2, 2, 3], 1, n_faults)]
    stacked = stacked_covariances(stacks, options)
    assert len(stacked) == len(stacks)
    for solver_input, block in zip(stacks, stacked):
        expected = _per_stack(solver_input, options)
        assert block.shape == expected.shape and block.dtype == expected.dtype
        assert block.is_contiguous()
        assert BackendTensor.t.equal(block, expected)


def test_stacked_rejects_mixed_fault_counts():
    rng = np.random.default_rng(3)
    options = KernelOptions(range=1.7, c_o=0.9)
    with pytest.raises(ValueError, match="number of faults"):
        stacked_covariances([_stack(rng, [3], 1, 0), _stack(rng, [3], 1, 1)], options)


def test_chunk_prebuild_groups_by_fault_count_and_splits_by_size(monkeypatch):
    from types import SimpleNamespace
    from gempy_engine.API.interp_single import _stack_ops
    from gempy_engine.core.compute_scope import StackCache

    rng = np.random.default_rng(11)
    options = KernelOptions(range=1.7, c_o=0.9, uni_degree=1)
    stacks = [_stack(rng, [3, 4], 2, 0), _stack(rng, [4], 2, 1), _stack(rng, [5], 3, 0), _stack(rng, [2, 3], 1, 0),
              _stack(rng, [3], 1, 1), _stack(rng, [6], 2, 0)]
    caches = [StackCache() for _ in stacks]
    builds = []
    original = _stack_ops.stacked_covariances
    monkeypatch.setattr(_stack_ops, "stacked_covariances",
                        lambda inputs, kernel_options: builds.append(len(inputs)) or original(inputs, kernel_options))
    monkeypatch.setattr(_stack_ops, "MAX_STACKED_SIZE", 30)  # fault-free stacks of 14, 16, 9 and 14 rows: two builds
    _stack_ops._prebuild_covariances(stacks, [SimpleNamespace(kernel_options=options)] * len(stacks), caches)
    assert sorted(builds) == [2, 2, 2]
    for solver_input, cache in zip(stacks, caches):
        assert BackendTensor.t.equal(cache.covariance, _per_stack(solver_input, options))


@pytest.mark.parametrize("uni_degree", [0, 1, 2])
@pytest.mark.parametrize("n_faults", [0, 2])
def test_chunk_kernel_input_equals_concatenated_stack_inputs(uni_degree, n_faults):
    from gempy_engine.core.compute_scope import StackCache
    from gempy_engine.modules.kernel_constructor._vectors_preparation import assemblies_key, _stack_assemblies
    from gempy_engine.modules.kernel_constructor.stacked_covariance import (_matrices_sizes, chunk_kernel_input,
                                                                            stack_kernel_inputs)
    rng = np.random.default_rng(13)
    options = KernelOptions(range=1.7, c_o=0.9, uni_degree=uni_degree)
    stacks = [_stack(rng, [3, 4], 2, n_faults), _stack(rng, [5], 3, n_faults), _stack(rng, [2, 2, 3], 1, n_faults)]
    for solver_input in stacks:
        solver_input.stack_cache = StackCache()
    chunk = chunk_kernel_input(stacks, options)
    assert chunk is not None
    reference = stack_kernel_inputs([cov_vectors_preparation(solver_input, options) for solver_input in stacks])
    for name in ("ori_sp_matrices", "cartesian_selector", "ori_drift", "ref_drift", "rest_drift",
                 "drift_matrix_selector", "ref_fault", "rest_fault"):
        a, b = getattr(chunk, name), getattr(reference, name)
        assert (a is None) == (b is None), name
        for field, value in (vars(a).items() if a is not None else ()):
            expected = vars(b)[field]
            if field == "n_faults_i":
                assert value == expected
                continue
            assert value.shape == expected.shape and value.dtype == expected.dtype, f"{name}.{field}"
            assert BackendTensor.t.equal(value, expected), f"{name}.{field}"
    for name in ("rest", "reference", "surface_ids"):
        assert BackendTensor.t.equal(getattr(chunk.nugget_scalar, name), getattr(reference.nugget_scalar, name))
    assert BackendTensor.t.equal(chunk.nugget_grad, reference.nugget_grad)
    assert chunk.block_layout.sizes == reference.block_layout.sizes

    for solver_input in stacks:  # the views left for the evaluation data side equal fresh assemblies
        cached = solver_input.stack_cache.data_side[assemblies_key(options)]
        solver_input.stack_cache = None
        fresh = _stack_assemblies(solver_input, options, _matrices_sizes(solver_input, options))
        for field in ("dips_ref_coord", "dips_rest_coord", "dips_ug", "dips_ref", "dips_rest", "faults"):
            a, b = getattr(cached, field), getattr(fresh, field)
            for x, y in (zip(a, b) if isinstance(a, tuple) else [(a, b)]) if a is not None else ():
                assert x.dtype == y.dtype and BackendTensor.t.equal(x, y), field
            assert (a is None) == (b is None)
