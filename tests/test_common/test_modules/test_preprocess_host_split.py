"""Surface points split by host-built indices equal the boolean-mask split (TensorsStructure.host_split)."""
import copy

import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import SurfacePoints, TensorsStructure
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess
from gempy_engine.modules.data_preprocess.data_preprocess_interface import prepare_faults


@pytest.fixture(autouse=True)
def torch_backend(restore_backend):
    pytest.importorskip("torch")
    BackendTensor._change_backend(AvailableBackends.PYTORCH)


def test_host_split_equals_mask_split():
    import torch
    rng = np.random.default_rng(5)
    counts = np.array([3, 0, 1, 5, 2])  # a zero is dropped; a single-point surface has no rest points
    structure = TensorsStructure(counts)
    assert structure.host_split() is not None
    masks = copy.copy(structure)
    masks.host_counts = None
    assert masks.host_split() is None

    surface_points = SurfacePoints(rng.random((counts.sum(), 3)), nugget_effect_scalar=rng.random(counts.sum()))
    by_index, by_mask = surface_points_preprocess(surface_points, structure), surface_points_preprocess(surface_points, masks)
    for name in ("ref_surface_points", "rest_surface_points", "nugget_effect_ref_rest", "nugget_effect_rest",
                 "nugget_effect_ref", "surface_ids"):
        a, b = getattr(by_index, name), getattr(by_mask, name)
        assert a.dtype == b.dtype and torch.equal(a, b), name

    fault_values = rng.random((2, counts.sum()))
    for a, b in zip(prepare_faults(fault_values, structure), prepare_faults(fault_values, masks)):
        assert a.dtype == b.dtype and torch.equal(a, b)


def test_subset_keeps_host_counts():
    from types import SimpleNamespace
    root = TensorsStructure(np.array([2, 3, 4, 1]))
    descriptor = SimpleNamespace(tensors_structure=root,
                                 stack_structure=SimpleNamespace(number_of_surfaces_per_stack_vector=np.array([0, 1, 4])))
    subset = TensorsStructure.from_tensor_structure_subset(descriptor, 1)
    np.testing.assert_array_equal(subset.host_counts, [3, 4, 1])


def test_chunk_preprocess_equals_per_stack():
    import torch
    from gempy_engine.core.data import Orientations
    from gempy_engine.modules.data_preprocess._input_preparation import chunk_preprocess, orientations_preprocess
    rng = np.random.default_rng(9)
    stacks = []
    for counts, n_ori in (([3, 1, 4], 2), ([5], 1), ([2, 2], 3)):
        counts = np.array(counts)
        stacks.append((SurfacePoints(rng.random((counts.sum(), 3)), nugget_effect_scalar=rng.random(counts.sum())),
                       Orientations(rng.random((n_ori, 3)), rng.random((n_ori, 3)), nugget_effect_grad=rng.random(n_ori)),
                       counts))
    structures = [TensorsStructure(counts) for _, _, counts in stacks]
    chunk = chunk_preprocess([sp for sp, _, _ in stacks], [o for _, o, _ in stacks], structures)
    assert chunk is not None
    for (sp, o, counts), structure, (sp_internal, ori_internal) in zip(stacks, structures, chunk):
        reference = TensorsStructure(counts)
        expected_sp, expected_ori = surface_points_preprocess(sp, reference), orientations_preprocess(o)
        for name in ("ref_surface_points", "rest_surface_points", "nugget_effect_rest", "nugget_effect_ref",
                     "nugget_effect_ref_unique", "surface_ids"):
            a, b = getattr(sp_internal, name), getattr(expected_sp, name)
            assert a.dtype == b.dtype and torch.equal(a, b), name
        for name in ("dip_positions_tiled", "gradients_tiled", "nugget_effect_grad"):
            a, b = getattr(ori_internal, name), getattr(expected_ori, name)
            assert a.dtype == b.dtype and torch.equal(a, b), name
        for a, b in zip(structure.host_split(), reference.host_split()):  # the local split left for the faults
            assert torch.equal(a, b)
