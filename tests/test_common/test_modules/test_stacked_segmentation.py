"""Several fields segmented at once give exactly the per-field segmentation (soft_segment_stacked)."""
import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.modules.activator._soft_segment import (segmentation_constants, soft_segment_stacked,
                                                          soft_segment_unbounded)


@pytest.fixture(autouse=True)
def torch_backend(restore_backend):
    pytest.importorskip("torch")
    BackendTensor._change_backend(AvailableBackends.PYTORCH)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("n_units", [2, 3, 6])
def test_stacked_segmentation_equals_per_field(dtype, n_units):
    import torch
    rng = np.random.default_rng(n_units)
    n_fields, n_points = 4, 513
    slopes = [50_000, 1_000.5, 7, 50_000]
    fields = [torch.as_tensor(rng.normal(size=n_points) * 3, dtype=getattr(torch, dtype)) for _ in range(n_fields)]
    edges = [torch.as_tensor(np.sort(rng.normal(size=n_units - 1)), dtype=getattr(torch, dtype)) for _ in range(n_fields)]
    ids = [(np.arange(n_units) * (k + 1) + 1).astype(np.int16) for k in range(n_fields)]

    constants = [segmentation_constants(i, slope) for i, slope in zip(ids, slopes)]
    stacked = soft_segment_stacked(torch.stack(fields), torch.stack(edges),
                                   torch.stack([c[0] for c in constants]), torch.stack([c[1] for c in constants]))
    for k in range(n_fields):
        expected = soft_segment_unbounded(fields[k], edges[k], ids[k], slopes[k])
        assert stacked[k][None, :].dtype == expected.dtype
        assert torch.equal(stacked[k][None, :], expected)
