"""The fused Triton kernels agree with the torch paths they replace (gempy_engine.modules.fused).

On a CUDA device the kernels run natively. Without one they run in Triton's interpreter, in a
subprocess: the interpreter must be chosen (TRITON_INTERPRET=1) before Triton is imported.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.stack_relation_type import StackRelationType as R

from .test_stacked_combine import RELATIONS, _output

INTERPRETER_RUN = os.getenv("GEMPY_TEST_TRITON_INTERPRETER") == "1"


@pytest.fixture(autouse=True)
def torch_backend(restore_backend):
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    BackendTensor._change_backend(AvailableBackends.PYTORCH)


def _patch_interpreter_range():
    """Triton 3.6's interpreter passes scalar arguments as 1-element arrays, which ``range`` rejects with
    NumPy >= 2.3; convert them first."""
    import triton.runtime.interpreter as interpreter
    original = interpreter._patch_lang_core

    def patched(lang, scope):
        original(lang, scope)

        def as_int(value):
            data = getattr(getattr(value, "handle", None), "data", None)
            return value if data is None else int(data.reshape(-1)[0])

        def interpreter_range(arg1, arg2=None, step=None, **kwargs):
            start, end = (0, arg1) if arg2 is None else (arg1, arg2)
            return range(as_int(start), as_int(end), 1 if step is None else as_int(step))

        scope.set_attr(lang, "range", interpreter_range)
        scope.set_attr(lang, "static_range", interpreter_range)

    interpreter._patch_lang_core = patched


def _check_segmentation(device):
    import torch
    from gempy_engine.modules import fused
    from gempy_engine.modules.activator._soft_segment import segmentation_constants, soft_segment_stacked
    rng = np.random.default_rng(0)
    for dtype in (torch.float32, torch.float64):
        for n_units, slope in ((2, 50_000), (3, 7.0), (6, 1_000.5)):
            n_fields, n_points = 3, 2_500
            edges = np.sort(rng.normal(size=(n_fields, n_units - 1)), axis=1)
            edges = torch.as_tensor(edges, dtype=dtype, device=device)
            z = torch.as_tensor(rng.normal(size=(n_fields, n_points)) * 2, dtype=dtype, device=device)
            z[:, :n_units - 1] = edges  # points on the edges
            constants = [segmentation_constants((np.arange(n_units) * (k + 2) + 1).astype(np.int16), slope)
                         for k in range(n_fields)]
            ids = torch.stack([c[0] for c in constants]).to(device)
            tau = torch.stack([c[1] for c in constants]).to(device=device, dtype=dtype)

            fused_values = fused.soft_segment_stacked(z, edges, ids, tau)
            expected = soft_segment_stacked(z, edges, ids, tau)
            assert fused_values is not None and fused_values.dtype == expected.dtype
            scale = float(expected.abs().max())
            assert torch.allclose(fused_values, expected, rtol=1e-5, atol=1e-5 * scale), (dtype, n_units)


def _check_combination(device, monkeypatch):
    import torch
    from gempy_engine.API.interp_single._masking_ops import _combine_stacked
    from gempy_engine.modules import fused
    rng = np.random.default_rng(1)
    for name, relations in RELATIONS.items():
        for gradients in (False, True):
            outputs = [_output(rng, relation, torch.float32, gradients) for relation in relations]
            for output in outputs:
                fields = output.exported_fields
                output.values_block = output.values_on_all_xyz.to(device)
                for attribute in ("_scalar_field", "_gx_field", "_gy_field", "_gz_field"):
                    if getattr(fields, attribute) is not None:
                        setattr(fields, attribute, getattr(fields, attribute).to(device))
                fields._n_points_per_surface = fields._n_points_per_surface.to(device)

            fused_result = _combine_stacked(outputs, relations, relations, gradients)
            monkeypatch.setattr(fused, "FUSED_KERNELS", "off")
            expected = _combine_stacked(outputs, relations, relations, gradients)
            monkeypatch.undo()
            for f, e in zip(fused_result, expected):
                assert torch.equal(f.squeezed_mask_array, e.squeezed_mask_array), name
                for a, b in [(f.final_block, e.final_block), (f.faults_block, e.faults_block)] + [
                        (getattr(f.final_exported_fields, n), getattr(e.final_exported_fields, n))
                        for n in ("_scalar_field", "_gx_field", "_gy_field", "_gz_field")]:
                    if a is None or b is None:
                        assert a is b, name
                        continue
                    assert a.dtype == b.dtype and torch.allclose(a, b, rtol=1e-6, atol=1e-6), name


def _cuda_available():
    import torch
    return torch.cuda.is_available()


def test_fused_kernels_on_cuda(monkeypatch):
    if not _cuda_available():
        pytest.skip("no CUDA device")
    _check_segmentation("cuda")
    _check_combination("cuda", monkeypatch)


def test_fused_kernels_in_interpreter():
    if INTERPRETER_RUN:
        pytest.skip("this is the interpreter run itself")
    env = dict(os.environ, TRITON_INTERPRET="1", GEMPY_FUSED_KERNELS="force", GEMPY_TEST_TRITON_INTERPRETER="1")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                             f"{__file__}::test_interpreter_run"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert " passed" in result.stdout


@pytest.mark.skipif(not INTERPRETER_RUN, reason="runs in test_fused_kernels_in_interpreter's subprocess")
def test_interpreter_run(monkeypatch):
    _patch_interpreter_range()
    _check_segmentation("cpu")
    _check_combination("cpu", monkeypatch)


def test_fused_path_not_taken_off_cuda():
    import torch
    from gempy_engine.modules import fused
    if fused.FUSED_KERNELS == "force":
        pytest.skip("forced")
    z = torch.zeros((2, 8))
    assert fused.soft_segment_stacked(z, torch.zeros((2, 1)), torch.ones((2, 2), dtype=torch.int16),
                                      torch.ones((2, 1))) is None
    assert fused.combine(z, z, torch.zeros(2), ["erode", "true"], ["erode", "false"], torch.float32,
                         torch.float32) is None
