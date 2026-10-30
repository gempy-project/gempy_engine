"""Fused GPU kernels (Triton) for per-level work that otherwise takes dozens of small torch ops.

The segmentation of a chunk's stacks and the combination of every stack's field into the lithology and
fault blocks run once per octree level. As torch ops their time is set by the launches, not the
arithmetic; here each is one kernel launch.

Results agree with the torch path to float rounding, not bit for bit (other exp and division, other
summation order). ``GEMPY_FUSED_KERNELS``: ``auto`` (default: CUDA tensors and Triton available),
``off``, or ``force`` (any device; tests run the kernels in Triton's interpreter). ``warm_up()`` compiles
the kernels before the first request.
"""
import functools
import os

FUSED_KERNELS = os.getenv("GEMPY_FUSED_KERNELS", "auto").lower()
_MAX_STACKS = 31  # bit masks of the stacks' rows stay 32-bit ints: one compiled signature


@functools.cache
def _kernel_module():
    try:
        from . import _kernels
    except ImportError:
        return None
    return _kernels


def _enabled(*tensors) -> bool:
    if FUSED_KERNELS == "off" or _kernel_module() is None:
        return False
    return FUSED_KERNELS == "force" or all(tensor.is_cuda for tensor in tensors)


def soft_segment_stacked(z, edges, ids, tau_k):
    """``_soft_segment.soft_segment_stacked`` as one kernel, or None when the fused path does not apply.

    Applies to (B, N) fields with K >= 2 units whose edges and temperatures share the fields' float dtype.
    """
    import torch
    if not _enabled(z, edges, ids, tau_k):
        return None
    if (z.dtype not in (torch.float32, torch.float64) or edges.dtype != z.dtype or tau_k.dtype != z.dtype
            or ids.shape[1] < 2 or z.shape[1] == 0):
        return None
    z, edges, ids, tau_k = (tensor if tensor.stride(-1) == 1 else tensor.contiguous()
                            for tensor in (z, edges, ids, tau_k))
    kernels = _kernel_module()
    n_fields, n_points = z.shape
    out = torch.empty((n_fields, n_points), dtype=z.dtype, device=z.device)
    grid = (n_fields, kernels.triton.cdiv(n_points, kernels.BLOCK))
    kernels.segment_kernel[grid](z, edges, tau_k, ids, out, n_points, ids.shape[1],
                                 z.stride(0), edges.stride(0), tau_k.stride(0), ids.stride(0), BLOCK=kernels.BLOCK)
    return out


def combine(scalars, values, limits, lithology_kinds, fault_kinds, scalar_dtype, fault_dtype):
    """Lithology mask, value, scalar-field and fault blocks of ``_combine_stacked`` as one kernel.

    ``scalars``, ``values``: (n_stacks, N) contiguous; ``limits``: (n_stacks,) erosion limits in the
    scalars' dtype. Kinds per stack: lithology ``"erode"``, ``"false"`` or ``"true"``; faults ``"erode"``
    or ``"false"``. Returns four (n_stacks, N) tensors, or None when the fused path does not apply.
    """
    import torch
    n_stacks, n_points = scalars.shape
    if not _enabled(scalars, values, limits) or n_stacks > _MAX_STACKS or n_points == 0:
        return None
    if not (scalars.is_contiguous() and values.is_contiguous() and values.shape == scalars.shape):
        return None

    def bits(kinds, wanted):
        return sum(1 << i for i, kind in enumerate(kinds) if kind == wanted)

    kernels = _kernel_module()
    device = scalars.device
    mask = torch.empty((n_stacks, n_points), dtype=torch.uint8, device=device)
    final = torch.empty((n_stacks, n_points), dtype=values.dtype, device=device)
    scalar_blocks = torch.empty((n_stacks, n_points), dtype=scalar_dtype, device=device)
    fault = torch.empty((n_stacks, n_points), dtype=fault_dtype, device=device)
    grid = (kernels.triton.cdiv(n_points, kernels.BLOCK),)
    kernels.combine_kernel[grid](scalars, values, limits, mask, final, scalar_blocks, fault,
                                 n_stacks, n_points, (n_stacks - 1) * n_points, bits(lithology_kinds, "erode"),
                                 bits(lithology_kinds, "true"), bits(fault_kinds, "erode"), BLOCK=kernels.BLOCK)
    return mask.view(torch.bool), final, scalar_blocks, fault


def warm_up(device=None):
    """Compile the kernels (float32) with a tiny call, so the first request does not; False without them."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else None)
    if device is None or not _enabled(torch.empty(0, device=device)):
        return False
    f32 = dict(dtype=torch.float32, device=device)
    z = torch.linspace(-1, 1, 8, **f32).reshape(2, 4)
    ids = torch.tensor([[2, 1], [2, 1]], dtype=torch.int16, device=device)
    soft_segment_stacked(z, torch.zeros((2, 1), **f32), ids, torch.ones((2, 1), **f32))
    combine(z, z, torch.zeros(2, **f32), ["erode", "true"], ["erode", "false"], torch.float32, torch.float32)
    return True
