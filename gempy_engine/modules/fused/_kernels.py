"""Triton kernels of ``gempy_engine.modules.fused``; imported only when Triton is available.

Sizes and loop counts are runtime arguments (not specialized), so one compiled kernel per dtype serves
every model and octree level.
"""
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def _sigmoid(z, edge, tau):
    return 1.0 / (1.0 + tl.exp(tl.clamp(-(z - edge) / tau, -50.0, 50.0)))


@triton.jit(do_not_specialize=["n_points", "n_units", "z_stride", "edges_stride", "tau_stride", "ids_stride"])
def segment_kernel(z_ptr, edges_ptr, tau_ptr, ids_ptr, out_ptr, n_points, n_units,
                   z_stride, edges_stride, tau_stride, ids_stride, BLOCK: tl.constexpr):
    """``soft_segment_stacked`` of one field (program axis 0) over a block of points (axis 1).

    Each edge's sigmoid is computed once: unit k's membership is s_{k-1} - s_k, between the first unit's
    1 - s_0 (as its own sigmoid) and the last unit's s_{K-2}.
    """
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offsets < n_points
    z = tl.load(z_ptr + row * z_stride + offsets, mask=in_bounds, other=0.0)
    edges_row = edges_ptr + row * edges_stride
    tau_row = tau_ptr + row * tau_stride
    ids_row = ids_ptr + row * ids_stride

    edge, tau = tl.load(edges_row), tl.load(tau_row)
    previous = _sigmoid(z, edge, tau)
    total = _sigmoid(-z, -edge, tau) * tl.load(ids_row).to(z.dtype)
    for k in tl.range(1, n_units - 1):
        current = _sigmoid(z, tl.load(edges_row + k), tl.load(tau_row + k))
        total += (previous - current) * tl.load(ids_row + k).to(z.dtype)
        previous = current
    total += previous * tl.load(ids_row + n_units - 1).to(z.dtype)
    tl.store(out_ptr + row * n_points + offsets, total.to(out_ptr.dtype.element_ty), mask=in_bounds)


@triton.jit
def _bit(bits, i):
    return ((bits >> i) & 1) == 1


@triton.jit(do_not_specialize=["n_stacks", "n_points", "last_offset", "lith_erode", "lith_true", "fault_erode"])
def combine_kernel(scalar_ptr, value_ptr, limit_ptr, mask_ptr, final_ptr, scalar_out_ptr, fault_ptr,
                   n_stacks, n_points, last_offset, lith_erode, lith_true, fault_erode, BLOCK: tl.constexpr):
    """``_lithology_mask``, ``_faults_mask`` and ``_combine_scalar_fields`` for a block of points.

    Inputs and outputs are (n_stacks, n_points), row-major. A stack's own lithology row is its erosion
    mask (scalar field above its limit), all False or all True, as the bit masks say; its fault row is
    its erosion mask or all False. Stack 0 keeps the points that the last stack's and its own rows share;
    every later stack the points of its own row that no earlier stack's row has. The blocks accumulate
    stack by stack.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offsets < n_points
    last = n_stacks - 1
    last_scalar = tl.load(scalar_ptr + last_offset + offsets, mask=in_bounds, other=0.0)
    last_row = tl.where(_bit(lith_erode, last), last_scalar > tl.load(limit_ptr + last), _bit(lith_true, last))

    value = tl.zeros([BLOCK], dtype=final_ptr.dtype.element_ty)
    scalar_total = tl.zeros([BLOCK], dtype=scalar_out_ptr.dtype.element_ty)
    fault = tl.zeros([BLOCK], dtype=fault_ptr.dtype.element_ty)

    # Stack 0
    scalar = tl.load(scalar_ptr + offsets, mask=in_bounds, other=0.0)
    values = tl.load(value_ptr + offsets, mask=in_bounds, other=0.0)
    eroded = scalar > tl.load(limit_ptr)
    own = tl.where(_bit(lith_erode, 0), eroded, _bit(lith_true, 0))
    keep = own & last_row
    none_before = own == 0
    value += tl.where(keep, values, 0.0).to(value.dtype)
    scalar_total += tl.where(keep, scalar, 0.0).to(scalar_total.dtype)
    fault += tl.where(_bit(fault_erode, 0) & eroded, values, 0.0).to(fault.dtype)
    tl.store(mask_ptr + offsets, keep.to(tl.uint8), mask=in_bounds)
    tl.store(final_ptr + offsets, value, mask=in_bounds)
    tl.store(scalar_out_ptr + offsets, scalar_total, mask=in_bounds)
    tl.store(fault_ptr + offsets, fault, mask=in_bounds)

    row = offsets
    for i in tl.range(1, n_stacks):
        scalar_ptr += n_points
        value_ptr += n_points
        mask_ptr += n_points
        final_ptr += n_points
        scalar_out_ptr += n_points
        fault_ptr += n_points
        scalar = tl.load(scalar_ptr + row, mask=in_bounds, other=0.0)
        values = tl.load(value_ptr + row, mask=in_bounds, other=0.0)
        eroded = scalar > tl.load(limit_ptr + i)
        own = tl.where(_bit(lith_erode, i), eroded, _bit(lith_true, i))
        keep = own & none_before
        none_before = none_before & (own == 0)
        value += tl.where(keep, values, 0.0).to(value.dtype)
        scalar_total += tl.where(keep, scalar, 0.0).to(scalar_total.dtype)
        fault += tl.where(_bit(fault_erode, i) & eroded, values, 0.0).to(fault.dtype)
        tl.store(mask_ptr + row, keep.to(tl.uint8), mask=in_bounds)
        tl.store(final_ptr + row, value, mask=in_bounds)
        tl.store(scalar_out_ptr + row, scalar_total, mask=in_bounds)
        tl.store(fault_ptr + row, fault, mask=in_bounds)
