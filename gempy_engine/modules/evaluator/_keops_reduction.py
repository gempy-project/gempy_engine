"""PyKeOps sum reductions without per-call host uploads.

PyKeOps encodes every Python float of a formula as its own parameter and uploads it as a
one-element tensor on every call (``LazyTensor.fixvariables``): one blocking copy per
constant, 28-41 per evaluation kernel. ``keops_sum`` packs the constants of a reduction into
one parameter vector, uploaded once (asynchronously on CUDA), and reads each back with
``Elem``/``Extract``, so the arithmetic is unchanged. It also keeps PyKeOps' per-call
``[dimout, tagIJ]`` bookkeeping tensor, created on the default device, off the GPU.

This relies on PyKeOps internals (``variables``, ``formula``, ``formula2``, ``_dtype``); a
LazyTensor that does not look as expected is reduced unchanged.
"""
import contextlib
import copy


def keops_sum(lazy, **kwargs):
    """``lazy.sum(**kwargs)`` with the formula's float constants packed into one parameter."""
    packed, device = pack_constants(lazy)
    with _bookkeeping_on_host(device):
        return packed.sum(**kwargs)


def pack_constants(lazy, dtype=None, device=None):
    """(LazyTensor with one parameter vector for all its float constants, its data device).

    ``dtype`` (a dtype name) and ``device`` are used when the formula has no concrete tensor to take
    them from (every input symbolic)."""
    variables = getattr(lazy, "variables", None)
    formula, formula2 = getattr(lazy, "formula", None), getattr(lazy, "formula2", None)
    dtype = getattr(lazy, "_dtype", None) or dtype
    if variables is None or not isinstance(formula, str) or dtype is None:
        return lazy, None
    device = next((d for d in (lazy.tools.device(v) for v in variables if type(v) is not list) if d is not None), device)
    constants = [v for v in variables if type(v) is list]
    if len(constants) < 2:
        return lazy, device

    offsets, values = {}, []
    for constant in constants:
        if tuple(constant) not in offsets:
            offsets[tuple(constant)] = len(values)
            values.extend(constant)
    parameters = _upload(lazy, values, device, dtype)
    tag = f"Var({id(parameters)},{len(values)},2)"
    for constant in constants:
        start = offsets[tuple(constant)]
        read = f"Elem({tag},{start})" if len(constant) == 1 else f"Extract({tag},{start},{len(constant)})"
        old = f"Var({id(constant)},{len(constant)},2)"
        formula = formula.replace(old, read)
        formula2 = formula2.replace(old, read) if formula2 else formula2

    packed = copy.copy(lazy)
    packed.variables = tuple(v for v in variables if type(v) is not list) + (parameters,)
    packed.formula, packed.formula2 = formula, formula2
    return packed, device


def _upload(lazy, values, device, dtype):
    if getattr(device, "type", None) == "cuda":
        import torch
        host = torch.tensor(values, dtype=getattr(torch, dtype), device="cpu").pin_memory()
        return host.to(device, non_blocking=True)
    return lazy.tools.array(values, dtype, device)


def _bookkeeping_on_host(device):
    if getattr(device, "type", None) != "cuda":
        return contextlib.nullcontext()
    import torch
    return torch.device("cpu")
