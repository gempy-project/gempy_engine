"""Shared scalar-only edge classification for contouring and NumPy cappers."""

import numpy as np


def scalar_crossing_parameters(start, end, iso, *, xp=np):
    """Return broadcast ``(crossing, t)`` for ``(1-t)*start + t*end == iso``.

    Inputs are floating arrays in the supplied NumPy or PyTorch namespace.
    ``scalar <= iso`` is inside: an entirely iso-valued edge never crosses,
    and an equal endpoint crosses only if the other endpoint is outside.
    Parameters are clamped to [0, 1], snapped within 1e-12 of an endpoint,
    and zero on non-crossing edges.
    Nonfinite inputs or interpolation arithmetic raise ValueError.
    The default namespace is NumPy, independent of the engine backend.
    """
    for name, value in (("start", start), ("end", end), ("iso", iso)):
        if not bool(xp.isfinite(value).all()):
            raise ValueError(f"Scalar crossing requires finite {name} values")

    crossing = (start <= iso) != (end <= iso)
    # Neutralize unused edges before dividing, without perturbing crossings.
    denominator = xp.where(crossing, end, 0) - xp.where(crossing, start, 0)
    numerator = xp.where(crossing, iso, 0) - xp.where(crossing, start, 0)
    if not bool((xp.isfinite(denominator) & xp.isfinite(numerator)).all()):
        raise ValueError("Nonfinite scalar crossing interpolation arithmetic")
    t = numerator / xp.where(crossing, denominator, 1)
    if not bool(xp.isfinite(t).all()):
        raise ValueError("Nonfinite scalar crossing interpolation parameter")
    t = xp.clip(t, 0, 1)
    t = xp.where(t <= 1e-12, 0, xp.where(t >= 1 - 1e-12, 1, t))
    return crossing, t
