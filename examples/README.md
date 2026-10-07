# Manual Contact-Aware Viewer

## Limitations Gallery

`contact_aware_limitations.py` shows six production-path panels: open interface
boundaries, coarse/fine onlap compared with its exact analytic seam, coarse/fine
same-group isolation, and real kriging faulted stratigraphy. Pink lines are
per-interface boundary edges, including legitimate contact seams, not automatically
cracks. White dots are shared contacts; cyan is the exact onlap intersection.
The reported seam distances use unit-cube coordinates. The last panel labels
unverified behavior instead of fabricating a finite-fault failure.

```bash
/home/leguark/.venv/2025/bin/python examples/contact_aware_limitations.py --gpu
```

Omit `--gpu` for NumPy CPU. CUDA mode uses PyTorch float64 without PyKeOps and
checks the analytic field device and original mesh tensor devices; it never
silently falls back to CPU. Contact reconciliation itself remains CPU/NumPy.
PyVista's rendering GPU is independent of the engine compute backend; the VTK
OpenGL renderer is reported after the window closes (or after offscreen rendering).
This is a visual GPU smoke test, not full GPU parity, finite-fault validation,
volume construction or a benchmark. It requires the same source checkout and
development fixture dependencies described below.

Press `H` for boundary/contact/seam overlays, `E` for triangle edges, `O` for
opacity, or `R` to reset the active camera. Each viewport rotates independently.
For an unattended screenshot, use `--off-screen --screenshot /tmp/opencode/contact_aware_limitations.png`.

## Five-Case Smoke Viewer

`contact_aware_smoke.py` is a standalone, manual-only smoke viewer, not a
pytest test or CI job. `examples/` is explicitly excluded from pytest recursion
in `pyproject.toml`. Run it from a source checkout with a development
environment and PyVista installed. It uses internal engine types and the
dev/test fixture factory `tests.fixtures.complex_geometries.one_fault_model`
(including its CSV data) for the real kriging example. This is not an installed
API example; an installed engine package alone does not provide those fixtures.
The fixture requires development/test dependencies such as pytest and pandas,
but the script calls its underlying factory directly without running pytest.

From the repository root, invoke the virtualenv interpreter directly:

```bash
/home/leguark/.venv/2025/bin/python examples/contact_aware_smoke.py
```

Use your own virtualenv's Python path as appropriate. The viewer shows planar
erosion, onlap, a curved three-way contact, same-group isolation, and faulted
stratigraphy using real kriging. Drag to rotate, scroll to zoom, Shift + drag
to pan, and press `C` for contact dots, `E` for triangle edges, `O` for opacity,
or `R` to reset the active camera. These are surface interfaces, not closed
lithological solids.

No image is created by default. With `python` resolving to your development
environment, explicitly request an offscreen screenshot:

```bash
python examples/contact_aware_smoke.py --off-screen --screenshot contact_aware_smoke.png
```

Offscreen rendering still requires a working VTK rendering backend on the
machine. To compute only onlap diagnostics without creating a plotter or
opening a window:

```bash
python examples/contact_aware_smoke.py --measure-onlap --onlap-resolution 64
```

The onlap diagnostic reports inactive shared vertices and shared row `x,z`
coordinates at an interior `y`, when available. It asserts that every ordinary
vertex with `contact_id >= 0` is referenced by final triangles and that unshared
vertices have exactly zero movement. It also reports substrate-plane and
contact-line distances. `--onlap-resolution` and `--isolation-resolution` each
accept `8`, `16`, `32`, or `64` (default `64`). Fine-resolution isolation asserts
that the upper same-group horizon remains separate from the older group.
