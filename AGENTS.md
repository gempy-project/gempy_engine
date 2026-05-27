# GemPy Engine — Agent Guide

## Project Overview

GemPy Engine is a Python library for implicit 3D geological modeling using cokriging interpolation. It generates scalar fields from surface point and orientation data, refines them over an octree, and optionally extracts surfaces via dual contouring.

**Key dependencies**: NumPy, PyTorch (optional), PyKeOps (optional), FastAPI (server), subsurface (output export).

## Build / Test / Run

### Workspace Environment

The virtual environment for this project is located at `/home/leguark/.venv/2025`. To avoid package collision or import errors, always run commands using the virtualenv's direct executables rather than global ones.

### Essential Commands

```bash
# Install in dev mode
/home/leguark/.venv/2025/bin/pip install -e ".[dev]"

# Run tests with the default/numpy backend
DEFAULT_BACKEND=numpy /home/leguark/.venv/2025/bin/pytest

# Run tests and skip approval tests
DEFAULT_BACKEND=numpy /home/leguark/.venv/2025/bin/pytest -k "not approval"

# Run a specific test file
DEFAULT_BACKEND=numpy /home/leguark/.venv/2025/bin/pytest tests/test_common/test_api/test_public_interface.py

# Run the FastAPI server
/home/leguark/.venv/2025/bin/uvicorn gempy_engine.API.server.main_server_pro:gempy_engine_App
```

*Note*: If you run a generic `pytest` command, you may hit `ImportPathMismatchError` due to leftovers in build directories (e.g., `build/lib/tests/conftest.py` vs `tests/conftest.py`). If this occurs, clean up the build artifacts first via `rm -rf build/`.

---

## Architecture

```
API/              ← Public entry points
  model/model_api.py    → compute_model() — the one main entry point
  interp_single/        → Octree-level interpolation loop + stack management
  dual_contouring/      → Multi-scalar dual contouring mesh extraction
  server/               → FastAPI endpoint wrapping compute_model()

core/             ← Data structures, backend abstraction, utilities
  backend_tensor.py     → BackendTensor singleton (NumPy vs PyTorch, CPU/GPU, PyKeOps)
  data/                 → All domain data classes (grids, inputs, options, outputs)
  config.py             → Environment-based configuration (AvailableBackends, feature flags)
  exceptions.py         → GemPyEngineInputError

modules/          ← Computational modules (no direct outside deps)
  kernel_constructor/   → Builds covariance matrices and kernel evaluation
  solver/               → Linear system solvers (NumPy, PyTorch, PyKeOps CG, GMRES)
  activator/            → Soft-sigmoid segmentation of scalar fields into unit blocks
  dual_contouring/      → Mesh generation from scalar field on octree corners
  evaluator/            → Evaluates weights against kernel to produce scalar fields
  octrees_topology/     → Octree refinement based on scalar field curvature
  data_preprocess/      → Prepares raw input for interpolation
  faults/               → Finite fault handling
  geophysics/           → Forward gravity and magnetic computation
  weights_cache/        → Disk/memory cache for interpolation weights
  topology/             → Topology edge extraction from octree
```

---

## Control Flow

```
compute_model(InterpolationInput, InterpolationOptions, InputDataDescriptor)
  │
  ├─ _check_input_validity()            ← Pre-computation validity check
  │
  ├─ interpolate_n_octree_levels()
  │    └─ for each octree level:
  │         └─ interpolate_on_octree()
  │              └─ interpolate_all_fields()
  │                   └─ _interpolate_stack()  (or _interpolate_stack_flat)
  │                        └─ for each stack (geological series):
  │                             ├─ input_preprocess() → SolverInput
  │                             ├─ compute_weights()  → solve Ax=b → weights
  │                             │    └─ solver_interface.kernel_reduction()
  │                             ├─ _evaluate_sys_eq()  → evaluate kernel → ExportedFields
  │                             └─ _segment()  → sigmoid activation → values block
  │                   └─ combine_scalar_fields()  → erosion/onlap masking
  │              └─ get_next_octree_grid()  → refine octree for next level
  │
  ├─ dual_contouring_multi_scalar()  (if mesh_extraction=True)
  │
  └─ Solutions(octrees_output, dc_meshes, gravity, magnetics)
```

---

## Key Concepts & Data Structures

### BackendTensor — The Compute Backend
`BackendTensor` is a **class with mutable class-level state** (not instantiated). It controls:
- **Engine backend**: `AvailableBackends.numpy` or `AvailableBackends.PYTORCH`.
- **GPU**: `use_gpu` flag (PyTorch only).
- **PyKeOps**: `pykeops_enabled` / `use_pykeops` for GPU-accelerated kernel ops.

The class provides aliases for backend operations: `BackendTensor.t` (the active backend module, e.g. NumPy or PyTorch), `BackendTensor.tfnp` (same, primarily for numpy-like syntax). All computation-agnostic code uses `BackendTensor.t.array(...)`, `BackendTensor.t.zeros(...)`, etc.

**Critical**: `pykeops_enabled` is **dynamically toggled** at runtime by `compute_weights()` and `_evaluate_sys_eq()` in `_interp_scalar_field.py` — they set and restore it per-call. Do not rely on the class-level value staying constant during a `compute_model()` run.

**Test backend** is configured via `tests/conftest.py`:
```python
BackendTensor._change_backend(engine_backend=backend, use_gpu=use_gpu)
```
Default test backend is `AvailableBackends.numpy`, GPU off.

### Stacks and Surfaces
The modeling is organized into **stacks** (geological series/sequences) each containing multiple **surfaces** (geological boundaries).

- `StacksStructure`: Describes the organization — number of surfaces, points, orientations **per stack**, masking relations (erosion, onlap, fault), and fault dependency matrix.
- `TensorsStructure`: Describes number of surface points per surface (lower-level split).
- `InputDataDescriptor`: Pairs `TensorsStructure` + `StacksStructure`. Frozen dataclass.

**Stacks are interpolated independently**, then combined via masking:
- **Erosion**: Upper stack's scalar field truncates lower stacks.
- **Onlap**: Lower stack's scalar field truncates upper stacks.
- **Fault**: Output modifies the combined block with fault offsets.

### Input Validation
Before interpolation starts, the `compute_model` entry point runs a comprehensive check `_check_input_validity(...)`. It raises a `GemPyEngineInputError` if it detects inconsistencies:
- Mismatches in orientation dip positions vs gradients lengths.
- Inconsistencies between `InterpolationInput` size and expectations set in `TensorsStructure` or `StacksStructure` (e.g. point/surface counts).
- Stacks or surfaces containing zero points or zero orientations.

### Options Classes & Deprecation Patterns
- `InterpolationOptions`: Pydantic `BaseModel` (with `ConfigDict`). Holds `kernel_options`, `evaluation_options`, `cache_mode`, `sigmoid_slope`, etc.
- `KernelOptions`: Standard `@dataclass` (not Pydantic). Holds `range`, `c_o`, `uni_degree`, `kernel_function`, `kernel_solver`, `compute_condition_number`.
- `EvaluationOptions`: `@dataclass`. Holds octree level counts, mesh extraction flags, gradient computation.

**Implicit Deprecation Gotcha**:
Several high-level attributes on `InterpolationOptions` are deprecated (such as `compute_scalar_gradient`, `number_octree_levels`, `number_octree_levels_surface`, and `mesh_extraction`). When modifying or creating configuration, access/set these on `options.evaluation_options` instead (e.g., `options.evaluation_options.number_octree_levels`).

### Grid System
`EngineGrid` is a composite that holds multiple grid types: `octree_grid`, `dense_grid`, `custom_grid`, `topography`, `sections`, `geophysics_grid`, `corners_grid`. Its `.values` property concatenates all non-None grids in a fixed order. Slices (`octree_grid_slice`, `dense_grid_slice`, etc.) partition the concatenated array to extract per-grid sub-arrays from final blocks.

### Octree Levels
The engine iterates over `number_octree_levels` levels. At each level:
1. Evaluate scalar field at current grid points
2. If not the last level, compute next octree grid from scalar field curvature
3. Set the new octree grid on `interpolation_input` via `set_temp_grid()`

The final octree output level's `InterpOutput.combined_scalar_field.final_block` is the geological model result.

### The Deep Copy Requirement
For NumPy backend (not PyTorch), `compute_model()` does `copy.deepcopy(interpolation_input)` to avoid side effects. Controlled by `NOT_MAKE_INPUT_DEEP_COPY` env var. The deep copy is **not** done for PyTorch backend — be aware of this when debugging PyTorch vs NumPy behavior differences.

---

## Environment Configuration

Configuration lives in `gempy_engine/config.py`, loaded from `.env` or `~/.env_gempy_engine`. The local `.env` is prioritized if both exist.

| Variable | Default | Purpose |
|---|---|---|
| `DEBUG_MODE` | `False` | Extra assertions and debug output |
| `DEFAULT_BACKEND` | `numpy` | `numpy` or `PYTORCH` |
| `DEFAULT_PYKEOPS` | `False` | Enable PyKeOps GPU kernels |
| `DEFAULT_TENSOR_DTYPE` | `float64` | Tensor data type |
| `OPTIMIZE_MEMORY` | `True` | Memory optimization flag |
| `SET_RAW_ARRAYS_IN_SOLUTION` | `True` | Populate `Solutions.raw_arrays` |
| `NOT_MAKE_INPUT_DEEP_COPY` | `False` | Skip input deep copy |
| `GEMPY_FLAT_STACKS` | `False` | Use parallel chunk-based stack processing |
| `GEMPY_MAX_CHUNK_SIZE` | `32000000000` | Max "cost" per chunk for flat stacks |
| `ONLY_LITH_SOLUTION` | `False` | Skip non-lithology blocks in output |
| `PYKEOPS_SOLVER` | `False` | Use PyKeOps for solver (not just kernel) |

---

## Coding Conventions

- **Bold separator comments**: `# region ... # endregion` used for logical code grouping.
- **Switch comments**: `# @off` / `# @on` to disable/enable code sections (e.g. around specific math blocks).
- **TODO tags**: `# TODO`, `# ?`, `# *` prefixed comments for notes/questions.
- **Underbar prefix**: Private module-internal functions use `_` prefix (e.g., `_solve_interpolation`).
- **Type annotations**: Heavy use of type hints throughout, but some are approximate (e.g., `Union` on `BackendTensor.tensor_types`).
- **Backend-agnostic code** uses `BackendTensor.t` for array ops; direct NumPy/PyTorch calls must be avoided in core computational paths.

---

## Testing

Tests are organized in `tests/`:
- `test_core/` — Data structure tests.
- `test_modules/` — Per-module tests (activator, dual contouring, kernel constructor, solvers, geophysics, etc.).
- `test_common/test_api/` — Integration tests using the public `compute_model()` API.
- `test_integrations/` — Multi-field, multi-grid, options integration.
- `test_server/` — Server endpoint tests.
- `test_pytorch/` — PyTorch backend gradient tests.
- `fixtures/` — Reusable pytest fixtures for models, grids, geometries.
- `test_dependencies/` — Checks for optional deps (PyKeOps compiler).

Test speed/requirement levels are set in `tests/conftest.py`:
```python
TEST_SPEED = TestSpeed.MINUTES
REQUIREMENT_LEVEL = Requirements.CORE
```

Approval tests (via `pytest-approvaltests`) store expected output in `.approved.txt` files. They use a PyCharm diff reporter configured in conftest.

---

## Common Pitfalls & Troubleshooting

1. **`BackendTensor` state leaks**: The singleton's class-level state persists across test runs. If a test changes the backend, subsequent tests may run with the wrong backend. Test conftest sets it at import time, but individual tests can still mutate it.

2. **`pykeops_enabled` vs `use_pykeops`**: `pykeops_enabled` is the runtime toggle, `use_pykeops` is the user-configured preference. The runtime code sets `pykeops_enabled = use_pykeops` at key points and may temporarily disable it.

3. **Deep copy in compute_model**: The deep copy can mask mutation bugs. If a test modifies `InterpolationInput` after calling `compute_model()`, it may or may not affect the result depending on the backend.

4. **Unit values default**: `InterpolationInput.unit_values` returns `np.arange(1000)` if never set — a non-obvious fallback.

5. **`KernelOptions` is a dataclass, `InterpolationOptions` is Pydantic**: They use different patterns for validation, serialization, and defaults.

6. **Grid slices must stay in sync**: The `EngineGrid.values` concatenation order must match the slice properties. Adding a new grid type requires adding it to both.

7. **`StacksStructure.stack_number` mutation**: The `stack_number` field is mutated during the stack loop — it's used as a cursor for active stack lookup via properties like `active_masking_descriptor`.

8. **`Solutions._raw_arrays` is only populated if `SET_RAW_ARRAYS_IN_SOLUTION=True` AND the first octree level has an octree grid**. Dense grid models skip raw arrays.

9. **Server mode uses hardcoded default options** in `main_server_pro.py:29-36` — changing `InterpolationOptions` defaults won't affect server behavior.

10. **The `compute_model` exception handler** catches and re-raises, but the `finally` block always clears the weight cache and GPU memory. If `compute_model` is called in a loop, each call starts with a clean cache.

11. **PyKeOps Permission Errors**:
    When running tests, some PyKeOps test scripts attempt to write cache files to hardcoded paths such as `/home/miguel` (e.g. in `tests/test_dependencies/test_pykeops.py`). This leads to a `PermissionError: [Errno 13] Permission denied`. This test can be ignored or skipped in environments where `/home/miguel` is inaccessible.

12. **PyKeOps LazyTensor Operations**:
    PyKeOps lazy evaluation has strict constraints on tensor inputs. Performing standard numpy functions or PyTorch ufuncs (like `sqrt`) on a raw `LazyTensor` can raise `TypeError: operand 'LazyTensor' does not support ufuncs` or `ValueError` due to axis expectations. Operations involving PyKeOps must strictly follow the wrapper functions or explicit lazy math structures.

13. **Global Pytest Path Mismatches**:
    When calling `pytest` without path restrictions, Python may attempt to collect from the `build/` directory if a previous compilation was performed. This causes an `ImportPathMismatchError` on `tests/conftest.py`. Always target tests explicitly or clear `build/` before running tests.
