# Dual Contouring Overview

## Module Layout
- `multi_scalar_dual_contouring.py`: Entry point for multi-stack extraction; handles masking, edge finding, interpolation, and triangulation per stack and returns `DualContouringMesh` objects.
- `_interpolate_on_edges.py`: Helper to locate edge intersections for a single stack, temporarily swap in a custom grid, interpolate exported fields on edge points, and package a `DualContouringData` payload.
- `_experimental_water_tight_DC_1.py`: Prototype routine that merges per-stack datasets and runs dual contouring once to investigate watertight outputs.
- `_mask_buffer.py`: Stateful placeholder intended for mask reuse, currently only provides a `clean` method.
- `__init__.py`: Empty; the package exposes no curated re-exports today.

## Data Flow Summary
1. `dual_contouring_multi_scalar` clones interpolation options, enabling gradient export required by triangulation.
2. Triangulation codes are generated (`get_triangulation_codes`) and masked per stack using `mask_generation` and `get_masked_codes`.
3. For each scalar field stack:
   - Validate stack relations with `_validate_stack_relations`.
   - Derive edge intersections and validity flags via `find_intersection_on_edge`.
   - Buffer intersection coordinates for later interpolation.
4. `_interp_on_edges` concatenates all intersections, swaps a temporary grid into the interpolation input, and calls `interpolate_all_fields_no_octree` to recover exported fields on edge points before restoring the original grid.
5. `DualContouringData` objects are built per stack, bundling edge/center coordinates, gradients, and metadata, then passed to `compute_dual_contouring` alongside masked triangulation codes to generate meshes.
6. Optional post-processing (`apply_faults_vertex_overlap`) is compiled but disabled; an experimental watertight path bypasses the standard pipeline when `options.debug_water_tight` is set.

## Noted Issues / Risks
- `_experimental_water_tight` invokes `all_meshes.append(*meshes)`, which fails if more than one mesh is returned, and it hardcodes the merge to two stacks while discarding scalar values and forcing `n_surfaces_to_export = 1`.
- `_validate_stack_relations` checks whether the previous stack relation equals both `Onlap` and `Erosion`, a condition that can never succeed, so the intended guard is dead code.
- `MaskBuffer` is cleared at the start of extraction but never populated or referenced again, introducing unnecessary shared state.
- Temporary grid swaps lack `try/finally` protection; exceptions during interpolation could leave `InterpolationInput` pointing at the wrong grid.
- `_interp_on_edges` re-interpolates all stacks on every invocation, even when only a subset needs updates, which can increase runtime cost.
- `apply_faults_vertex_overlap` is effectively unreachable due to the hard-coded `and False`, preventing debug runs from exercising fault-resolution logic.

## Improvement Opportunities
- Fix the watertight helper by switching to `extend`, generalizing the merge over all stacks, and preserving per-surface metadata, or gate it clearly as experimental.
- Replace `_validate_stack_relations` with accurate relation checks or remove it in favor of clearer upstream validation.
- Either implement mask caching within `MaskBuffer` or drop the class to avoid confusing statefulness.
- Wrap temporary grid assignments in context management to guarantee restoration when errors occur.
- Add selective interpolation paths (per-stack or cached results) to avoid redundant work in `_interp_on_edges` and `_interpolate_on_edges_for_dual_contouring`.
- Re-enable or refactor `apply_faults_vertex_overlap` so fault overlap correction can be toggled through configuration rather than being permanently disabled.

## Optimization Opportunities
- **Avoid repeated deep copies**: `dual_contouring_multi_scalar` performs a `copy.deepcopy` of `options` on every call (`gempy_engine/API/dual_contouring/multi_scalar_dual_contouring.py:50`). Profiling whether a shallow copy or explicit mutation of the gradient flag would suffice could trim overhead, especially when options include large backend tensors.
- **Cache triangulation artefacts**: `mask_generation` and `get_triangulation_codes` recompute masks and codes for every invocation (`gempy_engine/API/dual_contouring/multi_scalar_dual_contouring.py:60`). Persisting these per-octree level—or at least per session—could avoid redundant work when extracting multiple realisations with identical grids.
- **Incremental edge interpolation**: `_interp_on_edges` concatenates all intersection coordinates and re-runs `interpolate_all_fields_no_octree` even if only one stack changed (`gempy_engine/API/dual_contouring/multi_scalar_dual_contouring.py:188`). Introducing per-stack caching or dirty-bit tracking would let unchanged stacks reuse their previous edge fields.
- **Minimise array copies under masking**: Slicing `octree_grid.values[mask]` allocates new arrays for each stack (`gempy_engine/API/dual_contouring/multi_scalar_dual_contouring.py:113`). Refactoring `DualContouringData` to accept views or index arrays would reduce memory churn when masks are large.
- **Parallelise independent stacks**: Edge detection and triangulation are independent per stack (`gempy_engine/API/dual_contouring/multi_scalar_dual_contouring.py:75`). Leveraging multiprocessing, threading, or backend vectorisation could shorten wall-clock times on multi-core systems.
- **Use backend tensor ops in experiments**: `_merge_dc_data` relies on NumPy stacking (`gempy_engine/API/dual_contouring/_experimental_water_tight_DC_1.py:23`). Switching to `BackendTensor` abstractions would avoid CPU round-trips when the active backend is GPU-accelerated.
