# QEF System for Faults and Watertight Dual Contouring

## Overview

The Quadratic Error Function (QEF) system in GemPy's dual contouring pipeline places
vertices inside voxels by minimizing the distance to a set of constraint planes. Each
constraint is defined by a point on an isosurface edge and the scalar-field gradient at
that point. The system has been extended with **weighted QEF** to handle multi-surface
interactions (faults, erosion, onlap) and produce **watertight** meshes at surface
intersections.

---

## 1. Standard (Single-Surface) QEF

For every valid voxel the solver collects up to 12 edge-crossing constraints
`(p_k, n_k)` where `p_k` is the intersection point and `n_k` is the gradient.
Three additional **bias constraints** pull the vertex toward the mass-point
(centroid of the edge crossings) along each axis, preventing the solution from
drifting outside the voxel.

The minimisation problem is:

```
minimise  Σ_k  w_k · (n_k · (x - p_k))²
```

which reduces to the normal equations **AᵀWA x = AᵀWb** solved per-voxel.

**Key file:** `_gen_vertices.py`

- Lines 22-58: Build the `edges_xyz` (15 × 3) and `edges_normals` (15 × 3) arrays
  (12 edge constraints + 3 bias constraints).
- Lines 79-111: Apply `√w` scaling, form AᵀA and Aᵀb, and solve via matrix inversion.

---

## 2. Weighted QEF for Multi-Surface Constraints

When multiple surfaces (from different stacks) share the same voxel, extra constraints
from the *other* surfaces are injected so that the resulting vertex respects all
surfaces simultaneously. This is the mechanism that keeps meshes watertight at
fault/layer intersections.

### 2.1 Constraint Injection (`_weighted_qef_setup_multicore.py`)

1. **Voxel code generation** – Each valid voxel is assigned a unique integer code from
   its left-right octree position (`_find_vertex_overlap._generate_voxel_codes`).
2. **Pairwise intersection** – For every pair of surfaces `(i, j)`, `np.intersect1d`
   finds voxels present in both surfaces.
3. **Extra rows** – For each overlapping voxel in surface `i`, the edge-crossing points
   and gradients of surface `j` are appended as additional constraint rows with a high
   cross-surface weight (`DEFAULT_CROSS_SURFACE_WEIGHT = 10.0`).
4. **Parallelism** – The N² loop is distributed across threads
   (`ThreadPoolExecutor`); NumPy releases the GIL during `intersect1d` and array
   operations.

The extra constraints are stored in `DualContouringData.extra_edge_xyz`,
`extra_edge_normals`, and `extra_weights`.

### 2.2 Solving with Extra Constraints (`_gen_vertices.py`, lines 61-77)

If `extra_edge_xyz` is not `None`, the extra rows are concatenated to the standard
15-row arrays, the weight vector is extended, and the same AᵀWA solve is applied.

---

## 3. Overlap Logic and Mesh Modifications

After vertices are computed, overlapping voxels between surfaces are resolved by
modifying the meshes (triangle removal, not vertex sharing).

### 3.1 `_apply_vertex_overlap_logic.py`

- **`apply_relations_to_overlaps`** – Entry point. First applies fault logic, then
  (if `watertight` mode is enabled) applies non-fault overlap logic.
- **`_apply_fault_logic`** – Iterates over all `(origin_stack, destination_stack)`
  pairs from `faults_relations` and calls `apply_overlap_to_surface_pair` for every
  combination of origin/destination surfaces.
- **`_apply_non_fault_overlap_logic`** – For non-fault stack pairs, determines
  master/destination based on `StackRelationType` (ERODE, ONLAP) and applies the same
  triangle-removal logic.

### 3.2 `_apply_mesh_modifications.py`

`apply_overlap_to_surface_pair` removes triangles from the *destination* mesh at
voxels that overlap with the *origin* mesh, effectively cutting the layer mesh where
the fault passes through.

---

## 4. Implemented Improvements

### 4.1 Skip wQEF for Fault–Layer Pairs

Fault–layer pairs are **excluded** from weighted QEF constraint injection
(`_build_allowed_partners` in `_weighted_qef_setup_multicore.py`).  Since the
destination (layer) triangles at fault overlaps will be removed, injecting layer
gradients into the fault QEF would only distort the fault plane.  The wQEF is now
reserved for non-fault overlaps (erosion/onlap in watertight mode).

### 4.2 Pairwise Partner Filtering

`_build_allowed_partners` uses `faults_relations` and `surface_to_stack` to compute
per-surface partner sets.  Surfaces whose stacks are linked by a fault relation are
excluded; only non-fault overlapping stack pairs exchange QEF constraints.  When no
fault model is present (`faults_relations is None`), legacy behaviour (all pairs) is
preserved.

### 4.3 Fault Vertex Preservation in Averaging

In `_average_overlapping_vertices` (`multi_scalar_dual_contouring.py`):

1. **Save** original fault-surface vertices before global averaging.
2. **Average** all surfaces globally (handles non-fault overlaps).
3. **Copy** the original (unperturbed) fault vertex to destination layer voxels at
   fault overlaps.  Only the specific voxels shared between a fault origin and its
   destination are overwritten — other voxels on the fault surface keep their averaged
   value for non-fault overlaps.

This ensures the fault plane stays smooth while destination layers (whose triangles
will be removed) receive the fault's position for watertightness.
