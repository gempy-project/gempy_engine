# Joint Extraction (`joint`, `joint_contacts`)

`joint` and `joint_contacts` are opt-in, leaf-native dual-contouring modes. They
establish shared contact vertices and symbolic incidence before any triangle is
emitted, so meshes that meet at an erosion/onlap contact share the same vertex
by identity. This is not post-extraction vertex welding, contact
reconciliation, or a closed-volume mesher. `pretty` remains the default.

- `joint`: ordinary erosion/onlap contacts. Fault-free models only.
- `joint_contacts`: `joint`'s contacts plus `pretty`'s fault rule (horizons
  borrow the fault's vertices). Without faults it is exactly `joint`.

## Selection

```python
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data.options.evaluation_options import MeshExtractionMaskingOptions

evaluation = options.evaluation_options
evaluation.mesh_extraction = True
evaluation.mesh_extraction_overlap = "joint_contacts"  # or DualContouringOverlap.joint_contacts
evaluation.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT

meshes = compute_model(interpolation_input, options, data_descriptor).dc_meshes
```

`EvaluationOptions.mesh_extraction_overlap` defaults to `None`, meaning the
process default `DUAL_CONTOURING_VERTEX_OVERLAP` (read by `config.py` at import
time; `none` when unset). An explicit per-model value always wins.
`resolve_dual_contouring_overlap` accepts a name or a `DualContouringOverlap`
flag. `joint` and `joint_contacts` are exclusive: combinations with another flag,
unknown names and integers are rejected before any mutation. The joint dispatch
returns before legacy triangulation and overlap processing. With mesh
extraction disabled, the mode has no effect.

## Supported Domain

| Setting | Contract |
| --- | --- |
| Backend | NumPy or PyTorch (CPU or CUDA, with or without KeOps), `float64`, no autograd |
| Masking | `MeshExtractionMaskingOptions.INTERSECT` only |
| Extent capping | `MeshExtentCapping.NONE` only |
| Faults | `joint`: none. `joint_contacts`: any number of independent infinite faults (no fault offsets another fault) |
| Null-space stacks | Unsupported |
| Field callbacks | Actual finite raw scalar values and gradients for every exported surface |
| With faults | No micro correction, external interpolation or segmentation functions, or external fault values |

The leaf complex must cover the rectangular domain with aligned dyadic cubes,
2:1 face- and edge-balanced (corner-only jumps are allowed). There is no hidden
balancing, virtual refinement or uniform fallback; neither octree refinement
mode guarantees balance, so a production octree can be rejected.

## Reconstruction and Fields

`compute_model` passes the surface prefix `output[:number_octree_levels_surface]`
to the dispatcher. `_collect_joint_leaves` rebuilds the mixed-depth leaves from
all prefix levels: parents refined by the next level's `active_cells` are
removed. Raw stack scalar fields at the eight leaf corners are the samples.

New points (crossings, vertex checks) are evaluated with **fixed production
weights**: each `InterpOutput` keeps an independent copy of its actual solve
(`output.weights`; NumPy read-only, Torch clone). `production_weights` in
`fixed_weight_snapshots.py` copies them. While the weight cache is live, the
cache fingerprint must match the rebuilt solver input and the cached payload
must equal the output weights; mismatches raise `fingerprint mismatch ... no
solve allowed`. No query ever solves or reads the cache. After `compute_model`
returns (cache cleared) the output weights alone are the provenance.

- `joint`: one all-surface `sample_fields(points)` callback, cached by exact
  coordinate. External interpolation functions provide their own fields.
- `joint_contacts` with faults: `prepare_fault_drift_sampler` rebuilds the
  production fault drift at every query point (fault scalar, soft-segmented with
  the production ids, edges and slope, shifted by the frozen reference minimum),
  verified against the production drift on the full reference grid first. The
  core gets **targeted** queries: tile and minimal-edge node values are read
  from the leaf corners (shared nodes must agree), gradients are evaluated only
  for the crossing surface, and vertex checks are scalar-only. With KeOps,
  `query_batch` merges all requests per (stack, kind) into stacked block-sparse
  reductions, the evaluator FLAT stacks use.

## Pre-Triangle Construction

1. Canonical face tiles and minimal primal edges from the leaves
   (`build_adaptive_complex`), including fine edges hanging in coarse faces.
2. Production edge crossings, actual crossing gradients and production QEFs;
   one original branch per crossed leaf; tolerance-induced non-strict crossings
   and vanishing normals are rejected.
3. Contact junctions: isolated bilinear roots of an eligible cross-stack pair on
   canonical tiles. A junction leaf has one pair and exactly two tile roots; both
   surfaces use one QEF vertex on the root chord, with one symbolic key.
4. Minimal-edge rings kept by geological ownership (erosion/onlap
   controllers). Symbolic seam incidence is validated: two controller sides and
   one target side per internal seam.
5. Orientation from Hermite normals; nonzero area, opposite controller seam
   traversal, straddling and ownership of used vertices are checked. Only then
   are triangle arrays emitted.

These are sampled boundary and seam guards, not a cell-interior topology
certificate. Unsupported evidence raises (`unsupported_hanging_branch`,
`unsupported_multiway_junction`, `folded_controller_seam`, ...); there is no
fallback to another mode, no extra solve and no automatic cap.

## Fault Merge (`joint_contacts`)

Each fault surface maps to exactly the surfaces it affects (`fault_merge`). In
leaves where a fault and an affected surface both cross, the surface borrows
the fault's vertex by identity, so they meet exactly. The fault's QEF there
receives the borrowing surfaces' Hermite rows with `pretty`'s cross-surface
weight (10), which pulls it onto the intersection. Rings with borrowed vertices
use the production diagonals; all-fault triangles are dropped. Faults are never
ownership controllers.

- **Fallback region**: borrowed leaves plus the leaves sharing an edge with them.
  A contact junction there is not shared; both participants keep their own
  vertices without ownership certification. Deterministic, reported in
  `fault_overridden_junction_cells` and `fallback_region_leaf_count`.
- **Several faults in one leaf**: an affected surface crossed together with two
  or more of its faults has no unambiguous borrow; it keeps its own vertex there
  under the same fallback (`multi_fault_unborrowed_count`).
- Each fault is one conforming mesh. Near faults horizons follow the smooth
  production drift, as in `pretty`.

## Mesh Identity

One `DualContouringMesh` per exported surface, compacted to the vertices its
faces use:

| Attribute | Meaning |
| --- | --- |
| `mesh.edges` | Local triangle vertex indices |
| `mesh.joint_vertex_ids` | Local row to extraction-wide vertex ID |
| `mesh.joint_vertex_keys` | Symbolic key per local row (`regular` or `joint`, identities, leaf origin, span) |
| `mesh.joint_seam_edges` | All seam endpoint pairs in **extraction-wide** IDs |
| `mesh.contact_report` | Extraction-wide diagnostics (`sampled_point_count`, `borrowed_vertex_count`, `sampler`, ...) |

Equal IDs across meshes identify shared vertices with equal coordinates. Map
seams through `joint_vertex_ids`; do not index `mesh.vertices` with them.

## Model 7

- Natural root 9, depth 2 (`joint_contacts`): 414 fault leaves, 262 borrowed
  vertices, no junction fallback.
- Fully refined root 7 / minimum level 2: every fault vertex a horizon uses
  matches `pretty`'s fault vertex to 1e-9.
- Root 7 / minimum level 0 rejects with `unsupported_hanging_branch`.
- Torch CPU/CUDA give the same faces and keys as NumPy, vertices within about
  1e-4. KeOps evaluation itself differs from dense kernels by up to 1e-4, which
  can move a few vertices in every mode, `pretty` included.

## Code Layout

API files orchestrate; `modules/` files are pure functions that never import
another module.

| File | Role |
| --- | --- |
| `API/dual_contouring/joint_extraction.py` | Production bridge: leaf reconstruction, guards, `joint` / `joint_contacts` entry points, mesh compaction |
| `API/dual_contouring/joint_topology.py` | `extract_adaptive_topology`: sampling, QEFs, calls into the modules below |
| `API/dual_contouring/fault_drift_sampler.py` | Fixed-weight queries with production fault drift (`query`, `query_batch`) |
| `API/dual_contouring/fixed_weight_snapshots.py` | Production weight provenance, frozen solver inputs, shared contract checks |
| `modules/dual_contouring/joint_cell_complex.py` | Canonical tiles and minimal edges of a balanced leaf complex |
| `modules/dual_contouring/joint_cell_branches.py` | Corner/edge numbering, sampled branch classification |
| `modules/dual_contouring/joint_contact_relations.py` | Eligibility, fault pairs and truncation pairs between surfaces |
| `modules/dual_contouring/joint_lattice.py` | Lattice nodes, tolerances, shared corner lookup |
| `modules/dual_contouring/joint_ownership.py` | Controllers, fault-vertex borrowing, fallback region |
| `modules/dual_contouring/joint_edge_decisions.py` | Retained / aligned / hanging edge decisions |
| `modules/dual_contouring/joint_field_queries.py` | Exact-point cache and validated targeted batches |
| `modules/dual_contouring/joint_triangle_plan.py` | Junctions, edge rows, incidence and geometry validation, emission |

Tests: `test_joint_cell_complex.py`, `test_joint_topology.py`,
`test_joint_integration.py`, `test_joint_contacts.py`,
`test_fault_drift_sampler.py`, `test_joint_modules.py` in
`tests/test_common/test_modules/`. Benchmark:
`tests/benchmark/model7_backend_benchmark.py`.
