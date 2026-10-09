# Adaptive Joint Extraction

`joint` is an opt-in, leaf-native dual-contouring path for supported ordinary
geological contacts and one verified infinite independent planar fault. It
establishes shared contact representatives and symbolic incidence before emitting
triangles. It is not post-extraction vertex welding,
contact reconciliation, or a closed-volume mesher.

## Selection and Usage

Select the mode on the options passed to the engine's `compute_model`:

```python
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data.options.evaluation_options import (
    MeshExtentCapping,
    MeshExtractionMaskingOptions,
)

# interpolation_input, options, and data_descriptor are the prepared model.
evaluation = options.evaluation_options
evaluation.mesh_extraction = True
evaluation.mesh_extraction_overlap = "joint"
evaluation.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
evaluation.mesh_extraction_extent_capping = MeshExtentCapping.NONE

solutions = compute_model(interpolation_input, options, data_descriptor)
meshes = solutions.dc_meshes
```

`DualContouringOverlap.joint` is also accepted in place of the string. Configure
the production backend as CPU `float64` before computing. For cokriging, keep
autograd disabled (`BackendTensor.COMPUTE_GRADS=False`) and use
`InterpolationOptions.CacheMode.CACHE` or `IN_MEMORY_CACHE`. The ordinary
production interpolation must populate a valid weight cache; setting a cache
mode alone does not make an arbitrary saved octree extractable.

The process-wide default can instead be selected **before importing the engine**:

```bash
DUAL_CONTOURING_VERTEX_OVERLAP=joint DEFAULT_BACKEND=numpy DEFAULT_TENSOR_DTYPE=float64 /home/leguark/.venv/2025/bin/python your_model.py
```

`config.py` reads `DUAL_CONTOURING_VERTEX_OVERLAP` at import time, after loading
the configured dotenv file. When the variable is absent from both the environment
and loaded dotenv configuration, the global default is `none`.
`EvaluationOptions.mesh_extraction_overlap` defaults to `None`, meaning inherit
that global default. An explicit per-model value wins, including `"none"` or
`"pretty"` when the global default is `joint`. Changing the environment after
import is not a runtime mode switch.

`none` and `pretty` retain their legacy paths and behavior. `joint` is exclusive:
do not combine it with any other overlap flag. Combinations containing `joint`,
unknown names, integer substitutes, and strings such as `"joint|pretty"` are
rejected. Previously supported legacy enum bitfields remain supported.
The joint dispatch returns before legacy triangulation and overlap processing.
With mesh extraction disabled, selecting the mode does not produce meshes.

## Supported Domain

The leaf complex must fully cover a rectangular domain with non-overlapping,
aligned dyadic cubes. Leaf origins and spans are integer finest-lattice units;
spans are positive powers of two and origins are multiples of their spans. The
rectangular domain can contain multiple roots and need not itself be a cube or
have power-of-two dimensions. Physical extent bounds must be finite and
increasing.

Face neighbors and cells incident to an edge must differ by at most one depth
level (2:1 balance). Both face and edge balance are checked; a corner-only depth
jump is allowed. Incomplete coverage, overlap, invalid alignment, and unsupported
balance raise errors. There is no hidden balancing pass, virtual-leaf refinement,
or uniform-grid fallback.

The default `octree_refinement_mode` is `FAST`. Neither that default nor choosing
`OctreeRefinementMode.BALANCED` guarantees a 2:1-balanced leaf complex.
`BALANCED` denotes a support stencil, not a balancing algorithm or a guarantee of
watertightness. A production octree can therefore be rejected by joint extraction
even when its refinement mode is `BALANCED`.

Supported production settings are deliberately narrow:

| Setting | Joint contract |
| --- | --- |
| Backend | NumPy or PyTorch (CPU or CUDA, with or without KeOps), `float64`; all paths, including production `FAULT` banks |
| Masking | `MeshExtractionMaskingOptions.INTERSECT` only |
| Extent capping | `MeshExtentCapping.NONE` only |
| Faults | Exactly one infinite independent planar fault with verified production field and banks; finite, multiple, dependent, and nonplanar faults unsupported |
| Null-space stacks | Unsupported |
| Cokriging autograd | `COMPUTE_GRADS=True` is unsupported |
| Field callbacks | Actual finite raw scalar values and gradients for all exported surfaces |
| Fault micro/external callbacks | Micro correction, external interpolation callbacks, and externally supplied input fault values unsupported |

Fault dispatch checks fault relations, fault-bank inputs, and input fault values,
not only the presence of a `FAULT` stack label. `float32` and Torch autograd are
rejected rather than silently converted. Field queries run on the active device
(CUDA included); crossing and QEF tensors live on that device, while topology,
incidence and validation are host NumPy. The extracted mesh is not
differentiable. On Model 7 and a fault-free production plane, Torch CPU and CUDA
give the same faces, vertex keys and bank labels as NumPy, with vertices within
the production NumPy/Torch field difference (up to about 1e-4, `pretty`
included). Production KeOps evaluation differs from dense kernels, so KeOps meshes
can differ slightly in every mode.

### Fault Scope

The production fault path requires one `FAULT` stack exporting one separator
surface. Its directed boolean `faults_relations[source, target]` matrix must have
at least one fault-to-affected edge, no edges into the fault, and no edges from
other stacks. The fault is independent, not displaced by another fault. Unknown
fault metadata is rejected; entries must be `FaultsData` or `None` with one entry
per stack. Finite faults, null-space stacks, and per-stack grid subsets are not
supported.

The sampler verifies planarity using actual fixed-weight scalar and gradient
kernels at root points and off-observation domain probes. Fitting observations
to a plane is not sufficient. Actual gradients must be constant and aligned with
the scalar plane; their production scaling is preserved. Production drift must
span 0/1 and agree with the separator's signed scalar: **bank 0 is the positive
scalar side, bank 1 the negative side**, relative to the original fault isovalue.
The core also checks affine separator values and actual gradients at canonical
extraction points and subsequent queries.

Affected stacks are extracted independently in each bank. Unaffected independent
stacks are extracted once, with bank ID `-1`, not duplicated or fault-truncated.
Ordinary erosion/onlap relations within each partition remain in force, but
affected/unaffected cross-partition truncations are rejected rather than dropped.
The NumPy-only CPU `float64` and no-KeOps (evaluation and solver) restrictions
apply to production `FAULT` paths, not ordinary Torch CPU `float64` joint
extraction. Fault autograd is unsupported, as is ordinary cokriging autograd.
Fault micro correction and external
interpolation callbacks have no supported bank-aware contract.

## Production Reconstruction

`model_api.compute_model` passes the surface prefix
`output[:options.number_octree_levels_surface]` to the extraction dispatcher.
The adapter reconstructs mixed-depth leaves from **all levels in this prefix**,
not just its last level and not necessarily all volume-evaluation levels.
Parents marked refined by the next level's boolean `active_cells` mask are
removed; unrefined parents remain leaves, and all cells in the prefix's last
level remain leaves. A deeper volume level outside the prefix does not remove a
parent from the surface complex.

The finest lattice is derived from the root grid shape and prefix depth. Physical
cell centers determine canonical coordinates, with validation of the stored
integer coordinates; a constant signed/offset convention per level is accepted.
Every level must have the same extent, one output per stack, and eight raw corner
samples per cell. Exported `(stack_index, surface_index, isovalue)` metadata must
agree across levels. Raw stack scalar fields, rather than segmented lithology
blocks, supply the samples.

Query evaluation uses private input/options/stack state and restores its temporary
grid even on failure. The public joint computation also isolates the mutable
stack cursor, including the PyTorch path where the normal input deep copy is
skipped. Scalar and gradient evaluation are enabled on query options without
changing the caller's evaluation flags.

Public joint computation deep-copies caller fault metadata and mutable fault
buffers before production preprocessing, including when the full input copy is
disabled or skipped. Caller fault objects, buffer contents, and stack cursor are
preserved on success and failure; this isolation does not imply Torch fault
extraction support.

For ordinary cokriging, the adapter validates the cache fingerprint once per stack and
takes a private fixed-weight snapshot from the production solve associated with
the last supplied surface level. NumPy snapshots are copied and read-only;
PyTorch snapshots are detached clones. Subsequent query callbacks directly
evaluate those snapshots and do not re-resolve the shared cache or solve a new
interpolation system. Missing cache entries or fingerprint mismatches fail; no
query-time solve is permitted. External interpolation functions instead provide
their own raw scalar and gradient callbacks.

For the fault path, provenance is stricter: the actual production solve outputs
are retained as independent copied `ScalarFieldOutput.weights` (read-only NumPy
arrays), not aliases of the shared cache. Preparation uses the last supplied
**surface** output's full grid and original SP/ref/rest fault drift, even if the
input holds a deeper volume grid. Before any bank query, it requires both a valid
unforced solve cache fingerprint and an exact matching cache payload (including
dtype) against these actual output weights. Missing provenance, cache aliases,
fingerprint mismatches, and same-fingerprint corrupted payloads are rejected
without solving. Private immutable copies of solver state and actual weights
then serve queries directly: only coordinates and affected query drift change;
SP/ref/rest drift stays fixed. Queries neither solve nor access the cache, and
later shared-cache mutation cannot change an already prepared sampler. Inspection
snapshots are separate copies, not mutable access to query state.

Production cokriging retains the actual solve vector once per output in
`ScalarFieldOutput.weights`, for all cokriging outputs, not only fault extraction.
This adds one weight-vector copy per output (NumPy read-only copy; Torch clone
preserving its autograd graph), not a new solve or per-query snapshot. Fault
preparation makes its private immutable query copies once and checks provenance
consistency against the cached payload before querying. Retained output weights
therefore have a memory cost even when joint extraction is not selected.

Fault leaf corners, tile values, minimal edges, crossings, and gradients all use
that same bank sampler, not the original mixed-bank corner values. The fault
adapter deduplicates sampler queries by stack, effective bank, and exact
coordinate: every point is evaluated once per stack, affected stacks once per bank,
and fault/unaffected stacks once for both banks (`query(points, bank, stacks)`
accepts a stack subset; a bank is required only when an affected stack is
selected). The leaf cell complex is built once and shared by every partition. For bank 0 the
separator is `-(raw_fault - original_isovalue)`; for bank 1 it is
`raw_fault - original_isovalue`, with actual gradients oriented by the same sign.
Only the internally extracted separator level becomes zero. Original fault
isovalues, full stack metadata, and ordinary isovalue extrema remain unchanged.
The normalized positive side is excluded in each bank. This exact sampling and
normalization contract factors bank handling through the same core extraction.

## Pre-Triangle Construction

The implementation proceeds through these stages:

1. Build sparse canonical shared face tiles and minimal primal-edge segments
   from the original leaves. Edges include fine edges hanging within coarse
   faces; coarse cells can repeat in a cyclic incident-cell ring. Storage follows
   leaves and incidence, not a dense finest-grid volume.
2. Query canonical points for all surfaces together, caching each physical point.
   Validate callback values against original raw corner samples using
   field-scale tolerances plus raw-value roundoff allowances.
3. Preserve original production edge crossings, actual crossing gradients, and
   ordinary production QEF regularization. Require a single original branch per
   participating leaf and reject tolerance-induced non-strict crossings or
   vanishing Hermite normals.
4. Find isolated bilinear contact roots on canonical tiles. A supported junction
   leaf has one eligible cross-stack contact pair and exactly two tile roots.
   Combine both participants' original Hermite constraints into a regularized
   joint QEF constrained to the chord between those roots. Both surfaces use the
   same symbolic representative key and eventual global vertex ID.
5. Plan retained minimal-edge rings using geological erosion/onlap ownership.
   Collapse only consecutive cyclic repetitions caused by coarse leaves.
   Validate symbolic seam incidence before allocating final faces: an internal
   seam requires two controller triangle sides and one target triangle side,
   with no unmatched seam or fully shared triangle patch.
6. Orient symbolic triangles from Hermite normals and check nonzero area,
   resolvable winding, opposite controller seam traversal, geometric and target-
   field straddling, and actual retained-side ownership of used ordinary QEF
   vertices. Only then emit triangle index arrays.

Ordinary surfaces in one stack are not joined merely because they are nearby;
unrelated surfaces also retain independent identities. Joint representatives lie
on sampled contact-root chords, not necessarily on the exact nonlinear
intersection curve. Their ownership contract differs from the actual-field
retained-side check applied to ordinary representatives.

These checks are **sampled boundary and seam guards, not a full cell-interior
topology certificate**. They cannot rule out every hidden interior component or
intersection. A hanging-edge crossing without an original leaf branch is
rejected when detected, rather than repaired by hidden refinement. Multiway
contacts, multiple/nonisolated/tangential tile roots, grid-aligned junctions,
unsupported branches, degenerate faces, folded controller seams, and unowned
ordinary representatives can all cause extraction to fail.

Bank geometry alone opts into the strict-crossing primitive: an edge crosses
only when its scalar endpoints have strictly opposite signs, without
tolerance-extrapolated roots. Ordinary joint and legacy extraction retain their
existing crossing behavior and guards. Genuine aligned roots, including bank
corners or minimal-edge endpoints on a surface, are rejected before emission;
strict crossings do not add support for grid-aligned interfaces.

Production fault extraction uses a **global deferred-emission barrier**: both
banks and any unaffected partition first prepare symbolic triangle plans and
complete all incidence, geometry, and ownership validation. Only after every
partition passes can any triangle index array be emitted. A later-bank or
unaffected-partition rejection cannot leave an earlier bank already emitted.
Ordinary extraction retains its existing validation-before-emission path.

Physical crop boundaries do not receive caps. Minimal-edge rings with exterior
quadrants are not emitted as complete dual faces. Shared contact seams therefore
do not imply closed lithological volumes, globally watertight shells, or a
general-purpose manifold guarantee.

## Mesh Identity

The adapter returns one `DualContouringMesh` per original exported surface with
`stack_index`, `surface_index`, `exported_surface_index`, and `isovalue` metadata.
Each mesh is compacted independently to the global vertices referenced by its
faces:

| Attribute | Index space / meaning |
| --- | --- |
| `mesh.vertices` | Compact local NumPy coordinates |
| `mesh.edges` | Local triangle vertex indices, despite the attribute name |
| `mesh.joint_vertex_ids` | Local-row-to-global-vertex-ID mapping for this extraction |
| `mesh.joint_vertex_keys` | Canonical keys corresponding to those global IDs, in local row order |
| `mesh.joint_seam_edges` | Complete extraction-wide seam endpoint pairs in **global**, not local, IDs |
| `mesh.contact_report` | Extraction-wide diagnostics, not a per-surface topology certificate |
| `mesh.joint_bank_ids` | Typed optional NumPy array: one `int64` bank ID per local vertex on fault exports |
| `mesh.joint_face_bank_ids` | Typed optional NumPy array: one `int64` bank ID per local triangle on fault exports |

Fault exports combine disjoint bank components into that one mesh per original
surface, preserving export order and `geo_model` position-based mesh consumers.
Bank labels are `0`/`1` for affected surfaces and the fault, and `-1` for unaffected
surfaces extracted once. Every triangle's vertices have its face bank label.
These fields are declared on `DualContouringMesh`; ordinary meshes leave them
`None`. Bank components do not become extra exported surfaces.

The fault surface explicitly contains **two bank-side skins, not a single
conforming fault mesh**. Each skin shares its own symbolic contact representatives
with the corresponding affected bank. Even coincident opposite-bank vertices
never share keys or global IDs: canonical keys are bank-namespaced and numeric
global IDs occupy disjoint bank namespaces. There is no cross-bank welding,
closed-volume output, or single conforming fault-interface guarantee.

The `joint_vertex_ids`/`joint_vertex_keys` mapping includes ordinary vertices as
well as joint ones. Keys distinguish regular and joint representatives and encode
surface membership, leaf origin, span, and branch identity. Global IDs are shared
within one extraction, not persistent IDs across different models or grids.
The adapter attaches the shared identity attributes to the returned mesh instances.

To express local triangles in the shared global index space:

```python
mesh = meshes[0]
global_triangles = mesh.joint_vertex_ids[mesh.edges]
local_to_global = dict(enumerate(mesh.joint_vertex_ids))
global_to_local = {int(gid): local for local, gid in local_to_global.items()}
local_seams = [
    (global_to_local[int(a)], global_to_local[int(b)])
    for a, b in mesh.joint_seam_edges
    if int(a) in global_to_local and int(b) in global_to_local
]
```

Do not index `mesh.vertices` directly with `mesh.joint_seam_edges`. Every mesh
receives a copy of the complete global seam list, so some endpoints may not exist
in a particular surface's local vertex table. Equal global IDs across meshes
identify shared representatives with equal coordinates, without position-based
welding.

For ordinary extraction, `contact_report` records `algorithm`,
`coarsefine_seam_edge_count`,
`no_uniform_fallback`, `no_post_reconciliation`, `original_hermite_rows`,
`sampled_point_count`, `canonical_face_tile_count`, `minimal_edge_count`,
`finite_faults`, and `interior_topology`. The latter explicitly reports
`sampled_boundary_only_not_interior_certified`. The finite-fault diagnostic says
`unsupported`; the narrow infinite planar fault path does not change that limit.
Fault reports instead aggregate per-bank core diagnostics under `banks`, plus
`sampler` provenance/plane/bank diagnostics, total coarse/fine seam count,
`no_uniform_fallback`, `no_post_reconciliation`, and `cross_bank_id_joins=False`.
Their `interface` is `bank_side_fault_skin_not_single_conforming_fault_mesh`.
Counts describe the actual extraction and should not be read as accuracy or
closure guarantees.

## Model 7 Evidence

The real Model 7 fixture retains all 102 surface points and 8 orientations, the
fault, and the unconformity, with four original exports (`fault`, `rock3`, `rock2`,
`rock1`). Evidence has two distinct tiers: natural mixed-leaf production coverage
and controlled coarse/fine contact-transition coverage.

### Natural Mixed Leaves

The verified example default is root resolution 9, depth 2, minimum level 0,
without a patched refinement mask. Its natural production hierarchy contains
293 coarse span-2 leaves and 3488 fine span-1 leaves, totaling 3781 leaves with
full coverage of the `18 x 18 x 18` finest lattice
(`293 * 8 + 3488 = 5832`). Triangle counts per original mesh are
1496/544/1384/1878. There are 68 bank-0 seams and 119 bank-1 seams, totaling 187.
The coarse/fine **contact** seam count is zero despite mixed leaves. This proves
natural mixed-hierarchy extraction for this configuration, not contact seams
crossing a refinement transition, and does not establish arbitrary adaptive
hierarchy support.

### Controlled Contact Transitions

The root-resolution-7, two-surface-level fully refined production case also succeeds.
For full refinement, the finest domain is `14 x 14 x 14` with 2744 span-1 uniform
leaves, bank seam counts 52/91, and triangle counts per original mesh
884/312/832/1060. This is a uniform-leaf success through the leaf-native path,
not evidence of natural adaptive refinement.

The controlled mixed-resolution test retains actual production coarse parents
`(5, 3, 1)` and `(5, 3, 3)` in root coordinates: two span-2 leaves plus 2728 span-1
leaves cover the same finest domain. Bank seam counts are 51/90, with two actual
coarse/fine seam edges in each bank. Tests audit real controller/target triangle
incidence (two controller sides and one target side), bank identity, winding,
ownership, and no query solves. The refinement mask is explicitly controlled by
the test fixture; this qualifies as **controlled mixed-leaf testing, not natural
adaptive testing**. Neither case uses a hidden uniform fallback.
These controlled transitions supply the contact-transition evidence separately
from the natural root-9 default, whose contact seams do not cross coarse/fine
transitions.

These successes do not establish that all resolutions or hierarchies work.
The fixture's default root resolution 2/depth 3 can reject. Natural root 7/depth
2/minimum level 0 rejects with `unsupported_hanging_branch` before **any** triangle
emission across partitions; no solution or fallback is returned. Root 7 with
minimum level 1 or 2 fully refines and succeeds. Arbitrary minimum-level
hierarchies may raise `unsupported_hanging_branch`; lower-half Model 7 coarsening
is tested to reject `unsupported_dual_junction_incidence` before triangle
emission. Changing minimum level is not a general repair for unsupported topology.

### Runnable Example

[`examples/adaptive_joint_model7.py`](../../examples/adaptive_joint_model7.py)
can be run directly in PyCharm with no arguments. It configures NumPy CPU
`float64` without KeOps, computes the natural root-9/depth-2/minimum-level-0 case,
prints hierarchy and mesh diagnostics, and opens an interactive PyVista viewer.
Use `--check-only` to compute and validate without importing PyVista or rendering.
The `--root-resolution`, `--depth`, and `--min-level` flags select grid settings;
unsupported hierarchies report rejection rather than silently switching modes.

From the engine repository, when no active test run is competing for resources:

```bash
/home/leguark/.venv/2025/bin/python examples/adaptive_joint_model7.py --check-only
/home/leguark/.venv/2025/bin/python examples/adaptive_joint_model7.py --root-resolution 7 --depth 2 --min-level 2 --check-only
/home/leguark/.venv/2025/bin/python examples/adaptive_joint_model7.py --compare
```

`--compare` computes fresh `none`, `pretty`, and `joint` models with the same
root/depth/minimum-level settings and displays them together; it also accepts
`--check-only` for non-rendered diagnostics. Joint is computed first, so its
rejection cannot become a returned ordinary-mode fallback. Ordinary `none` and
`pretty` use production **smooth-fault fields**, while `joint` uses
**bank-conditioned fields and two fault skins**. Equal grid settings do not make
these equal field semantics: this is a visual/contract comparison, **not a fair
like-for-like algorithm benchmark**.

### Fixture Usage

From the engine repository, this uses the actual fixture with root resolution 7
and two levels, fully refined via the public minimum-level option. No production
function or refinement mask is patched. The fixture import requires the repository
test data and its dependencies; it is not a packaged public model loader.

```python
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.options.evaluation_options import (
    MeshExtentCapping,
    MeshExtractionMaskingOptions,
)
from gempy_engine.core.data.regular_grid import RegularGrid
from tests.fixtures.model7_combination import model7_combination_factory

BackendTensor._change_backend(
    AvailableBackends.numpy, use_gpu=False, use_pykeops=False,
    dtype="float64", grads=False,
)
inputs, options, descriptor = model7_combination_factory(
    number_octree_levels=2, mesh_extraction=True,
)
inputs.grid.octree_grid = RegularGrid(
    inputs.grid.octree_grid.orthogonal_extent, [7, 7, 7],
)
options.cache_mode = InterpolationOptions.CacheMode.IN_MEMORY_CACHE
evaluation = options.evaluation_options
evaluation.number_octree_levels_surface = 2
evaluation.octree_min_level = 2
evaluation.mesh_extraction_overlap = "joint"
evaluation.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
evaluation.mesh_extraction_extent_capping = MeshExtentCapping.NONE

solutions = compute_model(inputs, options, descriptor)
assert len(solutions.dc_meshes) == 4
for mesh in solutions.dc_meshes:
    print(mesh.exported_surface_index, len(mesh.edges), mesh.joint_face_bank_ids)
```

## `joint_contacts`: Joint Contacts With Pretty Fault Merge

`mesh_extraction_overlap = "joint_contacts"` (or
`DUAL_CONTOURING_VERTEX_OVERLAP=joint_contacts`) keeps `joint`'s adaptive leaves,
shared erosion/onlap junctions and validate-before-emit planning, but handles
faults with `pretty`'s rule instead of fault banks. Without faults it dispatches
to exactly `joint`.

**Fields.** A single extraction uses the production fields, including the smooth
fault drift. Leaf corners come from the octree; new points (crossings, vertex
checks) come from `prepare_fault_drift_sampler`, which evaluates the fixed
production weights and rebuilds the production drift at each query point:
each fault scalar is soft-segmented with the production ids, edges and
`sigmoid_slope`, then shifted by the frozen production reference minimum. The
rule is verified against the production drift on the full reference grid before
any query. Crossings are classified strictly, as in banked `joint`.

**Targeted queries.** The adapter passes the core a `field_query(points,
surfaces, gradients)` callback instead of full `sample_fields` queries (a full
query raises). Tile and minimal-edge node values come from the production leaf
corners; every such node is a leaf corner, and leaves sharing a node must agree
within the field tolerance and in sign (`inconsistent_corner_samples`).
Gradients are evaluated only for the crossing surface: at its edge crossings
and at the endpoints of edges it crosses. Vertex ownership checks are
scalar-only. On Model 7 this makes extraction cost the same as `pretty`.

**Batched, stacked queries.** The core sends its targeted queries as two batches:
every surface's crossing and crossed-edge endpoint gradients together, then both
vertex scalar checks together. With KeOps, the drift sampler's `query_batch`
merges the points of all requests sharing a (stack, kind). It evaluates every
fault scalar needed for the drift in one pass, then all remaining stacks in
stacked block-sparse reductions, the same evaluator FLAT stacks use (one per
kind; stacks are grouped by kernel options and drift-row count). On Model 7 this
replaces about 21 KeOps calls with 2 batches. Dense backends answer each request
separately, because merged batches only cross the evaluator chunk size and gain
nothing.

**Fault merge (core `fault_merge` contract).** Each represented fault surface
maps to exactly the surfaces it affects. In leaves where fault *f* has a vertex
and an affected surface crosses, that surface *borrows* the fault vertex by
identity (`regular(f, leaf)` key), so the meeting is exact by construction, not
welded. Its own discarded geometry there is neither solved nor validated. The
fault's QEF in those leaves receives the borrowing surfaces' Hermite rows with
`pretty`'s cross-surface weight (10), which moves the vertex onto the fault and
horizon intersection. Rings containing borrowed vertices use the production
diagonals; triangles made only of fault vertices are dropped (`pretty`'s `all`
rule). Faults are never ownership controllers.

**Fallback near faults.** A contact junction whose leaf lies in the fallback
region (borrowed leaves of any surface plus the leaves sharing an edge with
them) is not shared: both participants keep their own regular vertices, with
no ownership certification there. This is deterministic, not a retry after a
failure. The ring is needed because a kept seam's controller triangles reach the
leaves adjacent to it, where the target's field jumps across the fault.
`fault_overridden_junction_cells` and `fallback_region_leaf_count` report it.

**Output.** One mesh per original surface. Each fault is one conforming mesh.
There are no bank labels (`joint_bank_ids` and `joint_face_bank_ids` are
`None`). `joint_vertex_ids`, `joint_vertex_keys` and `joint_seam_edges` keep
`joint`'s meaning. `contact_report` adds `fault_cell_count`,
`borrowed_vertex_count`, `filtered_fault_triangle_count`,
`weighted_fault_vertex_count` and the sampler's drift-rebuild error.

**Backends.** `joint_contacts` with faults runs on NumPy and on Torch, on CPU
or CUDA, with or without KeOps (float64, no autograd). Field queries run on the
active device and return host arrays; topology, QEF assembly inputs and
validation stay on the host. Banked `joint` and the fault-free paths run on the
same backends. On Model 7, Torch
CPU/CUDA give the same faces and vertex keys as NumPy, with vertices within the
production NumPy/Torch field difference (about 3e-5). Production KeOps
evaluation differs from dense kernels by up to 1e-4 on Model 7's largest stack
(identical solved weights), which moves a few vertices and changes triangle
counts in every mode, `pretty` included.

**Scope.** Any number of independent infinite faults (no fault offsets another
fault). The graben fixture with two independent faults is tested. Rejections:
- `unsupported_multi_fault_cell`: a leaf would borrow from two faults. The
  graben at three octree levels hits this.
- Finite faults, fault-to-fault relations, external interpolation or
  segmentation functions, micro corrections, null space, capping, masking other
  than INTERSECT, float32, and Torch autograd (`COMPUTE_GRADS`).
- Every other `joint` guard still applies; Model 7 root 7 with minimum level 0
  still rejects with `unsupported_hanging_branch`.

**What it is not.** The horizon and fault contact is as good as `pretty`'s:
horizons end on shared fault vertices, but near the fault they follow the
smooth production drift and are smeared like `pretty`. There are no closed
volumes per fault block; use banked `joint` for that.

**Model 7.**
- Natural root 9 (414 fault leaves): 262 borrowed vertices, 390 dropped all-fault
  triangles, 85 shared contact seams and no junction fallback. The fault mesh has
  748 triangles, as in `pretty`.
- Fully refined root 7 / minimum level 2: every fault vertex a horizon uses
  matches `pretty`'s fault vertex to 1e-9. Here the fallback ring gives up 14
  junction leaves (52 seams instead of the 65 shared without it).

## Errors and Verification

Unsupported configurations and insufficient geometric evidence raise
`ValueError` or `NotImplementedError`; callback failures propagate. There is no
fallback to legacy `none`/`pretty`, uniform extraction, post-reconciliation,
additional interpolation solves, or automatic caps. Changing to another mode is
an explicit caller decision, not an error recovery performed by `joint`.

The relevant focused test files are:

- `tests/test_common/test_modules/test_adaptive_cell_complex.py`: canonical
  incidence, rectangular coverage, hanging rings, deterministic ordering,
  invalid leaves, face/edge balance rejection, and allowed corner-only jumps.
- `tests/test_common/test_modules/test_adaptive_topology_extraction.py`: mixed-
  leaf erosion/onlap seams on NumPy/Torch, original QEF preservation, independent
  surfaces, curved contacts, ownership, and pre-emission rejection reproductions.
- `tests/test_common/test_modules/test_adaptive_joint_integration.py`: environment
  and per-model selection, legacy overrides, prefix leaf reconstruction, adapter
  isolation, compact/shared IDs, public computation, and fixed-weight queries
  that forbid new solves and tolerate later shared-cache mutation. Public
  mixed-leaf computes verify coarse/fine shared edges and actual controller/target
  triangle incidence. Independent `none` versus `joint` plane comparisons check
  exact ordinary coordinates and oriented triangle multiplicities with both
  legacy and quad triangulation on both CPU backends.
- `tests/test_common/test_modules/test_adaptive_fault_sampler.py`: actual solve
  provenance, fingerprint and payload validation, immutable reference drift,
  production parity, no query solve/cache access, directed subsets, plane/bank
  verification, and unsupported fault contracts.
- `tests/test_common/test_modules/test_adaptive_fault_topology.py`: strict bank
  crossings, aligned-root rejection, controlled mixed-leaf seams, directed
  authorization, composite ownership, and pre-emission geometry guards. Standalone
  core tests on Torch do not authorize the production Torch fault sampler.
- `tests/test_common/test_modules/test_adaptive_fault_integration.py`: original
  export order, typed bank labels, disjoint keys/IDs, unaffected stacks extracted
  once, actual Model 7 production fields, controlled mixed-leaf incidence,
  coarsening rejection, and caller fault-metadata isolation on success/failure.

From the engine repository, run the focused suite when no full-suite run is
competing for resources:

```bash
DEFAULT_BACKEND=numpy /home/leguark/.venv/2025/bin/pytest tests/test_common/test_modules/test_adaptive_cell_complex.py tests/test_common/test_modules/test_adaptive_topology_extraction.py tests/test_common/test_modules/test_adaptive_joint_integration.py
DEFAULT_BACKEND=numpy /home/leguark/.venv/2025/bin/pytest tests/test_common/test_modules/test_adaptive_fault_sampler.py tests/test_common/test_modules/test_adaptive_fault_topology.py tests/test_common/test_modules/test_adaptive_fault_integration.py
```

Combined verification passed 791 selected tests covering adaptive ordinary and
fault extraction, production contact/QEF/triangulation/capping regressions,
weighted-QEF behavior, and the existing uniform topology and Model 7 experiments.
This was a focused regression selection, not the entire repository suite.
Independent review also reproduced geometric, provenance, and metadata-isolation
failures and confirmed their rejection or isolation after the fixes. Emission
sentinels verify that unsupported bank configurations fail before any partition
emits triangle arrays.

The runnable three-mode example was independently rendered off-screen and its
global seam identities and actual controller/target triangle incidence validated.
The comparison screenshot and report are
`/tmp/opencode/adaptive-joint-model7-comparison.png` and
`/tmp/opencode/adaptive-joint-model7-comparison.json`.

Matched solved-octree timings and the stage profile of the joint path are in
[`model7_matched_benchmarks.md`](model7_matched_benchmarks.md). Historical
uniform-extraction benchmark results do not describe the `joint` mode.

## Source Map

- `gempy_engine/config.py`: global environment default and exclusive-mode resolver.
- `gempy_engine/core/data/options/evaluation_options.py`: per-model selection,
  refinement stencil, masking, and capping options.
- `gempy_engine/API/model/model_api.py`: public computation, joint cursor
  isolation, surface-prefix dispatch, and final cache cleanup.
- `gempy_engine/API/dual_contouring/multi_scalar_dual_contouring.py`: early joint
  dispatch and unchanged legacy-mode dispatch.
- `gempy_engine/API/dual_contouring/adaptive_joint_extraction.py`: production
  guards, mixed-leaf reconstruction, fixed-weight callbacks, and local compaction.
- `gempy_engine/API/dual_contouring/adaptive_fault_extraction.py`: banked fault
  extraction and `extract_fault_merge_octree` (`joint_contacts`).
- `tests/test_common/test_modules/test_joint_contacts.py`: `joint_contacts` core
  contract, fallback, two-fault, Model 7 and graben coverage.
- `gempy_engine/API/dual_contouring/adaptive_fault_sampler.py`: verified planar
  production field, immutable actual-weight provenance, and fixed-drift bank queries.
- `gempy_engine/API/dual_contouring/adaptive_fault_extraction.py`: directed fault
  partition, exact bank normalization, global deferred-emission barrier, and
  original-surface assembly with disjoint bank namespaces.
- `gempy_engine/core/data/scalar_field_output.py` and
  `gempy_engine/API/interp_single/_interp_single_feature.py`: retained independent
  actual production solve weights.
- `gempy_engine/core/data/dual_contouring_mesh.py`: typed optional vertex/face
  bank-label fields.
- `gempy_engine/API/dual_contouring/adaptive_topology_extraction.py`: sampled
  fields, production QEFs, ownership, fully validated deferred symbolic plans,
  and diagnostics.
- `gempy_engine/modules/dual_contouring/adaptive_cell_complex.py`: sparse dyadic
  leaf incidence and coverage/balance validation.
- `gempy_engine/modules/dual_contouring/adaptive_joint_topology.py`: canonical
  tile roots, shared joint QEFs, seam incidence, orientation, and geometry guards.
- `examples/adaptive_joint_model7.py`: natural mixed-leaf Model 7 usage, hierarchy
  and bank/seam diagnostics, optional viewer, and field-semantics comparison.
