# Contact-Aware Meshing Implementation Plan

## Status And Goal

**Superseded target plan:** see [Voxel-Based Contact Reconciliation](voxel_contact_reconciliation.md)
for the agreed contact rules, architecture, tests, and next implementation steps.
The discussion established shared-cell averaging across eligible structural
groups, strict same-group isolation, accepted coarse-resolution sticking, and
reuse of the working `pretty` fault approach. Exact planar intersection and
independent-surface isolation are no longer the target contract.

The remainder of this document is a historical record of the initial plan,
characterization results, and planar prototype. Its affine-only support limits,
QEF-bypass policy, and delivery checklist must not be interpreted as requirements
for the revised voxel-based implementation. The first voxel-based runtime
milestone is now implemented; see the linked plan's Runtime Snapshot for current
behavior and remaining limitations. The planar implementation described below
is no longer the active `contact_aware` path.

The Phase 1 characterization suite and legacy benchmark harness are implemented.
The initial Phase 2 `contact_aware` mode now supports planar erosion/onlap with
explicit validation and procedural, array-only computational modules. Broader
relations, backends, and performance comparisons remain pending.

Add an opt-in post-dual-contouring mode that reconciles erosion and onlap
contacts while preserving the existing extraction modes. Build the correctness
catalogue and performance baselines before implementing the new geometry logic.

The initial output contract is matching contact seams on the existing geological
surface outputs, not closed meshes for individual lithological units. A horizon
can legitimately be open at the domain extent or where it terminates against
another horizon. Coincident vertices alone are not proof of matching topology.

## Existing Implementation

Relevant paths are relative to the engine repository:

- `gempy_engine/config.py`: `DualContouringOverlap` and the environment setting.
- `gempy_engine/API/dual_contouring/multi_scalar_dual_contouring.py`: extraction orchestration.
- `gempy_engine/modules/dual_contouring/weighted_qef_setup_multicore.py`: pre-extraction cross-surface constraints.
- `gempy_engine/modules/dual_contouring/overlapping.py`: vertex averaging, directed fault snapping, and fault triangle removal.
- `gempy_engine/modules/dual_contouring/dual_contouring_interface.py`: extraction masks and edge crossings.
- `gempy_engine/API/interp_single/_masking_ops.py`: erosion/onlap ownership semantics.
- `gempy_engine/API/dual_contouring/extent_capping.py`: separate domain-boundary closure.

The existing modes are `none`, `pretty`, and `watertight`, selected with
`DUAL_CONTOURING_VERTEX_OVERLAP`. In the active extraction path, both `pretty`
and `watertight` enter the same non-`none` branch. That branch injects QEF
constraints before generating meshes, then averages non-fault overlap vertices,
copies fault vertices to affected surfaces, and removes fault-overlap triangles.

The older `_apply_vertex_overlap_logic.py` distinguishes `watertight` from
`pretty`, but is not the helper called by the active orchestration. Do not
reactivate that helper or change the meaning of either existing mode as part of
this feature.

Current limitations to characterize before development:

- Mesh generation and overlap handling run inside the stack loop over accumulated surfaces, rebuilding earlier meshes.
- Sequential pairwise averaging can leave inconsistent positions at three-way junctions.
- Empty allowed-partner sets become unrestricted `None` values; absent fault matrices also disable filtering. This concerns partner eligibility, not empty actual cell intersections.
- Same-cell occupancy is used as a contact proxy, although nearby surfaces can share a cell without intersecting.
- Cross-constraint arrays are dense over each target surface, even for sparse overlaps.
- Existing weighted-QEF tests do not establish shared seam connectivity; the triangle-removal test checks index bounds rather than actual removal.
- Extent capping skips faulted or extraction-masked stacks and therefore does not supply general solid closure.

These are baseline observations, not authorization to change legacy behavior.

## Option And Compatibility Contract

`DualContouringOverlap.contact_aware` is appended to the existing enum. Select it with:

```bash
DUAL_CONTOURING_VERTEX_OVERLAP=contact_aware
```

Do not introduce a second `MeshContactMode` setting. Keep the existing default
`none`, environment loading, and import-time configuration behavior.

Requirements:

- Omitted configuration retains today's default behavior.
- `none`, `pretty`, and `watertight` retain their existing output and dispatch behavior.
- Only `contact_aware` invokes the new contact reconciliation stage.
- Existing fault relations remain directional; erosion/onlap must not reinterpret unrelated fault pairs.
- Existing serialized model inputs remain valid; this environment setting does not require a new model payload field.
- Do not silently promote an existing mode to the new implementation.

Use fresh subprocesses for configuration tests because the setting is read at
import time. Verify every existing enum value and the new value, including the
default with the relevant environment variable explicitly unset.

## Architecture And Scope

Implement new erosion/onlap geometry operations after dual contouring. Do not
change interpolation, geological field combination, or octree refinement.

The insertion point is in `dual_contouring_multi_scalar`, after meshes and their
cell-to-vertex associations exist, before final output conversion and extent
capping. The new stage receives meshes plus existing extraction metadata:

- Surface-to-stack and per-stack surface identities.
- Cell codes, edge crossings, and gradients from `DualContouringData`.
- Existing corner scalar values and ownership masks.
- Stack relations and directional fault relations.

Mesh arrays alone are insufficient: they do not identify geological ownership
or distinguish actual contacts from accidental cell overlap.

For existing modes, leave orchestration and helper calls unchanged. For the new
mode, collect surfaces and perform final mesh/contact processing once, avoiding
the repeated accumulated-mesh processing in the legacy branch. Keep the dispatch
small; do not duplicate interpolation or edge-extraction code.

### Existing QEF And Fault Behavior

The current fault path is not exclusively post-processing: it also uses
pre-extraction cross-surface QEF constraints. The new contact stage must not be
described as replacing or correcting those constraints.

The characterization suite demonstrates that existing enabled-mode QEF
preparation can move parallel nonintersecting surfaces and reverse their order.
Reusing that preparation cannot meet a no-contact geometric-isolation contract
relative to independent extraction. The approved initial policy omits all
cross-surface QEF constraints in the new, non-fault branch. This changes new-mode
extraction preparation; the contact algorithm itself remains entirely after
dual contouring. Legacy modes retain their existing constraints and ordering.

The new branch replaces broad erosion/onlap averaging with contact-aware
reconciliation. Faulted models are explicitly unsupported initially. A future
extension must preserve directed fault copying and triangle removal without
applying broad non-fault averaging first. Leave the legacy sequence unchanged.

Do not silently alter QEF preparation or compensate for arbitrary upstream
distortion with mesh snapping. Displacement is possible, not inevitable for
every same-cell candidate; geometry and partner normals determine its effect.

### Initial Support Boundary

- Require NumPy CPU float64. Float32 QEF rounding can exceed the initial absolute spatial tolerance; PyTorch additionally uses origin-centered QEF regularization. Reject those configurations rather than concealing displacement with a larger tolerance.
- Require one or two stacks, exactly one surface per stack. Two-stack relations must be `[ERODE, BASEMENT]` or `[ONLAP, BASEMENT]`.
- Require scalar fields affine on supplied corner samples and planar extracted patches. Curved scalar fields are rejected, not approximated with fitted planes.
- Require `MeshExtractionMaskingOptions.INTERSECT`, which retains contact cells but does not guarantee that the finite extracted triangles reach the seam.
- Do not claim initial support for `DISJOINT`; missing contact cells cannot be recovered reliably after extraction.
- Reject `RAW`, faults, null-space stacks, and extent capping in the initial mode.
- No global remeshing, mesh-Boolean dependency, or new full-volume ownership allocation by default.
- No differentiability guarantee for topology-changing post-processing. Mesh vertices and triangles are returned as NumPy arrays; `vertices_tensor` retains the pre-overlap solve output. Preserve and document that split rather than implying reconciled topology has differentiable tensor vertices.

Cell extraction masks are distinct from pointwise geological ownership. Pass
corner scalar data, ownership information, and relations explicitly: these are
not stored in `DualContouringData`. Use the orchestration's stable surface
metadata; `n_surfaces_to_export` currently stores a stack index in this path.

## Post-Processing Algorithm

1. Build sparse candidate pairs using shared cell codes and geological relations.
2. Determine the controlling contact and retained side from existing masking semantics, including onlap chains. Do not infer precedence solely from stack indices.
3. Reject same-cell pairs that do not represent an actual geological intersection or ownership transition.
4. Compute seam geometry in local candidate patches. For the first implementation, target the extracted piecewise-linear geometry, with analytic fields used as independent accuracy checks.
5. Split crossing triangles where necessary and create consistent seam vertices and edge segments on the participating patches.
6. Clip the truncated horizon on the correct side and locally retriangulate affected patches. Preserve unaffected geometry.
7. Process multi-surface junctions as groups with a deterministic policy, not successive pairwise averages or last-writer assignments.
8. Validate local geometry and connectivity, then attach a small diagnostic report to the affected output.

Use existing samples first. Additional scalar evaluation to improve implicit
intersection accuracy is a later, measured extension, not an implicit
requirement of the first version. A shared or joint QEF can be evaluated as a
local vertex-placement technique; it must not replace actual seam connectivity.

Separate candidate discovery, geometric intersection, and geological clipping
only where independent tests or reuse justify the boundaries. Avoid building a
general contact framework before the first two-surface cases work.

## Correctness Catalogue

Each case should have a stable identifier, construction parameters, expected
owners, geometric oracle, required support, and expected diagnostic outcome.
Use small analytic scalar fields and direct extraction inputs first. Reuse the
same case definitions for production-path integration tests where practical.

| Family | Required cases | Primary oracle |
| --- | --- | --- |
| Basic erosion | Horizontal and oblique controlling contacts; several horizons terminating on one contact | Analytic intersection and retained half-space |
| Basic onlap | Horizontal and oblique substrate contacts; opposite geometric orientation | Correct substrate and retained side |
| False overlaps | Parallel nearby surfaces in one cell; unrelated stacks; geologically hidden horizons; same-stack surfaces | No unintended joining or contact-stage movement |
| Alignment | Contact through a corner, along a grid edge, and on a cell face; perturbations to either side | Stable tie policy and valid local topology |
| Tangency | Nearly tangent crossing; exact tangency without crossing | Correct contact classification, no invented crossing |
| Curvature | Smooth curved contact with known implicit equation; disconnected intersection curves | Residual bounds, convergence, component count |
| Thin units | Resolved thin layer; pinch-out; multiple crossings inside one cell; unresolved sub-cell unit | Correct resolved geometry or explicit insufficient-support outcome |
| Junctions | Three-way contact; several horizons meeting one boundary; multiple contact groups in one cell | Shared seam graph and deterministic junction policy |
| Relations | Erosion chain; onlap chain; mixed erosion/onlap; interrupted chain with a fault | Ownership consistent with existing geological masking |
| Fault interaction | Fault crossing contact; affected and unaffected stacks; multiple faults; finite-fault termination | Preserve fault separation and directional semantics |
| Boundaries | Seam reaches model extent; contact near missing refinement support; extraction-mask boundary | Classify legitimate boundary versus missing support |
| Degeneracy | Empty patch; zero-area input face; coincident surfaces; near-zero gradient where used | Defined rejection or tie policy, no NaN output |

For representative cases, test translations, anisotropic extents, uniform
scaling, resolution changes, and float64/float32 with dtype-aware tolerances.
Permute processing order while keeping geological precedence fixed. Do not
permute stacks and accidentally change the model's geology.

Avoid a full Cartesian product. Maintain a mandatory fast suite and an extended
suite covering more resolutions, backends, and compound cases.

## Independent Oracles And Diagnostics

Do not derive expected seams with the function under test or use legacy output
as the correctness oracle for the new mode.

- Planar cases: analytic intersection lines, contact planes, and retained half-spaces.
- Curved cases: implicit residuals and convergence against refined references.
- Geometry: finite coordinates, valid face indices, nonzero area, consistent orientation, and absence of unintended duplicate local faces.
- Seams: compare edge segments and connectivity, not only nearest vertex distances. Per-mesh duplicate coordinates can be legitimate when meshes are separately indexed.
- Ownership: verify retained triangle interiors against independent analytic classifications and existing mask semantics.
- Isolation: unchanged patches remain unchanged relative to the stage input; characterize any inherited QEF movement separately.
- Stability: deterministic connectivity and geometry within stated backend tolerances.

Do not require every edge of the combined geological surface network to have
two incident faces. A legitimate multi-surface junction can be non-manifold as
a network. Require paired boundary segments at a two-surface seam; assert
closed-manifold incidence only when explicitly testing a closed solid boundary.

Proposed diagnostics should distinguish reconciled contacts, legitimate domain
boundaries, ambiguous/coincident contacts, and insufficient extraction support.
Unsupported option combinations should fail clearly. Locally unresolved cases
must retain safe geometry and report the limitation; they must not be silently
declared watertight. Finalize this policy with explicit tests before exposing it.

## Test Layout

- `tests/fixtures/contact_cases.py`: small case catalogue, added only if shared by multiple suites.
- `tests/test_common/test_modules/test_contact_reconciliation.py`: analytic local geometry and seam invariants.
- `tests/test_common/test_modules/test_contact_reconciliation_integration.py`: small `compute_model()` cases, mode isolation, fault interaction, and input non-mutation.
- `tests/benchmark/test_benchmark_contacts.py`: extraction-stage and end-to-end performance cases.

Use `test_extent_capping.py` as a template for direct analytic geometry tests and
`test_extent_capping_integration.py` for production-path compatibility tests.
Reuse `simple_models.py` and `simple_geometries.py` for realistic unconformity
regressions, but do not substitute those fixtures for analytic oracles.

Pin backend, dtype, environment flags, and threading for baseline comparisons.
Capture representative pre-change legacy meshes or compact numerical references
so that omitted-option versus explicit-mode equality is not the only regression
check. Those two paths could otherwise regress together. Keep intentional
new-mode failures explicit and temporary; do not hide legacy limitations with
blanket skipped tests.

## Performance Benchmarks

Separate correctness checks from timed callables. Follow the existing
`benchmark.pedantic` convention in `tests/benchmark/test_benchmark_one_feature.py`.

Measure three scopes:

1. Contact reconciliation alone using prepared extraction data.
2. Extraction including current QEF preparation and triangulation.
3. End-to-end `compute_model()`.

For mutation-based stage benchmarks, supply fresh mesh inputs on each round.
Keep fixture construction/copying outside the timed stage and report end-to-end
cost separately. Measure peak process memory in isolated runs; Python allocation
tracking alone does not account for native arrays or GPU allocations. GPU runs
need synchronization around timed operations and separate device-memory metrics.

Vary resolution, stack count, horizons per stack, actual contact count, and sparse
versus dense cell overlap. Record active cells per surface, candidate pairs,
actual contact cells, input/output triangles, inserted seam vertices, and dtype.

The current pairwise intersection path is roughly `O(S^2 V log V)` for `S`
similarly sized surfaces with `V` cells in the dense-pair case, before repeated
stack-loop execution. Each dense cross-constraint partner adds about 672 bytes
per target voxel at float64, excluding concatenation and solve temporaries.
These are structural estimates, not measured baselines.

Prefer sparse work proportional to real candidate patches. Report runtime and
memory ratios against legacy modes on identical inputs, including no-contact
models. Do not impose fragile wall-clock thresholds in ordinary correctness CI.
Set performance budgets after collecting repeatable measurements.

## Delivery Phases

### Phase 1: Characterize And Specify

- [x] Build initial analytic case catalogue and independent checking utilities.
- [x] Capture legacy output references for all existing modes.
- [x] Measure inherited QEF distortion in false-overlap cases.
- [x] Add stage/end-to-end benchmark harness and initial timing/memory smoke baselines.
- [ ] Collect repeated timing distributions and multi-stack sparse/no-contact baselines.
- [x] Decide whether new-mode extraction preparation may omit/restrict broad QEF constraints.
- [ ] Finalize tolerance, tie, unsupported-input, and diagnostic policies.

Acceptance: the suite identifies known limitations without requiring a new mode
and separates legacy regressions from new-feature correctness expectations.

### Phase 2: Opt-In Dispatch And Basic Contacts

- [x] Add `DualContouringOverlap.contact_aware` and subprocess configuration tests.
- [x] Introduce isolated `match` dispatch while preserving existing branches and legacy-only Flag combinations.
- [x] Collect and process meshes once in the new branch.
- [x] Reject faulted inputs until fault-only handling is supported and tested.
- [x] Implement two-surface planar erosion/onlap clipping and matching seam edges.
- [x] Verify input non-mutation, tensor contract, and honest diagnostics.

Acceptance: basic analytic contacts pass; old modes match pre-change references;
no-contact stage inputs remain untouched; unsupported inputs fail or report as
specified. Revisit QEF preparation explicitly if inherited constraints prevent
the stated geometric contract.

### Phase 3: Junctions And Realistic Relations

- [ ] Handle alignment ties, curved contacts, and disconnected seams.
- [ ] Add grouped junction processing and resolved pinch-outs.
- [ ] Validate erosion/onlap chains against existing masking semantics.
- [ ] Cover fault-contact intersections without reconnecting displaced horizons.
- [ ] Run multi-resolution and extended backend/dtype cases.

Acceptance: seam connectivity and ownership pass independently of processing
order; unresolved geometry is diagnosed rather than manufactured.

### Phase 4: Optimize And Document

- [ ] Reuse cell indexes and candidate matches within the new stage.
- [ ] Reduce temporary allocations and benchmark no-contact overhead.
- [ ] Publish runtime/memory comparisons and the supported-case matrix.
- [ ] Document environment selection, limitations, and relation to extent capping.
- [ ] Consider local implicit refinement only if measured accuracy requires it.

Do not optimize legacy branches or switch defaults in this phase. Promotion of
the new mode and closed lithological-volume assembly are separate future work.

## Phase 1 Findings And Reproduction

Implemented files:

- `tests/fixtures/contact_cases.py`: `build_contact_case(name, resolution=6)` returns fresh RAW extraction inputs for planar erosion, onlap, parallel false overlap, and a three-way junction. Resolution can be an integer or three integers. Ownership metadata describes expected clipping; it does not clip fixture inputs.
- `tests/test_common/test_modules/test_contact_reconciliation.py`: 17 analytic and legacy characterization tests, including independent plane equations, retained-side checks, boundary-edge checks, and junction ordering.
- `tests/test_common/test_modules/test_contact_reconciliation_integration.py`: seven test cases covering isolated configuration imports, production output references, dispatch, and raw/QEF/post-overlap comparisons.
- `tests/test_common/test_modules/contact_legacy_reference.json`: static sampled numerical references and full connectivity digests for one nonempty production unconformity.
- `tests/benchmark/test_benchmark_contacts.py`: 36 legacy benchmark combinations across three scopes, two models, two root resolutions, and three modes.
- `tests/benchmark/contact_benchmark_runner.py`: isolated Linux process peak-RSS and cold-call timing runner.

Characterization tests intentionally assert today's defective outcomes. They are
legacy baselines, not acceptance tests claiming those outcomes are correct. Add
separate new-mode assertions when implementing reconciliation.

In a single unit cell, independent horizontal planes at `z=0.4` and `z=0.6`
are extracted at their original heights without cross-constraints. Existing QEF
injection moves them to approximately `0.577778` and `0.422222`, reversing their
order. The expected weighted solution is `(5*z_self + 40*z_partner)/45`:
four own-edge rows plus one bias row versus four partner rows weighted by 10.
Different-stack averaging then collapses both planes to `z=0.5`. Same-stack
averaging skips the pair, but the empty-partner filtering defect can leave the
QEF displacement intact.

In the unit-cube analytic catalogue, legacy averaging leaves 30 wrong-side
triangles for erosion and 20 for onlap. Coincident cell vertices do not create
boundary edges on the analytic seam. The three-way case has a vertex spread of
about `0.014488` and an order-dependent coordinate change of `0.006667`.

The production unconformity uses `INTERSECT`, legacy triangulation, float64 NumPy,
and two octree levels. Existing modes rebuild accumulated mesh sets of sizes
`1`, `2`, and `4`. `pretty` and `watertight` have identical active dispatch and
output. QEF displacement before averaging reaches approximately `0.05389`,
`0.15380`, `0.05360`, and `0.05226` model units across its four surfaces.

Run correctness characterization from the repository root:

```bash
DEFAULT_BACKEND=numpy /home/leguark/.venv/2025/bin/pytest \
  tests/test_common/test_modules/test_contact_reconciliation.py \
  tests/test_common/test_modules/test_contact_reconciliation_integration.py -q
```

Run the benchmark smoke matrix (increase rounds for meaningful distributions):

```bash
env DEFAULT_BACKEND=numpy DEFAULT_PYKEOPS=False DEFAULT_TENSOR_DTYPE=float64 \
  NOT_MAKE_INPUT_DEEP_COPY=False LINE_PROFILER_ENABLED=False \
  DUAL_CONTOURING_FAULT_OVERLAP_THREADING=False \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  GEMPY_CONTACT_BENCHMARK_ROUNDS=1 \
  /home/leguark/.venv/2025/bin/python -m pytest \
  tests/benchmark/test_benchmark_contacts.py --benchmark-only \
  --benchmark-min-rounds=1 --benchmark-warmup=off --benchmark-disable-gc -q
```

Run isolated process memory measurement:

```bash
/home/leguark/.venv/2025/bin/python tests/benchmark/contact_benchmark_runner.py \
  --scope extraction --case unconformity --resolution 4 --mode pretty
```

Initial one-round smoke timings for the unconformity with `pretty`:

| Root resolution | Final overlap pass | Extraction | End-to-end |
| --- | --- | --- | --- |
| 4 x 4 x 4 | 77 microseconds | 553 milliseconds | 1.640 seconds |
| 8 x 8 x 8 | 95 microseconds | 627 milliseconds | 2.109 seconds |

These are local smoke measurements, not stable performance ratios or budgets.
Separate isolated runs at resolution 4 measured lifetime peak RSS of roughly
681 MiB for prepared overlap/extraction and 694 MiB for end-to-end computation,
with imports alone contributing roughly 645-659 MiB. Report total lifetime peak,
startup peak, prepared peak, and pre-call current RSS; do not treat high-water
mark differences as actual stage allocation peaks.

Remaining gaps: analytic cases are axis-aligned RAW-support cases, not masked
production acceptance tests; no curved/faulted contacts or backend portability
claims are established. The performance no-contact model is single-stack, so it
does not measure sparse multi-stack pair-search overhead. Shared-cell counts are
candidates, not proven contact counts. The overlap benchmark isolates only the
final accumulated pass; extraction includes QEF, repeated earlier passes, and
cache cleanup. These observations describe legacy modes, which remain unchanged.

## Initial Runtime Implementation

New module boundaries follow the procedural data flow:

```text
API.prepare_planar_contact
  -> modules.contact_planes.fit_contact_plane(points, scalars, isovalue)
  -> explicit planes and geological relation
API.dual_contouring_multi_scalar
  -> independent extraction, once
API.reconcile_contact_meshes
  -> modules.contact_geometry.reconcile_planar_contact(arrays, planes, retained_sign)
  -> returned arrays and report
```

`contact_planes.py` and `contact_geometry.py` import only NumPy. Neither imports
the other or reads backend/configuration state. They do not mutate input arrays.
The API validates options, passes data, and attaches returned geometry to meshes.
Temporary input and stack-cursor copies avoid mutating caller state without
deep-copying callback objects or tensor graphs.

Plane fitting validates affine consistency before extraction. Geometry clips the
truncated patch, splits intersecting controller triangles, and unions seam
breakpoints so both patches contain matching edge segments. It validates seam
incidence: target edges have one incident face, controller edges have one at a
patch boundary or two internally. Boundary seam counts are reported explicitly.
Coincident planes, partially supported seams, nonplanar input, and coordinates
whose precision cannot represent the requested tolerance fail with `ValueError`.
Wholly discarded targets return empty faces; retained parallel surfaces never
join. Output arrays may contain unused vertices after clipping.

For reconciled two-stack outputs, `mesh.contact_report` describes the result and
role. `vertices_tensor`, `dc_data`, and `support_report` continue to describe
pre-reconciliation extraction, not final vertex/cell correspondence. Single-stack
output remains independent extraction without a contact report.

Additional tests:

- `test_contact_planes.py`: independent affine fits and validation failures.
- `test_contact_geometry.py`: perpendicular/oblique seams, mismatched segmentation, ownership, winding, boundary incidence, insufficient support, coordinate precision, and non-mutation.
- `test_contact_aware_integration.py`: synthetic extraction and true `compute_model()` tests using external affine functions, early support rejection, new-stage isolation, legacy Flag compatibility, and callback identity preservation.

The production tests choose planes that actually intersect the finite extracted
triangle support. For example, at root resolution 6 with two octree levels,
erosion at `z=0.39` leaves the target patch ending at `z=0.375`; the successful
erosion test instead uses `z=0.36`. This limitation cannot be repaired by snapping
a missing seam into existence. Refinement/support improvements remain separate
work. A field supplied through ordinary kriging is not necessarily affine even
when its observations describe a plane, so it may be rejected by this initial
strict support contract.

The generalized sparse candidate algorithm above remains future work. The
initial implementation scans the two finite patches and does not yet claim
sparse multi-stack scalability or closed lithological solids.
