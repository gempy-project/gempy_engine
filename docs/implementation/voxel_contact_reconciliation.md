# Voxel-Based Contact Reconciliation Plan

## Status And Authority

This document records the agreed direction after reviewing the planar
`contact_aware` prototype. It supersedes the target architecture and acceptance
criteria in `contact_aware_meshing.md`. That document remains a record of the
characterization work and the implemented, now-superseded planar prototype.

The first voxel-based runtime milestone is implemented. `contact_aware` now
uses cell-local grouping and relation-directed triangle handling instead of
affine plane fitting and triangle clipping. The sections below record the
agreed contract and the remaining work; the runtime snapshot describes the
implemented limits rather than claiming closed simulation-ready volumes.

## Goal

Extend the existing `pretty` overlap approach using shared octree metadata.
The intended progression is:

```text
Consistent geological interfaces
    -> lithological volume assembly
    -> numerical simulation meshes
```

Construct contacts from cell identities, geological ownership, and existing
edge/triangle connectivity. Do not treat surfaces as unrelated meshes whose
intersections must be rediscovered with a general mesh-repair algorithm.

Volume assembly should reuse shared interface identities and orientation.
Domain closure and volume-element generation remain subsequent work, but must
not require another independent contact-repair stage.

## Agreed Contact Rules

### Shared Cells

The active extraction uses a common final octree level. A vertex's integer cell
ID is the primary key for overlap handling. Eligible surfaces with vertices in
the same cell may share a position, even if their continuous surfaces would not
intersect at finer resolution.

Coarse-resolution sticking is accepted. Do not require exact scalar-field or
triangle intersections to authorize sharing. Resolution artifacts are secondary
diagnostics, not the central design problem or automatic test failures.

### Structural Group Isolation

Never merge surfaces in the same structural group, even when they share a cell.
At the engine layer, use the surface-to-stack mapping to enforce this rule.

The exclusion also applies indirectly. For example, if A and B belong to one
group and C belongs to another, averaging A-C and B-C into one common position
must not collapse A and B into a shared contact identity.

Do not group every vertex in a voxel into one unrestricted connected component.
Define eligible contact sets using geological relations, with at most one
surface from each group in a shared set. Where several same-group surfaces
compete for another group's vertex, first exclude ineligible geological roles
and wholly hidden ordinary cells. Order remaining candidates by original squared
vertex distance, then stable `(stack_index, surface_index)` identities. Merge
sets only if their group identities are disjoint and every cross-set ordinary
pair is allowed. Tests cover competing contacts and traversal-order independence.
Do not silently resolve this ambiguity by transitive merging or processing order.

Same-group pinch-outs do not override this exclusion. Smooth approach of
interfaces comes from interpolation; no forced same-group weld is required.

### Non-Fault Position Sharing

For an eligible cross-group contact set, compute a common mean once from the
original participating positions and assign it to all members.

Two-member sets reduce to the existing pairwise average. Multi-member sets must
not use sequential pairwise averaging: it leaves different final positions and
can depend on traversal order. Preserve eligible-set membership and shared IDs
alongside the positions for later topology/volume assembly.

These sets are provisional until triangle handling finishes. For ordinary
contacts, retain only members referenced by surviving triangles, then recompute
the mean from their original positions. Restore unsupported members to their
original positions and clear their IDs; restore the remaining member as well
when a set becomes a singleton. Do not regroup competing horizons. Surviving
IDs remain stable (and may contain gaps). Fault-anchored sets retain their
directional assignments and overlap metadata, including discarded targets.

### Faults

Use the working `pretty` fault behavior as the reference, not symmetric averaging:

- The affected layer takes the fault vertex position; the fault remains the controlling surface.
- Directional fault relations determine which layers are affected.
- Preserve the existing fault-overlap triangle treatment.
- Do not reinterpret unrelated fault pairs as erosion/onlap contacts.

Do not globally disable or redesign working fault QEF preparation to satisfy
the superseded independent-surface accuracy contract. Characterize any new-mode
QEF eligibility changes explicitly, particularly same-group leakage, without
changing legacy behavior.

Finite faults and smooth fault termination remain later corner cases. Ordinary
bounded extraction patches are not geologically finite surfaces, and their
independent mesh boundaries should not define the geological contact rule.

### Erosion/Onlap Connectivity

Extend fault-style cell-local handling to erosion/onlap using geological
relations, corner ownership, valid edge crossings, and ordered incident cells.
These determine which triangles continue, terminate, or represent a shared
interface. Shared cell positions alone do not authorize arbitrary triangle
deletion.

Retain the current single-surface connectivity outside contact cells. Prefer
local modifications to existing connectivity over global triangle intersection,
Boolean operations, or independent-patch remeshing.

Shared positions can remain separately indexed in output meshes initially.
Retain a stable common contact identity so later volume assembly does not have
to infer sharing from approximate coordinate equality.

## Architecture And Style

- Keep `DualContouringOverlap.contact_aware` as the opt-in setting, with explicit `match` dispatch.
- Preserve `none`, `pretty`, `watertight`, existing defaults, and legacy-only Flag behavior.
- Put computation in procedural modules with explicit inputs and returned outputs.
- Modules must not import or call each other; the API passes data between them.
- Avoid global/backend state changes, caller-array mutation, and unnecessary classes or frameworks.
- Reuse existing extraction, cell coding, triangulation, and fault operations wherever practical.
- Keep new contact reconciliation after ordinary dual contouring where possible; pass extraction metadata rather than reconstructing it from final triangles.

Existing useful metadata:

- Common integer leaf-cell coordinates and lattice bounds.
- Per-surface active-cell to dual-vertex mapping.
- Twelve edge-crossing flags per cell and corresponding intersection/gradient samples.
- Corner scalar values and geological ownership classifications.
- Surface/group identities, stack relations, and directional fault relations.
- Canonical grid-edge keys and incident-cell ordering used by triangulation.

The API has access to masks and scalar outputs that are not currently stored in
`DualContouringData`. Pass those arrays explicitly. Extraction candidate masks,
actual edge crossings, and geological ownership are distinct inputs; do not
substitute one for another.

Proposed flow:

```text
API: collect extraction and geological metadata
    -> module: identify eligible cell-local contact sets
API: pass sets and original vertex arrays
    -> module: compute shared positions and contact identities
API: pass identities, relations, ownership and local connectivity
    -> module: determine local triangle changes
API: pass retained faces, original positions and provisional contact groups
    -> module: finalize supported ordinary memberships and original-position means
API: assemble returned meshes and reports
```

The exact module split should follow reusable operations and test boundaries.
Do not introduce a general contact framework merely to realize this diagram.

## Revised Tests And Benchmarks

Primary correctness checks:

- Eligible cross-group contacts share positions and stable contact identities.
- Same-group surfaces are never merged directly or transitively.
- Multi-way contact results are independent of surface processing order, with geological precedence held fixed.
- Fault-to-layer position copying and directional triangle behavior match the working implementation.
- Erosion/onlap triangle eligibility follows corner ownership and stack relations.
- Non-contact cells retain their existing geometry/connectivity.
- Triangle indices remain valid; local changes do not create unintended duplicate faces or invalid topology.
- Modules leave input arrays untouched and API orchestration preserves caller state.
- Existing modes match their numerical references.

Add a mandatory competing-contact case: two surfaces in group A plus one in
group B, all in one cell. It must exercise the no-transitive-same-group-merge
rule, not merely the direct same-stack pair exclusion.

Nearby parallel surfaces from eligible different groups sharing a voxel are an
accepted sticking example. Do not retain the planar prototype's requirement
that they remain geometrically isolated. Same-group parallel surfaces must stay
distinct. Distinguish post-processing eligibility from inherited QEF behavior
in the tests so they do not accidentally enforce contradictory contracts.

Keep existing legacy characterization references. Treat plane-fitting and
triangle-clipping tests as prototype coverage, not the new mode's acceptance
criteria. Update or retire superseded new-mode tests deliberately, rather than
making broad skip/xfail changes.

Benchmark preparation, cell grouping/position sharing, local topology handling,
full extraction, and end-to-end computation. Vary resolution, groups, surfaces
per group, and sparse/dense overlap. Include fault and no-overlap baselines.
Record cell/contact counts, vertices, triangles, runtime, and process/device
memory as appropriate. Avoid fragile timing thresholds in correctness CI.

Affine-only, NumPy/float64-only, and two-stack restrictions belong to the current
prototype, not the agreed voxel-based target. Establish supported backend/dtype
behavior with tests; do not carry those restrictions forward merely because
strict planar clipping required them.

## Implementation Sequence

1. Specify and test eligible contact-set construction, including competing same-group surfaces and fault precedence.
2. Replace the planar-only orchestration with metadata-driven cell grouping and one-shot position sharing. Retain opt-in dispatch and legacy references.
3. Reuse/preserve fault behavior and implement erosion/onlap local triangle rules using ownership and existing cell/edge incidence.
4. Add multi-way contacts, chains, curved-field examples, and backend/dtype coverage without a plane-fitting gate.
5. Benchmark the revised mode and retain contact identities needed by lithological volume assembly.

Steps 1-3 and initial portions of steps 4-5 are implemented. Broader compound
models, GPU verification, repeatable performance distributions, and closed
volume assembly remain pending. The independent planar mesh-correction modules
and their direct tests are retained as prototype utilities, but are not called
by the new runtime path or used as its correctness oracle.

## Runtime Snapshot

- `modules/dual_contouring/contact_cells.py`: sparse cell buckets, disjoint eligible sets, original-position means, fault anchors, stable contact IDs, and conflict diagnostics. It imports no other computational module.
- `modules/dual_contouring/contact_topology.py`: group roles, erosion/onlap chain dependencies and boundary thresholds, corner-ownership removal, directed fault triangle removal, and shared-ID patch handling. It imports no other computational module.
- `API/dual_contouring/contact_reconciliation.py`: prepares relations and fault QEF partner sets, then orchestrates vertex and triangle operations with explicit array inputs.
- `API/dual_contouring/multi_scalar_dual_contouring.py`: collects once, extracts once, passes per-vertex cell/ownership metadata, and preserves the original solve tensor and caller state.
- `weighted_qef_setup_multicore.py`: optional explicit partner override, including empty sets; omitted override preserves legacy preparation.

Ordinary contact grouping excludes a cell only when all its corners are
unowned. Partial ownership preserves boundary support. Topology removes wholly
unowned ordinary triangles even when the controlling null-space group exports
no mesh. Fault interfaces are not lithological volume owners and must not be
removed merely because their ownership mask is false.

After triangle removal, `finalize_cell_vertices` prunes ordinary memberships
without retained triangle support. This prevents an onlap overlap strip from
leaving a second displaced substrate row after its target triangles disappear.
Reports distinguish `provisional_contact_count`, final `contact_count`,
`unsupported_contact_member_count`, and `dissolved_contact_count`. Position
conflict diagnostics and topology reports still describe the provisional pass.
The manual five-example PyVista viewer lives in `examples/contact_aware_smoke.py`,
outside pytest/CI collection; see `examples/README.md` for invocation.

New-mode QEF preparation includes only directed fault partners in both solve
directions, excluding fault-fault constraints as in the working preparation.
Ordinary sharing does not inject broad cross-surface QEF constraints. Fault
anchors refer to the extracted solve positions at entry to reconciliation, not
to unconstrained scalar-field intersections.

Fault sets are processed before ordinary sets and cannot be overwritten by an
ordinary mean. Competing fault controllers or fault chains are diagnosed rather
than averaged. Same-group exclusion also applies to competing fault targets;
rejected snaps remain distinct. All actual directed fault-overlap target indices
are retained for the existing all-three-indices triangle-removal rule.
After topology and finalization, the narrowly supported junction-attachment pass
below may copy an ordinary boundary endpoint to an existing anchor. This is not
an ordinary mean or a change to fault eligibility.

Triangle operations retain vertex indices, cell correspondence, and winding.
Redundant target patches are removed only when all three distinct contact IDs
match a supported controller triangle. Chains retain a surviving controlling
copy; cycles that would delete every copy fail explicitly. No new vertices,
plane fits, geometric intersection gate, or global retriangulation are required.

Each mesh's `contact_report` retains its per-vertex `contact_ids` (`-1` when
unshared), network contact/conflict counts, and topology removal counts. IDs are
deterministic for fixed cell/member sets under input reordering, not persistent
identifiers across changes in model geometry or resolution. `vertices_tensor`
and support diagnostics describe the original solve/pre-contact triangulation;
returned NumPy mesh positions are detached, with no differentiability claim.

Current support and limits:

- CPU extraction is tested with NumPy and PyTorch, including float32 and float64; GPU execution has not been verified.
- Detached arrays participating in sharing must use one common floating dtype, ensuring identical assigned positions.
- Multiple groups/surfaces, curved fields, RAW/INTERSECT/DISJOINT extraction, and non-exported null-space groups are covered without an affine-fitting gate.
- Extent capping is rejected in this mode until added boundary vertices receive consistent contact identities. Legacy extent-capping behavior is unchanged.
- Matching voxel positions and shared patch identities do not establish closed lithological solids or valid simulation volume elements.

Focused verification from the repository root:

```bash
DEFAULT_BACKEND=numpy PYTHONPATH=. /home/leguark/.venv/2025/bin/pytest \
  tests/test_common/test_modules/test_contact_cells.py \
  tests/test_common/test_modules/test_contact_topology.py \
  tests/test_common/test_modules/test_contact_aware_integration.py \
  tests/test_common/test_modules/test_compound_contacts.py \
  tests/test_common/test_modules/test_fault_junctions.py \
  tests/test_common/test_modules/test_contact_reconciliation.py \
  tests/test_common/test_modules/test_contact_reconciliation_integration.py \
  tests/test_common/test_modules/test_quad_triangulation.py \
  tests/test_common/test_modules/test_extent_capping.py \
  tests/test_common/test_modules/test_extent_capping_integration.py -q
```

The integration coverage includes real fault QEF execution on both CPU backends
and exact NumPy mesh parity with `pretty` for the two-surface directional fault
cases. Legacy numerical/connectivity references remain separate checks.

Enable the benchmark matrix with `GEMPY_CONTACT_AWARE_BENCHMARKS=1`. The prepared
cell suite covers sparse/dense overlap, no overlap, multiple horizons per group,
and directed fault snapping. Extraction/end-to-end cases compare the new mode
with legacy modes. Do not dispatch `contact_aware` through the legacy overlap-only
benchmark: its replacement is the dedicated `contact-cells` scope.

```bash
GEMPY_CONTACT_AWARE_BENCHMARKS=1 GEMPY_CONTACT_BENCHMARK_ROUNDS=1 \
  DEFAULT_BACKEND=numpy /home/leguark/.venv/2025/bin/pytest \
  tests/benchmark/test_benchmark_contact_cells.py \
  tests/benchmark/test_benchmark_finalize_cells.py \
  tests/benchmark/test_benchmark_contacts.py --benchmark-only -q

/home/leguark/.venv/2025/bin/python tests/benchmark/contact_benchmark_runner.py \
  --scope contact-cells --case dense --mode contact_aware --size 256 --dtype float64
```

One-round runs and isolated RSS runs are smoke verification, not stable speedup
claims or stage-only memory measurements. Full fault-model timing and larger
sparse multi-group workloads still need repeatable measurements.

Recorded milestone verification:

- 318 tests passed across cell/topology/integration, legacy references, retained planar utilities, quad triangulation, and extent-capping suites.
- Three existing fault-QEF production tests passed with the default mode and again with `contact_aware`; the higher-resolution case was deliberately deselected from these smoke runs.
- 64 benchmark smoke cases passed with the new-mode opt-in enabled. Four `contact_aware`/legacy-overlap-only combinations were deliberately skipped in favor of the prepared cell suite.
- Isolated RSS runners completed for prepared dense contacts and production unconformity extraction in the new mode.
- Independent review reproductions confirmed the null-space ownership and fault-fault QEF fixes. No full-repository or GPU verification was performed.

## Compound Contact Validation

The extraction acceptance matrix now includes partitioned three-stack erosion
and onlap ownership, three-way junctions, curved contacts, same-group competition,
and mixed fault/ordinary contacts on both CPU backends and floating dtypes.
These are analytic field/gradient extraction fixtures, not full kriging models.
Checks cover retained face indices/order, positive area, orientation relative to
the original faces, connected retained patches, edge incidence at most two per
individual surface, equal cells/positions per contact ID, and unchanged unshared
positions. The ordinary seam cases require every internal target boundary edge
to match an actual eligible controller edge by its pair of contact IDs; vertices
merely having equal positions is insufficient. Domain boundaries are identified
using the common grid's integer-cell limits. Directed controller/target surfaces
must not retain duplicate fully shared triangle patches.

`test_compound_contacts.py` additionally exercises a redundant middle patch in
a three-surface truncation chain. Triangle removal drops that participant and
one complete shared cell, while a two-member attachment survives. It checks
original-position recomputation, restored unsupported rows, matching attachment
edges, no collapsed/inverted retained faces, stable IDs/positions/faces under
every surface permutation, same-group isolation, and an adjacent directional
fault anchor, in float32 and float64. This validates those constructed cases,
not a universal guarantee against under-resolved geometric defects.

Prior validation milestone: **396 focused regression tests passed**, including the
new compound checks and existing legacy references, plus **all four production
weighted-QEF tests** in `contact_aware`, including higher resolution. The twelve
finalization benchmark cases passed when enabled and were skipped when disabled;
three larger production-model benchmark cases passed. That milestone included a
fault-junction gap characterization, now replaced by connectivity acceptance.

### Supported Fault-Junction Attachment

In the extracted `fault_mixed` case, fault stack 0 controls stack 1 but not stack
2. Stack 1 truncates stack 2. At cell `(2, 2, 2)`, provisional grouping correctly
rejects the ordinary merge into a fault set. Previously this left two unmatched
internal stack-2 boundary edges:

- `(1, 2, 2) -> (2, 2, 2)`
- `(2, 2, 2) -> (3, 2, 2)`

`attach_fault_junctions` now closes this specific gap after face removal and
ordinary membership finalization. The policy is a cell-local seam endpoint
attachment, not broad absorption of an unaffected stack into a fault:

- Only an unshared row on a surviving ordinary target boundary may attach, via an existing directed truncation relation.
- The ordinary controller must already belong to a fault-anchored contact in the same integer cell.
- A previously shared neighboring endpoint must establish an actual retained controller edge. Every already shared boundary neighbor must match such an edge in the same controller.
- Targets participating on either side of the directed fault graph are excluded. No fault chain, new fault partner, overlap-removal target, QEF constraint or masking relation is introduced.
- All ordinary cross-pairs in the extended contact must be allowed, and each contact still contains at most one member per stack. Competing candidates use original distance and stable surface IDs.
- The endpoint copies the controller's finalized position and existing contact ID. Existing members and already shared target rows are never moved or regrouped.
- Every incident target face must remain nondegenerate and correctly oriented relative to original extraction. Attachments creating duplicate fully shared patches are rejected.
- Candidate edge evidence uses snapshot IDs, not newly accepted attachments, so this pass is not iterative transitive closure.

Reports add `fault_junction_attachment_count`, `fault_junction_rejected_count`,
`fault_junction_attachments`, and `fault_junction_rejections` when this pass runs.
The original `fault_anchored` conflicts remain provisional-grouping diagnostics;
the attachment records describe their narrowly supported endpoint resolution.
Contact counts and surviving IDs remain stable because no new group is created.

`test_fault_mixed_closes_junction_without_moving_fault_anchor` confirms one
attachment closes both missing edges while retaining the original fault position.
The mixed-fault case now participates in the same edge/connectivity acceptance
matrix as ordinary contacts on both CPU backends and dtypes. Separate unit tests
cover isolation, multiple allowed target stacks, absent/misdirected edge evidence,
non-propagation, controller immutability, geometry and duplicate-patch rejection.
These are fixture-level connectivity guarantees, not a claim that arbitrary
under-resolved junctions are repaired. Rejected cases remain distinct and need
refinement or a richer junction representation. Closed shells, extent capping,
finite-fault corner cases and GPU validation remain deferred.

Post-attachment verification: **420 focused regression tests passed**, including
the mixed-fault edge acceptance matrix and twenty attachment unit cases. **All
four production weighted-QEF tests passed** in `contact_aware`, including the
higher-resolution case. Existing legacy reference and two-surface fault parity
checks remain green. No full-repository or GPU run was performed.

### Performance Measurements

The opt-in finalization suite times only `finalize_cell_vertices`, excluding
preparation and external fixture copies (the function's own output copies remain
timed). Its prepared three-member contacts either retain
all members or discard one unsupported member per group. At 8192 contacts
(24,576 vertices), five-round float64 medians with CPU/BLAS threads pinned to one
were **6.37 ms supported** and **33.13 ms pruned**. The supported run included a
176.88 ms outlier; these short measurements are not a stable scaling guarantee.
Fresh-worker lifetime peak RSS was about **607 MiB** for either case, including
imports, preparation, copies, and execution, not stage-only allocation. Current
`/proc` RSS and `ru_maxrss` differed in those workers; no allocation delta is
inferred from them.

Before the junction-attachment addition, the production unconformity benchmark (root `8^3`, two octree levels,
4096 leaf cells) passed for `none`, `pretty`, and `contact_aware`. Five-round
end-to-end medians were 2.198 s, 2.187 s, and 2.120 s respectively, with lifetime
peak RSS around 806-809 MiB. The new mode retained 52 contact sets and 1555
triangles, versus 1622 triangles in the legacy modes. These small timing
differences do not establish a speedup. The previously omitted production
weighted-QEF `higher_resolution` test also passed in `contact_aware` on NumPy.
GPU execution and full-repository verification remain outstanding.

```bash
GEMPY_CONTACT_AWARE_BENCHMARKS=1 GEMPY_CONTACT_BENCHMARK_ROUNDS=5 \
  DEFAULT_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/leguark/.venv/2025/bin/pytest tests/benchmark/test_benchmark_finalize_cells.py -q

/home/leguark/.venv/2025/bin/python tests/benchmark/contact_benchmark_runner.py \
  --scope finalize-cells --case compound_pruned --mode contact_aware --size 8192 --dtype float64
```

## Deferred Work

- Closed lithological solids and simulation volume-element generation.
- Finite-fault termination and smooth-decay corner cases.
- Automatic refinement or detailed recovery of under-resolved geometry.
- General mesh Boolean/intersection machinery; it is not required by this plan.
