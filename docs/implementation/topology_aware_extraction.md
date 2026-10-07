# Experimental Joint Dual Contouring

## Corrected Scope

`gempy_engine.API.dual_contouring.topology_aware_extraction.extract_topology_aware`
is an explicit NumPy CPU entry point for bounded uniform/conforming cells. It is
not dispatched by `compute_model` or any overlap enum. Existing production code,
defaults, and outputs are unchanged. Closed geological volumes and caps are not
part of this experiment.

The previous Freudenthal/marching-tetrahedra surface prototype was removed from
this API and module. Its 984-triangle comparison and
`/tmp/opencode/topology-aware-comparison.png` are **historical**, not evidence for
the corrected dual implementation. Exact affine seam projection was not the
reason for this correction: retaining dual contouring and ordinary mesh quality
was. Coarse contact sticking remains a legitimate discretization tradeoff; no
monotonic convergence or skinny-triangle-free mesher claim is made.

## Input Contract

- Three strictly increasing, finite, uniformly spaced node axes; unequal spacing between axes is allowed. At most 4096 cells. No hanging faces, adaptive octree transitions, caps, or boundary completion.
- `scalar_samples` and signed `ownership`: `(S,nx,ny,nz)` in `indexing='ij'` order. `gradient_samples`: `(S,nx,ny,nz,3)`. All values must be finite. Actual supplied gradients are interpolated onto the original Hermite crossings; they are **not forced to equal a corner-fit plane or interpolant gradient**. Differing tangent planes are normal QEF input, not an automatic rejection.
- Unique stable `(stack_index, local_surface_index)` identities, stack relations, and complete per-stack isovalue arrays. The API subtracts selected isovalues. IDs are invariant to surface input reordering when these identities are retained.
- Sampled topology supports cell-affine corner approximants, bilinear sampled extrusions, and nonaffine sampled cells that are strictly monotone along one axis and have one connected boundary contour. The latter use a consistent bilinear primal-face decider, not independent tangent-plane intersections. Other interiors or multiple uncertified boundary contours are explicitly unresolved. These assumptions describe sampled approximants, not proof that the original geological field has no unsampled features.
- By default, junctions require **cell-affine corner approximants**, with a nondegenerate approximate intersection segment supported by neighboring primal faces. The original fields and Hermite normals can be curved and varying. This includes additive/separable sampled fields such as the gallery paraboloid. The opt-in `sampled_junctions=True` extension below supports bounded nonaffine sampled contacts. Single-cell coincidences and parallel disjoint approximants do not establish a junction. Grid-edge/corner-aligned endpoints still require a sided incidence rule or refinement and are rejected.
- Ordinary ownership is positive retained, nonpositive hidden. By default, a changing mask must equal exactly one eligible directed controller's isovalue-subtracted samples or their negative within absolute tolerance `1e-10`. The sampled-junction option additionally accepts the exact minimum of all eligible negative controller fields, only when other controllers have uniform sign in each contact cell. Boolean and arbitrary scaled masks remain unsupported. All-positive/full-hidden surfaces need no joint controller.
- Optional `interior_samples` are finite, **isovalue-subtracted** cell-center values of shape `(S,nx-1,ny-1,nz-1)`. A center revealing a zero/sign crossing absent from unanimous corner signs is rejected as insufficient interior evidence. Curved center values need not equal the corner average. Their maximum approximant residual is reported; passing the check is not general interior-topology certification.
- Fault interfaces, directed faults, and finite-footprint/slip metadata are explicitly rejected. There is no corrected dual finite-fault or bank-incidence adapter yet.

Inputs are not mutated. Tolerances currently assume reasonably scaled scalar
samples and geometry; sample-aligned zeros, saddle ties, near-dependent junction
constraints, zero Hermite gradients, and degenerate emitted faces are explicit
errors rather than arbitrarily displaced geometry.

## Generation Algorithm

### Primal Evidence And Branch Identity

1. Reuse production `find_intersection_on_edge(..., strict_crossings=True)` on
   the original 12 cube edges. Interpolate supplied gradients and ownership at
   these crossings. Canonical edge identities are lower integer endpoint plus
   axis, matching production complete-quad triangulation.
2. Construct a graph of crossing edges connected through each primal face. Two
   crossings connect directly. Four crossings use the bilinear asymptotic
   determinant `f00*f11-f10*f01`, with consistent cyclic face ordering on both
   neighbors. A saddle on the zero set is deterministically rejected.
3. Connected boundary contours identify branch-specific Hermite subsets. An
   affine corner approximant has one branch. Bilinear sampled extrusions support
   disconnected hyperbola branches. A nonaffine ordinary cell can also be accepted
   when its sampled trilinear interpolant is strictly monotone along one axis and
   the face graph has one branch; multiple uncertified contours remain unresolved.
   The sampled evidence does not certify every feature of the original field.

The test field `(x-.5)*(y-.5)-.1`, extruded in z, establishes two separate
southwest/northeast branches in the same cell. Tests verify the actual edge
pairings, each representative's own production QEF, and two disconnected output
components, not merely a count of two representatives.

### Hermite And Joint QEF Constraints

4. For every ordinary branch, call the existing
   `generate_dual_contouring_vertices` with its Hermite subset and unchanged
   three unit-strength mass-point bias rows. Uncomplicated cells therefore use
   the production solver literally, not a replacement QEF approximation.
5. For a supported two-interface truncation contact, use the existing
   `build_contact_relations` geological graph to select the unique controller.
   Intersect both cell-affine **corner approximant** zero planes with the cube and
   record matching segment endpoints on canonical primal faces. Shared corner
   data make these face restrictions consistent between neighbors; independently
   fitted Hermite tangent planes are not used to invent neighboring seam endpoints.
   A shared identity requires this evidence and group-distinct participants.
6. Allocate one shared **cell dual representative**, not a primal intersection
   vertex. Its QEF contains both participants' original Hermite plane rows and
   one combined-Hermite mass-point bias, subject to the **approximate sampled
   seam**, not exact original curved-surface equations. Minimize the actual QEF
   on its bounded segment, retaining every supplied Hermite row's magnitude.

For Hermite points `p_i`, normals `n_i`, combined mass `m`, and approximate seam
`x(t)=p+t*d`, minimize

```text
E(t) = sum_i [n_i dot (p + t*d - p_i)]^2 + ||p + t*d - m||^2
t* = [sum_i (n_i dot d)(n_i dot (p_i - p)) + d dot (m - p)]
     / [sum_i (n_i dot d)^2 + d dot d]
```

Clamp `t*` to the cube-supported segment interval before any mesh emission.
The mass bias makes this one-dimensional Hessian positive; nonfinite objectives
are explicit errors. Curved tangent residuals generally do **not** vanish. Only
for compatible planar Hermite rows does the solution reduce to mass projection.
Independent KKT tests with actual noncoincident curved tangent rows and two scalar
weights verify optimality and a measurable improvement over mass-only projection.

This shared representative replaces the ordinary representative for each
participating single branch before incidence or triangles are emitted. It does
not average independently extracted mesh vertices. Ordinary branches remain
separate; same-stack participants cannot share directly or transitively. A
second contact competing within the same cell is rejected rather than merged.
General multiway cells and extra junction-specific branch representatives are
not implemented in this bounded case.

### Dual Incidence And Emission

7. Plan one complete dual quad per retained canonical crossing edge using its
   four incident cells and each local edge's branch label. The ownership decision
   is made at the shared Hermite crossing, not by clipping or removing a triangle.
   Crop-boundary edges without four cells emit no quad, matching production.
8. Map participating cell incidences to the shared dual representative. Before
   emitting any triangles, require each interior supported junction face to
   correspond to a dual seam edge with exactly two controller-side quad
   incidences and one retained-target incidence. Missing/mismatched support,
   unsupported fully shared triangles in the planned quad split, and competing participants reject the
   whole call. There is no partial successful repair result.
9. Emit the production complete-quad 1--3 split, `[0,1,3]` and `[2,3,1]`, with
   production Hermite-normal winding. Check nonzero area and resolvable winding.

The controller's regular quad incidence already provides its two sides at the
seam in these supported cases, so no extra patch subdivision is necessary. A
configuration needing additional controller/junction incidences is rejected,
not forced through a single representative. Transitions between regular and
joint cells are in the same quad-generation plan; independently extracted
meshes are never stitched afterward.

The computational topology module imports no other computational module. API
code composes production edge/QEF primitives with local evidence and emission.
No contact reconciliation, contact-cell/geometry mesh operations, triangle
clipping, weld/snap/averaging, mesh Boolean, VTK cleaning, tetra surface emission,
or mesh-quality repair is used.

## Output And Parity

The result contains global `vertices`, per-surface `faces`, stable `vertex_keys`,
cell coordinates/component counts, local `branch_labels`, canonical
`primal_edges` per triangle, `junction_cells`, shared `seam_edges`, and
`affected_faces`. Original `hermite_data` is available for independent parity
checks. Unreferenced ordinary branch representatives are retained as local
evidence, including for an open one-cell crop.

Only actual junction-cell representatives change. `affected_faces` is exactly
the quads incident to those representatives, not an expanded arbitrary band.
All retained ordinary vertices and all nonaffected oriented triangles match
existing DC exactly after stable-key index mapping. Global array indices change
when shared keys replace separate representatives; bitwise numeric-ID equality
across those different pools is not claimed.

`include_reference=True` adds a diagnostic independent reference from the same
production branch QEFs, identical retained primal-edge decisions, and the same
quad split. It is emitted after the joint plan is established and is never used
to repair experimental output. Six no-contact plane cases (all axes, both normal
directions) additionally compare
against production `triangulate_quads` directly, asserting identical arrays
without key remapping: 64 vertices and 98 triangles at eight cells per axis.

## Viewer And Quality

The optional PyVista viewer now defaults to an equivalent-input comparison:
**Existing DC QEF / identical support** versus **Joint dual contouring**. This is
not labeled as a new `compute_model` integration or as the production
post-reconciled contact-aware mode. Both panels use identical original Hermite
samples, gradients, retained crossing-edge support, and quad convention.

The default `--case planar` analytic onlap fields remain `f0=x-.25*z`, level `.4`, and
`f1=.25*x+z`, level `.6`, with `[ONLAP, BASEMENT]`. The target keeps `f1-.6>0`.
Colors highlight junction-incident triangles in blue/orange and regular patches
in gray/tint, cameras are linked, and shared dual seam edges are white. The analytic guide is optional
and is not the quality acceptance metric.

```bash
CUDA_VISIBLE_DEVICES='' DEFAULT_BACKEND=numpy DEFAULT_PYKEOPS=False PYTHONPATH=. \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  LIBGL_ALWAYS_SOFTWARE=1 /home/leguark/.venv/2025/bin/python \
  examples/topology_aware_comparison.py --off-screen
```

Current artifacts:

- `/tmp/opencode/topology-dual-comparison.png`
- `/tmp/opencode/topology-dual-quality.json`

Minimum angles are in degrees. Aspect ratio is **longest edge / smallest
altitude**, equivalently `Lmax^2/(2*area)`; an equilateral triangle has aspect
`2/sqrt(3)`. The viewer reports `[0,5,50,95,100]` percentiles of per-triangle
minimum angle and aspect ratio for all, affected-contact, and noncontact regions.
These expose regressions; they are not universal quality thresholds.

Measured eight-cell onlap comparison:

| Quantity | Existing QEF Reference | Joint Dual |
| --- | ---: | ---: |
| Total triangles | 196 | 196 |
| Junction-incident triangles | 42 | 42 |
| Noncontact triangles | 154 | 154 |
| Overall minimum angle | 27.266 degrees | 27.266 degrees |
| Contact minimum angle | 42.852 degrees | 29.154 degrees |
| Contact median aspect | 2.005635 | 2.114601 |
| Shared-ID seam edges | 0 | 7 |

All 154 noncontact triangles and 144 retained regular vertices match exactly.
The local contact triangle quality **degrades** in this example even though seam
incidence is shared. No global triangle inflation occurs, and no quality cleanup
is used to conceal the degradation. Full percentile distributions are in the
JSON artifact, not just the minimum or seam residual.

### Curved Erosion Smoke

`--case curved` uses the coordinator's original fields and actual analytic node
gradients:

```text
f0 = z - .43 - .25*(x-.5)^2 - .15*(y-.5)^2
grad(f0) = [-.5*(x-.5), -.3*(y-.5), 1]
f1 = x - .57; grad(f1) = [1, 0, 0]
ownership = [1, -f0]; relations = [ERODE, BASEMENT]
```

These corner samples fit a local affine approximant, but the original field is
curved and its Hermite tangent planes differ. Ordinary cells use those actual
normals and the production QEF unchanged. The joint representative solves the
actual constrained objective above. This smoke produces **actual dual triangles
and shared seam edges**, not an analytic guide or a swallowed rejection.

```bash
CUDA_VISIBLE_DEVICES='' DEFAULT_BACKEND=numpy DEFAULT_PYKEOPS=False PYTHONPATH=. \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  LIBGL_ALWAYS_SOFTWARE=1 /home/leguark/.venv/2025/bin/python \
  examples/topology_aware_comparison.py --case curved --off-screen
```

Artifacts are `/tmp/opencode/topology-dual-curved-comparison.png` and
`/tmp/opencode/topology-dual-curved-quality.json`. Both panels use equivalent
Hermite data and retained primal-edge support. The JSON includes per-surface
absolute analytic scalar residuals at unique referenced vertices, contact-region
vertices, and the actual shared seam, alongside full quality percentiles.

Measured at eight cells per axis:

| Quantity | Existing QEF Reference | Curved Joint Dual |
| --- | ---: | ---: |
| Total triangles (surface counts) | 164 (122 / 42) | 164 (122 / 42) |
| Contact / noncontact triangles | 42 / 122 | 42 / 122 |
| Shared seam edges | 0 | 7 |
| Contact minimum angle | 44.691 degrees | 39.882 degrees |
| Contact median aspect | 2.000 | 2.003 |
| Contact maximum aspect | 2.014 | 2.235 |
| Maximum curved-surface scalar residual | 0.000960689 | 0.001548437 |
| RMS curved-surface scalar residual | 0.000797403 | 0.000887083 |
| Maximum plane-surface scalar residual | 0 | 1.11e-16 |

All 122 noncontact triangles and all 128 retained regular representatives match
exactly. Noncontact quality distributions are identical. Joint contact quality
and curved-surface residuals degrade locally; this is reported, not cleaned away.
The nonzero curved residual is accepted discretization error, not an exact
analytic-projection claim or a universal convergence guarantee.

Ordinary curved fields with nonaffine sampled data are also tested using a graph
with a mixed `x*y` term and varying true gradients. Those ordinary cells extract
with native production QEF/quad parity. Their junctions remain unsupported by
the default affine contract; the opt-in sampled-face extension below supplies
consistent approximate boundary constraints. No claim of general curved-junction
extraction or production dispatch is made.

## Finite-Fault Status

The existing actual-engine characterization is retained in
`test_topology_finite_fault_characterization.py`. At two octree levels, a real
fixture with strike/dip radius `.75` measured unchanged fault meshes (24 vertices,
30 triangles, maximum vertex delta zero), 22 referenced vertices outside taper
support, maximum drift delta `1`, and maximum dependent scalar delta `4.60046637`.

Finite slip support and a displayed fault isosurface are distinct semantics;
this observation is not automatically a defect or a demand to clip that surface.
The nonnegative `finite_fault_scalar` multiplier is not a signed geometric
footprint. Plateau zeros cannot identify a subcell termination. Production
integration requires conforming sample identities, actual footprint/tip
constraints when that geometry is intended, and possibly separate bank fields
and ownership for discontinuous displacement.

The previous synthetic continuous target with caller-supplied vanishing slip
was a controlled tetra-reference extraction test, **not physical fault-bank
integration**. It is not presented as passing evidence for the corrected dual
implementation. Directed faults, competing fault controllers, and fault chains
now fail explicitly until their dual incidence and physical ownership contract
can be implemented. No fault controller is silently converted into ordinary
contact eligibility.

## Verification

```bash
CUDA_VISIBLE_DEVICES='' DEFAULT_BACKEND=numpy DEFAULT_PYKEOPS=False PYTHONPATH=. \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/leguark/.venv/2025/bin/pytest -q \
  tests/test_common/test_modules/test_topology_aware_extraction.py \
  tests/test_common/test_modules/test_topology_finite_fault_characterization.py \
  tests/test_common/test_modules/test_topology_failure_examples.py
```

Tests cover direct production parity, two-interface erosion/onlap shared seam
edges, exact unaffected-region parity, true bilinear two-branch QEFs and face
decisions, input reordering, same-group isolation, ownership, winding/area,
objective-level constrained QEF verification, quality formulas, and explicit
unsupported inputs. The 12 requested production contact/extent regression files
are verified separately under the same CPU-only environment. The corrected
prototype, retained characterization and gallery tests pass 52 tests; the
requested legacy files are verified separately (420 tests). Curved tests cover
actual supplied normals, ordinary/native parity, real shared incidence, winding,
no duplicate triangles, deterministic IDs, same-group isolation, true curved
center evidence, and actual constrained-QEF optimality. No CUDA computation is
used for this verification.

The requested legacy regression invocation (420 passed on CPU) is:

```bash
CUDA_VISIBLE_DEVICES='' DEFAULT_BACKEND=numpy DEFAULT_PYKEOPS=False PYTHONPATH=. \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/leguark/.venv/2025/bin/pytest -q \
  tests/test_common/test_modules/test_contact_cells.py \
  tests/test_common/test_modules/test_contact_topology.py \
  tests/test_common/test_modules/test_contact_aware_integration.py \
  tests/test_common/test_modules/test_compound_contacts.py \
  tests/test_common/test_modules/test_fault_junctions.py \
  tests/test_common/test_modules/test_contact_reconciliation.py \
  tests/test_common/test_modules/test_contact_reconciliation_integration.py \
  tests/test_common/test_modules/test_contact_planes.py \
  tests/test_common/test_modules/test_contact_geometry.py \
  tests/test_common/test_modules/test_quad_triangulation.py \
  tests/test_common/test_modules/test_extent_capping.py \
  tests/test_common/test_modules/test_extent_capping_integration.py
```

## Failure Gallery

`examples/topology_aware_failures.py` now exercises two successful cases: planar
onlap control and the actual curved erosion mesh. The four remaining rejection
cases are a contact on an interior grid face, competing three-way contacts,
scalar-rescaled bilinear branches, and a coordinate-rescaled plane. It checks
each expected error prefix and verifies that
the unscaled equivalents of both scale cases succeed. Those last two cases expose
absolute-tolerance sensitivity, not different physical zero sets.

Rejected panels show dense VTK contours of the unmasked analytic fields, clearly
labelled as geometry guides, **not experimental output or fallback extraction**.
Each display is normalized to its domain, so the tiny-coordinate case is magnified.
The original exact-Hermite/corner-fit rejection was removed by implementing the
curved QEF extension above; no tolerances were relaxed to hide the other four
cases. The previous curved rejection screenshot is historical, not current
extraction evidence.

```bash
CUDA_VISIBLE_DEVICES='' DEFAULT_BACKEND=numpy DEFAULT_PYKEOPS=False PYTHONPATH=. \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  LIBGL_ALWAYS_SOFTWARE=1 /home/leguark/.venv/2025/bin/python \
  examples/topology_aware_failures.py --off-screen \
  --screenshot /tmp/opencode/topology-dual-curved-gallery.png
```

Use `--check-only` to verify outcomes without importing PyVista, or
`--off-screen --screenshot /tmp/opencode/topology-dual-curved-gallery.png` to render
unattended. `test_topology_failure_examples.py` characterizes the six current
outcomes without rendering; its expectations should change when these limits are
addressed. A passing rejection test does not mean the missing feature is solved.

## Real Erosion Piles

`examples/topology_aware_models.py` computes the existing real fixtures without
changing observations, kernel settings, solve, stack relations, or production
extraction defaults. It deep-copies fixture inputs, adds node and center queries
to `custom_grid`, enables scalar gradients, and reads root-output
`custom_grid_slice` scalar fields and `gx_field/gy_field/gz_field`. Detached NumPy
arrays and actual per-stack surface-point isovalues drive experimental extraction.
This is **uniform node resampling**, not adaptive-octree joint extraction.

For ERODE/BASEMENT piles the adapter constructs signed conjunctions using minimum
and negative isovalue-subtracted controller fields. Their positivity matches
every actual `squeezed_mask_array` node exactly. For interface extraction only,
the interface's own stack threshold is excluded: it determines the volume side,
not external truncation along that same isosurface. No Boolean mask is passed as
a signed field. The unusual final ERODE in the simple fixture is preserved,
including its reciprocal constraints; it is not relabeled BASEMENT to force success.

### Bounded Sampled Junction Extension

`sampled_junctions=True` is explicit and defaults to false. Nonaffine contact
cells still require supported single-branch sampled topology. Each of their six
faces uses its canonical four-node bilinear restrictions for both participants.
A quadratic resultant enumerates isolated intersection roots; residual and
Jacobian checks reject unresolved, tangent, or coincident restrictions. Multiple
roots on one face, grid-edge-aligned roots, and anything other than two isolated
boundary roots per cell are rejected. The two roots define an **approximate
bounded chord**, on which the original combined-Hermite QEF is minimized. The
chord is not claimed to lie on either original kriging field or even exactly on
the trilinear interior intersection. It is not a marching surface construction.

Neighboring cells independently compute the same face restriction and must agree
on its physical intersection point. All existing primal-face support, 2:1 dual
incidence, fully shared patch, winding, and degeneracy guards remain enabled.
There is no unsafe diagnostic bypass, welding, snapping, clipping of meshes,
reconciliation, or post-extraction repair.

The option also recognizes exact sampled conjunctions of all eligible negative
controllers. In a candidate contact cell, every other controller must be uniformly
inactive or uniformly exclude the cell. Mixed competing-controller signs reject
the call, as do multiple contact participants needing the same cell representative.
This is a locally separable subset, not general composite/multiway extraction.

### Measured Results

| Fixture / Uniform Cells | Outcome | Surfaces | Joint Triangles | Shared Edges |
| --- | --- | ---: | ---: | ---: |
| `unconformity_complex_factory`, 16 cubed | Supported | All 4, in 3 stacks | 450 / 682 / 180 / 214 | 45 |
| `unconformity_complex_factory`, 8 cubed | Rejected competing cell participants | All 4 supplied | None | None |
| `simple_geometries.unconformity`, 8 and 16 cubed | Rejected reciprocal/competing cell constraints | All 3 supplied | None | None |

The supported real pile retains `[ERODE, ERODE, BASEMENT]`. Actual isovalues are
`[0.020090213263275203]`, `[-1.3621832466438971e-7]`, and
`[-1.8692829458652156, -1.4033048176069638]`. CPU `compute_model` takes about
18 seconds here. Its unchanged adaptive production meshes have
`3973 / 6155 / 1421 / 1851` triangles; these are **not** the same-data reference.
The reference literally uses independent production QEF representatives on
identical resampled Hermite data, ownership decisions, and retained quad support.

Both same-data panels contain 1,526 triangles, 278 contact and 1,248 noncontact.
All noncontact geometry, oriented connectivity, and 1,295 retained ordinary
representatives match exactly. There are 48 junction cells and 45 verified shared
faces, with maximum physical endpoint disagreement **zero**. Each of three
contacts has 15 actual shared-ID edges, each with two controller and one target
triangle incidences. The reference has zero shared-ID edges. There are zero
degenerate or duplicate triangles on every surface. Center-to-corner-average
scalar residual reaches `0.0029160838610282624`, showing that these are genuinely
nonaffine sampled fields, not fabricated affine approximants.

Contact quality degrades locally: minimum angle is `10.118365` degrees for the
reference versus `7.446124` for joint DC; median aspect is `3.520907` versus
`3.542025`, and maximum contact aspect is `5.847451` versus `7.720814`.
Noncontact quality distributions are exactly equal. Shared topology is not a
claim of improved triangle quality, exact field projection, general interior
certification, capped volumes, or global watertightness of this open crop.

### Commands And Artifacts

Run from the engine repository. Explicit dtype and overlap override local `.env`
settings; without `DUAL_CONTOURING_VERTEX_OVERLAP=none`, the pre-existing
finite-fault unchanged-vertices characterization fails under the local overlap
mode. No production setting or `.env` file is edited.

```bash
export CUDA_VISIBLE_DEVICES='' DEFAULT_BACKEND=numpy DEFAULT_PYKEOPS=False
export DEFAULT_TENSOR_DTYPE=float64 DUAL_CONTOURING_VERTEX_OVERLAP=none PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

/home/leguark/.venv/2025/bin/python examples/topology_aware_models.py \
  --case complex --resolution 16 --check-only

LIBGL_ALWAYS_SOFTWARE=1 /home/leguark/.venv/2025/bin/python \
  examples/topology_aware_models.py --case complex --resolution 16 --off-screen \
  --screenshot /tmp/opencode/topology-real-complex.png

LIBGL_ALWAYS_SOFTWARE=1 /home/leguark/.venv/2025/bin/python \
  examples/topology_aware_models.py --case unconformity --resolution 16 --off-screen \
  --screenshot /tmp/opencode/topology-real-unconformity.png

/home/leguark/.venv/2025/bin/pytest -q \
  tests/test_common/test_modules/test_topology_real_models.py \
  tests/test_common/test_modules/test_topology_aware_extraction.py \
  tests/test_common/test_modules/test_topology_finite_fault_characterization.py \
  tests/test_common/test_modules/test_topology_failure_examples.py
```

The successful screenshot compares independent same-data DC against joint DC,
with linked cameras, all four surfaces, and actual shared edges in white.
The rejected screenshot explicitly shows actual production DC and stack-0 scalar
nodes, **not** a successful joint mesh or substitute marching extraction. Its
production meshes contain `102 / 490 / 0` triangles; the empty third output is
reported and cannot be plotted as a nonempty surface. Matching `.json` files
beside the screenshots contain full metrics and rejection reasons.

Omit `--off-screen` for interactive viewing; no interactive window was opened
during verification. `--check-only` requires no viewer and prints the supported
or rejected result explicitly. The new real-model/face-root tests pass alongside
all existing 58 checks; the separate 420-test production regression command above
also passes with this explicit CPU environment.

## Actual Model7 Bank Experiment

`examples/topology_aware_model7.py` implements the actual
[g07_combination fault + unconformity](https://docs.gempy.org/examples/geometries/g07_combination.html)
experiment using the separately supplied, unchanged
`tests/fixtures/model7_combination.py`. All 102 surface points and eight
orientations remain present. World extent is `[0,2500,0,1000,0,1000]`;
the original three stacks remain `FAULT, ERODE, BASEMENT`, with directed
fault dependencies `0 -> 1,2`. All four surfaces (`fault, rock3, rock2, rock1`)
are represented. The existing real-model experiment and production solvers,
defaults, configuration, fault meshes and overlap behavior are unchanged.

### Genuine Bank Queries

`API/dual_contouring/model7_bank_experiment.py` captures detached NumPy copies
of the root-level production evaluator input, actual solved weights, and actual
evaluation options. Root metadata is used because custom grids are evaluated at
the root, not at the final adaptive octree level. Capture is a scoped
`unittest.mock.patch` around the original evaluator, which still runs unchanged;
it is process-local and **not thread-safe**, not a production API or dispatcher.

`query_bank` reuses `_evaluate_sys_eq`. Only query coordinates and
`FaultsData.fault_values_everywhere` change. Dependent query drift is exactly
0 or 1; `fault_values_on_sp`, `fault_values_ref`, `fault_values_rest`, observations,
and weights remain unchanged. Kernel construction consequently receives the
forced query drift through `faults_j`, while retaining the original solve-side
fault terms. These are genuine two-sided extensions of the faulted solved
fields, not fields solved without faults, and no point is laterally displaced.
Node Hermite gradients and surface isovalues come from the actual engine.
No discontinuous assembled field is interpolated across the separator.

The production segmentation is inspected and numerically verified, not assumed
to be a Heaviside function. Model7 reverses unit IDs during activation, so after
`_modify_faults_values_output` subtracts the minimum, its infinite-fault drift is
the **descending** sigmoid of the fault scalar minus its isovalue: bank 0 is
the positive-scalar side and bank 1 the negative-scalar side. The adapter checks
this orientation against production samples. At `14^3`, 101 near-fault probes
include 27 nonsaturated drift samples; shifted sigmoid parity error is
`1.9082e-16`. The same-bank sampler matches production on 1493 bank-0 and 4700
bank-1 saturated queries per dependent stack, with maximum scalar/gradient
errors `4.5103e-14 / 1.1991e-13`. The two bank fields differ by
`0.06521656063359901` and `0.05910903889915131`; the fault has not been disabled.

### Pre-Emission Incidence

`extract_bank_skin` supplies explicit directed separator/horizon contacts to a
private uniform extraction helper. The ordinary `extract_topology_aware` entry
point **still rejects faults by default**. The separator remains `FAULT`, never
an erosion stack. Eligible fault and erosion edges are recorded separately.
Horizon ownership is the retained bank half-space, intersected with the negative
youngest-stack field for older horizons. Shared sampled face-root chords,
branch-specific production Hermite QEFs, bounded joint QEFs and quad incidence
are established and validated before triangle emission. There is no independent
extract-and-snap, weld, clipping, marching extraction or post-repair.

Each bank has its own tagged vertex/surface identities and its own fault skin.
The two skins lie on the same infinite planar separator but have distinct
bank-specific seam tessellations. They are **bank-side fault skins, not one
conforming fault mesh**. Opposite-bank horizon IDs never merge. Same-stack
horizons cannot share a joint representative. Incidence validation only counts
a shared edge when both junction keys have the same participant pair; a triangle
is a shared patch only if all three vertices share two participants, not merely
because all three vertices happen to be junctions.

### Verified Results

The default uniform resolution is `14^3` cells (2744 cells, 3375 nodes).
Actual center samples also check for sign crossings absent from corner evidence;
this is not general interior-topology certification.

| Metric | Bank 0 | Bank 1 |
| --- | --- | --- |
| Triangles: fault skin / rock3 / rock2 / rock1 | 442 / 78 / 156 / 182 | 442 / 234 / 676 / 878 |
| Actual shared seam edges | 52 | 91 |
| Joint cells | 56 | 98 |
| Minimum total triangle angle | 16.06325 degrees | 1.70128 degrees |
| Minimum contact triangle angle | 16.06325 degrees | 17.44673 degrees |
| Maximum total aspect ratio | 3.76120 | 34.50345 |
| Degenerate / duplicate triangles | 0 / 0 | 0 / 0 |
| Bank-side / actual-engine winding violations | 0 / 0 | 0 / 0 |

All actual shared edges equal planned seams. Each seam has two controller
triangles and one target triangle; other surfaces have zero incidence.
Noncontact geometry and oriented connectivity are exactly equal to independent
same-bank production QEF reference results. The independent reference uses the
same retained primal-edge support, not the assembled discontinuous field.
Shared-face disagreement is zero. Original production triangle counts are
`126 / 112 / 236 / 344`; tests compare **every production mesh vertex and face**
against an uncaptured baseline, including the fault mesh, with exact equality.

There is no triangle-quality improvement claim: bank 1 retains a 1.70128-degree
noncontact triangle also present in the independent QEF reference. Actual older
surface erosion residual is at most `2.3826e-9` scalar units, checked against an
explicit `1e-8` tolerance. Sampled seam chords are not exact continuous-field
intersection curves; center/corner approximation residual reaches `0.00864559`.

### Scope And Rejections

Support is deliberately bounded to one infinite planar fault affecting both
Model7 stacks, uniform conforming cells, and the existing sampled branch/contact
contract. Finite/nonplanar/multiple faults and adaptive extraction are rejected.
There are open crop boundaries, no caps, no closed-volume/global watertightness
claim, and no merging of the two fault skins into a single geological interface.
Multiple competing actual contacts in one cell still require richer incidence
than this one-pair-per-cell planner and are rejected, not overwritten.

Resolution `16^3` remains honestly unsupported: bank 0 rejects
`unsupported_shared_patch: quad split would emit a fully shared triangle`;
bank 1 rejects `sample_aligned_interface: zero node needs a sided extraction rule`.
The rejected viewer shows actual **production faulted Model7** on the left and
geometric bank indicators on the right, not a successful fallback. Increasing
resolution is not asserted to monotonically resolve these topology contracts.

### Commands And Artifacts

Run from the engine repository with the same CPU environment as above:

```bash
export CUDA_VISIBLE_DEVICES='' DEFAULT_BACKEND=numpy DEFAULT_PYKEOPS=False
export DEFAULT_TENSOR_DTYPE=float64 DUAL_CONTOURING_VERTEX_OVERLAP=none PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

/home/leguark/.venv/2025/bin/python examples/topology_aware_model7.py --check-only --resolution 14
/home/leguark/.venv/2025/bin/python examples/topology_aware_model7.py --off-screen --resolution 14 \
  --screenshot /tmp/opencode/topology-model7.png
/home/leguark/.venv/2025/bin/python examples/topology_aware_model7.py --off-screen --resolution 16 \
  --screenshot /tmp/opencode/topology-model7-rejected16.png

/home/leguark/.venv/2025/bin/pytest -q \
  tests/test_common/test_modules/test_topology_model7.py \
  tests/test_common/test_modules/test_topology_aware_extraction.py \
  tests/test_common/test_modules/test_topology_real_models.py \
  tests/test_common/test_modules/test_topology_failure_examples.py \
  tests/test_common/test_modules/test_topology_finite_fault_characterization.py \
  tests/test_common/test_modules/test_contact_topology.py
```

The screenshot parent is verified before output. Both screenshots have matching
JSON reports. Successful panels use linked cameras, consistent surface colors,
purple fault skins, white actual seams, and solid bank-0 versus edged bank-1
indicators. Standalone/PyCharm runs explicitly select NumPy CPU float64 without
KeOps and print the active backend. No interactive window was opened.

Verification: 101 tests passed, including all 65 prior topology experiment tests,
10 new focused Model7 tests and 26 contact-topology tests. The supplied fixture
agent's file was not edited. Additional agent delegation was unavailable in this
session because no subagent tool was exposed. No installations or commits were
made.
