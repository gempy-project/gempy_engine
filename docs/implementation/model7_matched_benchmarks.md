# Matched Model 7 Benchmarks

## Results

The production `none`, `pretty`, and adaptive `joint` extractors were measured
using the verified Model 7 configuration: root resolution 9, depth 2, minimum
level 0, 102 surface points, and 8 orientations. The solved hierarchy contains
293 coarse leaves and 3488 fine leaves, covering an 18 x 18 x 18 finest lattice.

Measurements used NumPy CPU float64, no KeOps or GPU, and one numerical-library
thread on an AMD RYZEN AI MAX+ 395. Production `pretty` still uses its default
Python QEF thread pool. Configurations ran sequentially, with two discarded
warmups and seven measured repetitions per mode and scope.

Times are seconds. Ratios use unrounded medians. Results below are from
2026-10-08, after the joint optimizations described in the next section; the
same host, settings, and command were used for the original baseline.

| Scope | Mode | Median | p95 | Baseline median |
|---|---|---:|---:|---:|
| Extraction | none | 1.469 | 1.483 | 1.917 |
| Extraction | pretty | 1.478 | 1.502 | 1.923 |
| Extraction | joint | 3.460 | 3.470 | 19.638 |
| Full computation | none | 3.609 | 3.721 | 4.945 |
| Full computation | pretty | 3.625 | 3.740 | 5.202 |
| Full computation | joint | 5.617 | 5.760 | 23.098 |

| Median Ratio | Extraction | Full Computation | Baseline Extraction | Baseline Full |
|---|---:|---:|---:|---:|
| joint / none | 2.356 | 1.556 | 10.246 | 4.671 |
| joint / pretty | 2.341 | 1.549 | 10.213 | 4.440 |
| pretty / none | 1.006 | 1.005 | 1.003 | 1.052 |

Extraction sample ranges were 1.460-1.488 for `none`, 1.466-1.507 for `pretty`,
and 3.441-3.471 for `joint`; full computation 3.579-3.726, 3.593-3.752, and
5.569-5.767. The `none`/`pretty` improvement comes only from the evaluator garbage
collection change below. The p95 is a linearly interpolated percentile of seven
samples, not a robust tail estimate. The run took 187 seconds (baseline 557).

## Four Modes, Including `joint_contacts`

The same command (same host and settings) with `joint_contacts` added, after
its targeted field queries (see below). All four modes recorded three cache hits
and zero solves.

| Scope | Mode | Median | p95 |
|---|---|---:|---:|
| Extraction | none | 1.489 | 1.495 |
| Extraction | pretty | 1.479 | 1.496 |
| Extraction | joint | 3.514 | 3.537 |
| Extraction | joint_contacts | 1.493 | 1.502 |
| Full computation | none | 3.652 | 3.678 |
| Full computation | pretty | 3.664 | 3.700 |
| Full computation | joint | 5.756 | 5.866 |
| Full computation | joint_contacts | 3.685 | 4.061 |

`joint_contacts` costs 1.01x `pretty` for extraction and full computation, and
0.43x `joint` for extraction. It returns 2494 vertices, 4400 triangles and 85
shared contact seams: one fault mesh and no bank skins, so the output is not
equivalent to the other modes.

The first `joint_contacts` version took 2.55 s to extract. It queried every leaf
corner and all stacks at every point, with gradients everywhere, through the
generic `sample_fields` contract. The targeted `field_query` contract removes
the corner re-evaluation, evaluates gradients only for the crossing surface
(and only at the endpoints of edges it crosses), and checks vertices with
scalars only. Geometry is unchanged: identical faces and vertex keys, and
vertices within 2e-13. Raw results are kept at
`/home/leguark/Projects/gempy_project/benchmark_results/model7-matched-results-2026-10-08-4modes-reduced.json`.

## Backends: NumPy, Torch CPU, CUDA and KeOps

`tests/benchmark/model7_backend_benchmark.py` runs each backend in a fresh
process: full `compute_model` runs on the same Model 7 configuration, two
warm-ups and seven repeats, with CUDA-synchronised timers around the extraction
dispatcher. It is an end-to-end comparison, not the matched solved-octree
protocol above, and NumPy threads are not pinned. GPU: RTX 2070 SUPER (Turing;
its float64 rate is low).

| Backend | pretty extraction | joint_contacts extraction | pretty total | joint_contacts total |
|---|---:|---:|---:|---:|
| NumPy | 1.763 | 1.546 | 4.161 | 3.943 |
| Torch CPU | 0.754 | 0.784 | 2.536 | 1.780 |
| Torch CUDA | 0.327 | 0.575 | 1.130 | 1.312 |
| Torch CUDA + KeOps | 0.105 | 0.662 | 0.168 | 0.725 |

Banked `joint` rejects every Torch backend by design (3.813 s extraction on
NumPy). On CUDA, field queries stop dominating `joint_contacts`: a KeOps profile
puts them at about a quarter of extraction. Most of the rest is host-side
topology, chiefly the pure-Python leaf cell complex construction, then triangle
planning and key bookkeeping, which no device accelerates. KeOps meshes differ
slightly in every mode (see the backend note in `adaptive_joint_extraction.md`).
Raw results are kept at
`/home/leguark/Projects/gempy_project/benchmark_results/model7-backends-2026-10-08.json`.

## Joint Optimization Profile

A `cProfile` of one joint extraction (same configuration, single run) attributed
the original 20.95 seconds as follows:

| Stage | Seconds | Cause |
|---|---:|---|
| Outer bank corner sampling | 8.4 | 30,248 corner rows per bank for 4,827 unique points |
| Core seed query | 2.0 | Re-evaluated the same 4,827 lattice points per bank |
| Evaluator `gc.collect` | 5.1 | 33 full collections, ~0.15 s each over ~300k import-time objects |
| Branch classification | 2.1 | Per-leaf Python `allclose`/`diff` checks for every surface |
| Junction planning | 1.6 | `face_pair_root` called on every tile for every pair |
| Cell complex | 0.6 | Rebuilt for each bank |

Changes, all preserving rejection behavior, guards, and the callback contract:

- `_BankFieldMemo` in `adaptive_fault_extraction.py` evaluates each exact point
  once per stack: per bank for affected stacks, once for fault/unaffected stacks.
  The core's seed query is then fully served from it.
- The fault sampler `query` accepts a `stacks` subset; the unaffected partition
  no longer evaluates affected stacks with an arbitrary bank.
- `generic_evaluator` collects garbage only when evaluation was actually chunked.
  Single-chunk temporaries are already released by reference counting. This is
  a shared production path; multi-chunk behavior is unchanged.
- `classify_cell_branches` computes per-cell predicates vectorized and visits only
  cells that can raise or have crossings, in the original order.
- `plan_adaptive_junctions` skips tiles where a participant has no sign change
  (exactly `face_pair_root`'s first early return) and vectorizes leaf decisions.
- The core point cache stores column indices into concatenated blocks; tile and
  minimal-edge nodes are built vectorized; an optional prebuilt `cell_complex`
  lets bank partitions share one complex.

After the changes, a profile attributes about 2.4 seconds to genuine kernel
evaluation (two multi-chunk stack-2 corner seeds are 1.2 seconds of it) and about
1.6 seconds to topology construction. Triangle topology, vertex identities, seam
edges, bank labels, and the root-7/minimum-level-0 rejection are unchanged for
root 9/minimum level 0 and root 7/minimum level 2. Vertices differ from the
baseline by at most 2.5e-14 because points are now evaluated in different
batches; every change after the memo is bitwise identical.

Remaining options, not implemented: scalar-only validation queries and per-surface
crossing-gradient queries would save under 0.4 seconds here but require extending
the reviewed `sample_fields` contract.

## Timing Contract

Extraction is measured inside the public `compute_model` extraction hook, while
the verified production cache remains live. Interpolation runs once. Each
extractor receives a fresh copy of the same descriptor, input, options, and exact
solved octree. The same cache snapshot is restored before every call.

The timer surrounds the real production extraction dispatcher. It includes all
mode-specific field and gradient queries, bank sampling, topology construction,
and mandatory provenance/geometry guards. Copying, cache restoration, validation,
optional reporting, plotting, and final model cleanup are outside this timer.

Full computation instead times unpatched `compute_model` on fresh fixtures with
identical settings. Internal copying, interpolation, extraction, solution
construction, and normal cleanup are included. Fixture construction, imports,
external validation, and reporting are excluded.

Mode orders cycle through six permutations. Seven measured repetitions are not
perfectly position-balanced; temporal effects are visible in full-model samples.
No timings are instrumented with solver/cache profilers.

## Validation And Output

All 54 calls, including warmups, matched the same solved-array fingerprint.
Canonical leaf identities and fields matched across modes and scopes. Timed
extraction preserved the original baseline and complete cache signatures,
including entry keys, fingerprints, and weight payloads.

Separate untimed extraction audits recorded three cache hits and zero solves for
each mode. The audit checks real solver calls and cache routes; these counters
are not installed in measured calls. Retained actual production weights were
checked against the live cache before snapshot copying. Final cache cleanup
completed.

Within each mode, oriented triangles, mesh digests, bank labels, and shared IDs
were stable across repetitions and scopes.

| Mode | Vertices | Triangles |
|---|---:|---:|
| none | 2670 | 4848 |
| pretty | 2670 | 4454 |
| joint | 2988 | 5302 |

Joint returned 187 seam edges: 68 in bank 0 and 119 in bank 1. This natural
mixed-leaf configuration has zero coarse/fine contact seam edges. It reported no
opposite-bank ID joins, uniform fallback, or post-reconciliation.

These are extraction costs on the same solved octree, not equivalent-output
algorithm speedups. Ordinary modes contour the final-level cells using smooth
fault fields. Joint uses all mixed-level leaves and bank-conditioned fields,
with two bank-side fault skins rather than one conforming fault mesh. Additional
geometry and guarantees are part of its measured cost. These CPU results do not
establish Torch or GPU performance. The profile above was a separate
instrumented run; benchmark timings are uninstrumented.

## Reproduction

From the engine repository:

```bash
/home/leguark/.venv/2025/bin/python tests/benchmark/model7_matched_benchmark.py \
    --scope both --root-resolution 9 --depth 2 --min-level 0 \
    --warmups 2 --repeats 7 --output /path/to/model7-matched-results.json
```

The output parent must already exist. Each invocation must start in a fresh
process. The worker pins the backend and numerical thread settings before
imports. `--scope extraction` or `--scope model` measures only that scope.
Unsupported grids reject rather than falling back to another mode.

Raw samples, per-call audits, settings, host metadata, hashes, and process-lifetime
RSS are written to the `--output` JSON; the 2026-10-08 run is kept outside the
repository at `/home/leguark/Projects/gempy_project/benchmark_results/model7-matched-results-2026-10-08.json`. The process peak includes
imports, preparation, copies, validation, and both scopes; it is not a stage-only
memory measurement. The original baseline run completed in approximately 557 seconds.
