# Model7 Extraction Benchmarks

## Scope

Two standalone workers measure different Model7 workloads:

| Worker | Supported workload |
|---|---|
| `tests/benchmark/model7_production_benchmark.py` | Production adaptive-octree modeling and extraction, overlap `none` or `pretty`, NumPy CPU or Torch CPU/CUDA, `float64` or `float32`. |
| `tests/benchmark/model7_joint_benchmark.py` | Experimental joint extraction of two bank-side skins on a bounded uniform grid, NumPy CPU `float64` only. |

The production worker is the relevant benchmark for fine octrees and production
Torch. The joint experiment is not an adaptive-octree or Torch implementation,
and its timings are not a like-for-like replacement for production extraction.
Neither worker benchmarks `watertight` overlap.

Run configurations sequentially: concurrent jobs would contend for CPU and memory.

## Reproduction

Run commands from `/home/leguark/Projects/gempy_project/gempy_engine`, using the
project virtualenv directly. Each production configuration must run in a fresh
process so backend and overlap settings are applied before numerical imports.
Both workers emit one JSON report on stdout and progress on stderr; `--output`
also writes the JSON report. Its parent directory must already exist.

Use two discarded warmups and seven measured repetitions. The following template
runs the production CPU sweep sequentially. Bound runtime and memory when
extending the depth range.

```bash
for backend in numpy torch; do
  for mode in none pretty; do
    for depth in 2 3 4 5; do
      /home/leguark/.venv/2025/bin/python tests/benchmark/model7_production_benchmark.py \
        --backend "$backend" --device cpu --dtype float64 \
        --mode "$mode" --depth "$depth" --scope both \
        --warmups 2 --repeats 7 \
        --output "/tmp/opencode/model7-production-${backend}-cpu-float64-${mode}-depth${depth}-both.json"
    done
  done
done
```

`--scope both` measures full model calls first, then extraction-only calls, and
checks extraction output against the full-model baseline. For scope-isolated
process memory records, use separate invocations with `--scope model` and
`--scope extraction`, and distinct output names. An extraction-only invocation
still performs an untimed interpolation setup; it has no full-model mesh baseline
for the cross-scope parity check.

Torch CUDA is supported by the production worker, but the CUDA driver is
currently unavailable on this machine. CUDA results remain unavailable; this is
not evidence about GPU performance. After driver availability is restored, an
explicit reproduction command is:

```bash
/home/leguark/.venv/2025/bin/python tests/benchmark/model7_production_benchmark.py \
  --backend torch --device cuda --dtype float64 \
  --mode pretty --depth 5 --scope both --warmups 2 --repeats 7 \
  --output /tmp/opencode/model7-production-torch-cuda-float64-pretty-depth5-both.json
```

Repeat with `--mode none` and matching depths for the CUDA
matrix. CUDA requests fail explicitly when unavailable, with no CPU fallback.
NumPy rejects `--device cuda`. Both production backends support `--dtype float32`
as a separate configuration; do not mix dtype results in one comparison.
PyKeOps is disabled in all these workloads.

The bounded joint case uses resolution 14:

```bash
/home/leguark/.venv/2025/bin/python tests/benchmark/model7_joint_benchmark.py \
  --resolution 14 --scope all --warmups 2 --repeats 7 \
  --output /tmp/opencode/model7-joint-numpy-cpu-float64-resolution14-all.json
```

Joint `--scope splits` runs only the split pipeline; `--scope end-to-end` runs
only the fresh-input combined pipeline. `--scope all` runs these as independent
pipelines, not as a sum of stage medians. The worker accepts resolutions 2 through
16, but some resolutions, including 16, may reject topology. Increasing its
resolution is not a substitute for a fine production octree.

## Production Contract

`--depth` sets both `number_octree_levels` and
`number_octree_levels_surface`, and must be at least 2. The factory root remains
fixed at `2 x 2 x 2` (eight cells). Depth changes refinement levels, not root
resolution. The workload is adaptive, so depth alone does not describe its size.

The report records `cells_per_level`, `refined_parents_per_transition`,
`actual_leaf_cells`, `final_level_cells`, and `effective_levels`. Every refined
parent has eight children, and leaf accounting includes unrefined cells at every
coarser level:

```text
actual_leaf_cells = sum(cells_per_level) - sum(refined_parents_per_transition)
```

Do not label the final-level cell count as the total leaf count, or treat the
adaptive workload as a fully populated uniform grid. Keep these counts alongside
mesh vertices, triangles, scalar dtypes, and requested/effective settings when
reporting a depth sweep.

| Scope | Timed work | Untimed preparation |
|---|---|---|
| `model` | Entire public `compute_model` lifecycle, including internal copies, interpolation, production extraction, solution construction, and cache/GPU cleanup. | Fresh fixture construction before every call. |
| `extraction` | Production `dual_contouring_multi_scalar`, cache initialization, and matching cache/GPU cleanup. | Interpolation setup and a fresh deepcopy of its descriptor, input, options, and solved octrees before every call. |

For extraction setup, mesh extraction remains enabled in the options so corner
fields are evaluated as in a full compute. Only the extraction entry point is
suppressed during that untimed setup. No production function is replaced in a
timed call.

These are warmed-process measurements, not warm-weight-cache measurements.
Production cleanup clears weights between calls. Extraction reuses solved
octrees but starts with cold weights and may solve weights for edge-gradient
evaluation. Its timing is therefore not guaranteed to be pure meshing of
precomputed gradients.

Fixture preparation, explicit extraction-input copying, garbage collection,
external validation, workload accounting, and JSON serialization are outside the
timers. Internal production checks remain part of the timed API. CUDA is
synchronized before and after calls.

Repeated outputs are checked outside timing: ordered triangle indices must
match exactly, vertices use `allclose` with `rtol=atol=1e-5`, and mesh/workload
counts must remain unchanged. This is within-configuration repeat parity, not a
claim of cross-backend, cross-dtype, or cross-mode equivalence.

## Joint Contract

Resolution 14 is a uniform `14 x 14 x 14` cell case. The worker samples corners
and cell centers, with additional solve-only probes. It extracts two disjoint
bank-side fault skins, not a single conforming fault mesh or closed geological
volumes. Adaptive grids, Torch, finite faults, and multiple faults are outside
this experiment's supported scope.

The split timers follow data dependencies within a freshly prepared pipeline:

| Timer | Included work |
|---|---|
| `solve_capture` | Unmodified three-level solve, custom-grid evaluation, gradients, capture copies, and API guards; no production mesh. |
| `bank_corner_gradient_sampling` | One independent and four forced-bank queries on corners and centers, plus array preparation; excludes solve-only probes. |
| `both_bank_extraction` | Both `extract_bank_skin(include_reference=False)` calls on prepared fields, including adapter checks and tags. |
| `end_to_end_fresh_inputs` | Independently measured solve/capture, sampling, and both-bank extraction on a fresh case each round. |

Fixture CSV loading, input creation, and grid preparation are excluded from all
joint scope timers, including end-to-end. End-to-end is a combined compute
pipeline, not process startup or fixture-to-JSON latency.

Intrinsic topology, ownership, and incidence guards and diagnostic metadata
remain timed: the extractor API does not provide a switch to disable them.
External parity and mesh-quality validation runs only on the last measured
result per scope, after all timers stop. Imports, warmups, plotting, reference
generation, and JSON serialization are excluded. Center checks are not an
interior-topology certificate.

## Threads And Memory

Both workers pin numerical-library threads to one before import. Production
also sets Torch intra-op and inter-op threads to one, disables the configured
dual-contouring threading flags, and records its environment. However, production
`pretty` QEF still uses an unconditional `ThreadPoolExecutor` with default
workers. Thus numerical single-threading does not mean that `pretty` extraction
has no Python worker threads. The joint worker additionally applies
`threadpoolctl` limits and verifies the reported numerical pools.

Production RSS uses Linux `/proc/self/status`: `VmRSS` is current process RSS and
`VmHWM` is the process-lifetime high-water mark since exec. Import RSS is reported
separately, but total peaks still include imports, fixture preparation, copies,
validation, and earlier scopes. These are not isolated stage-memory peaks, and
subtracting import RSS does not turn them into stage allocations.

CUDA allocator peaks reset per timed call but include live prepared tensors and
the extraction baseline; non-PyTorch GPU allocations are excluded. The joint
worker does not report RSS or GPU memory. Preserve these distinctions rather
than deriving stage-memory comparisons from incompatible records.

Both reports retain raw timings and summarize median and p95. The p95 uses NumPy
linear percentile interpolation on seven measured samples, not a robust tail
estimate. A one-repeat run is only a pilot, not a performance estimate.

## Results

The completed sequential CPU sweep ran on an AMD RYZEN AI MAX+ 395 with 32
logical cores, using Python 3.13.9, NumPy 2.4.4, and Torch 2.11.0+cu130. Numerical
threads were limited to one, subject to the production `pretty` thread-pool
caveat above. These are CPU results, despite the CUDA-enabled Torch build; the
unavailable CUDA driver prevented GPU measurements, with no CPU fallback.

All 19 configurations are marked `passed` in
`/tmp/opencode/model7_benchmark_sweep_status.json`, and the sequential log
`/tmp/opencode/model7-benchmark-sweep.log` ends with `DONE`. Every report contains
two discarded warmups and seven measured repetitions: these are completed
measurements, not one-repeat pilots. Times below are seconds, rounded to four
decimal places; ratios use the unrounded medians.

### Production Float64

All 16 configurations used CPU `float64`, `--scope both`, and matching overall
and surface depths. Extraction uses the cold-weight-cache contract described
above; full model includes the complete public API lifecycle.

| Depth | Backend | Mode | Extraction Median | Extraction p95 | Full Model Median | Full Model p95 |
|---|---|---|---:|---:|---:|---:|
| 2 | NumPy | none | 0.5245 | 0.5286 | 1.4377 | 1.4735 |
| 2 | NumPy | pretty | 0.5334 | 0.5355 | 1.4519 | 1.4550 |
| 2 | Torch | none | 0.5388 | 0.5453 | 1.4818 | 1.4886 |
| 2 | Torch | pretty | 0.5464 | 0.5508 | 1.4743 | 1.4803 |
| 3 | NumPy | none | 0.7137 | 0.7158 | 2.2789 | 2.2943 |
| 3 | NumPy | pretty | 0.7119 | 0.7163 | 2.2727 | 2.2965 |
| 3 | Torch | none | 0.7505 | 0.7578 | 2.3367 | 2.3515 |
| 3 | Torch | pretty | 0.7474 | 0.7508 | 2.3600 | 2.3742 |
| 4 | NumPy | none | 1.4973 | 1.5144 | 4.4813 | 4.5146 |
| 4 | NumPy | pretty | 1.5135 | 1.5162 | 4.4904 | 4.5317 |
| 4 | Torch | none | 1.6638 | 1.6854 | 4.8635 | 4.8897 |
| 4 | Torch | pretty | 1.6683 | 1.6793 | 4.9144 | 4.9391 |
| 5 | NumPy | none | 5.0749 | 5.0984 | 14.7993 | 14.9677 |
| 5 | NumPy | pretty | 5.0850 | 5.1264 | 14.6286 | 14.7189 |
| 5 | Torch | none | 5.7919 | 5.8940 | 16.6339 | 16.7402 |
| 5 | Torch | pretty | 5.8347 | 5.8976 | 16.7631 | 16.7834 |

Counts are identical between the two float64 backends at each depth and mode,
and between model and extraction scopes. They are listed once per depth, not
repeated in every timing row. Triangle totals differ by overlap mode.

| Depth | Actual Leaves, All Levels | Final-Level Cells | Triangles, none | Triangles, pretty |
|---|---:|---:|---:|---:|
| 2 | 64 | 64 | 134 | 98 |
| 3 | 512 | 512 | 818 | 774 |
| 4 | 2,696 | 2,496 | 3,628 | 3,322 |
| 5 | 14,456 | 13,440 | 16,344 | 14,670 |

The extraction median ratio `pretty / none` spans 0.9975 to 1.0170 for NumPy
and 0.9959 to 1.0142 for Torch across depths 2 through 5. Thus this sweep shows
nearly equal elapsed extraction times for the two modes, not equal output meshes
or a statistically established performance equivalence.

For the same depth and mode, extraction median ratios `Torch / NumPy` are:

| Depth | none | pretty |
|---|---:|---:|
| 2 | 1.0272 | 1.0244 |
| 3 | 1.0515 | 1.0498 |
| 4 | 1.1113 | 1.1023 |
| 5 | 1.1413 | 1.1474 |

Torch CPU extraction took 2.4% to 14.7% longer than NumPy in these float64
configurations. This is a measured CPU/backend comparison, not a CUDA prediction.

### Torch Float32

The two additional CPU configurations used depth 4 and `--scope both`. Float64
rows are repeated here only to make the dtype comparison explicit.

| Dtype | Mode | Extraction Median | Extraction p95 | Full Model Median | Full Model p95 |
|---|---|---:|---:|---:|---:|
| float64 | none | 1.6638 | 1.6854 | 4.8635 | 4.8897 |
| float32 | none | 0.9154 | 0.9263 | 3.1205 | 3.2000 |
| float64 | pretty | 1.6683 | 1.6793 | 4.9144 | 4.9391 |
| float32 | pretty | 0.9126 | 0.9205 | 3.2039 | 3.2333 |

| Dtype | Actual Leaves | Final-Level Cells | Vertices | Triangles, none | Triangles, pretty |
|---|---:|---:|---:|---:|---:|
| float64 | 2,696 | 2,496 | 2,038 | 3,628 | 3,322 |
| float32 | 2,731 | 2,536 | 2,047 | 3,650 | 3,344 |

Float32 extraction medians are 0.5502 (`none`) and 0.5470 (`pretty`) times their
float64 counterparts; full-model medians are 0.6416 and 0.6519 times,
respectively. Cross-dtype geometry parity is **unverified**, and refinement and
triangle counts changed. These timings therefore do not establish an equivalent
mesh at lower cost. Each dtype passed its own repeat and model/extraction parity
checks.

### Joint Resolution 14

The NumPy CPU float64 joint case used 2,744 uniform cells, 3,375 corner nodes,
2,744 center queries, and 101 solve-only extra probes. Fixture CSV/input/grid
preparation was excluded from every timer, including end-to-end.

| Stage | Median | p95 |
|---|---:|---:|
| Solve/capture | 2.2955 | 2.3288 |
| Bank corner/center gradient sampling | 2.2023 | 2.2125 |
| Both-bank extraction on prepared fields | 1.0429 | 1.0559 |
| Independent fresh-input end-to-end pipeline | 5.5449 | 5.6069 |

| Bank | Vertices | Triangles Per Surface | Total Triangles | Seam Edges |
|---|---:|---|---:|---:|
| 0 | 1,188 | 442, 78, 156, 182 | 858 | 52 |
| 1 | 1,586 | 442, 234, 676, 878 | 2,230 | 91 |

Both scopes produced these two bank-side skins, totaling 3,088 triangles. They
are not the same outputs as the production meshes. For context only, depth-4
production float64 extraction medians divided by joint prepared-field extraction
are 1.44/1.45 for NumPy `none`/`pretty` and 1.60/1.60 for Torch. These ratios are
**not pure extractor speedups**: production extraction starts with cold weights
and can solve edge-gradient weights, whereas the joint timer receives prepared
gradients after a separately timed 2.2023-second sampling stage. The adaptive
versus uniform grids, different outputs, and intrinsic joint diagnostics further
prevent a like-for-like speedup claim.

### Validation And Artifacts

No validation failures were reported. All 18 production reports passed
within-configuration repeat parity and extraction-versus-full-model parity.
Joint validation passed for the last measured result of both scopes, preserving
observations and showing zero scalar/gradient error at saturated production-bank
nodes. Both banks had zero reported degenerate or duplicate triangles,
nonmanifold edges, bank-side violations, and winding violations; shared-edge and
seam-incidence checks passed. This does not add cross-backend/cross-dtype geometry
parity, reference parity, or an interior-topology certificate.

Production peak RSS records remain process-lifetime high-water marks including
imports, preparation, copies, validation, and the earlier model scope. They are
not extraction-stage memory peaks. The joint report has no RSS measurement.

The 19 raw JSON artifacts are in `/tmp/opencode/`, with exact naming patterns:

- `production_{numpy,torch}_{none,pretty}_f64_d{2,3,4,5}.json` (16 files).
- `production_torch_{none,pretty}_f32_d4.json` (2 files).
- `joint_r14.json` (1 file).

Braces above denote filename alternatives. These coordinator artifact names
differ from the illustrative `--output` names in Reproduction. Raw samples,
unrounded summaries, settings, counts, memory records, and validation details
remain in the JSON files; status/log elapsed times are whole-worker durations,
not the stage timings in these tables.
