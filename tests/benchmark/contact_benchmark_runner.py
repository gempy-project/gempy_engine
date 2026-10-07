"""Isolated Linux peak-process-RSS baseline (native arrays included, GPU excluded).

Example from any directory, using the virtualenv interpreter directly:
  /home/leguark/.venv/2025/bin/python /path/to/contact_benchmark_runner.py \
      --scope overlap --case unconformity --resolution 4 --mode pretty

Each invocation launches a fresh worker with pinned environment and BLAS threads.
Peak RSS is the process lifetime high-water mark, INCLUDING imports, preparation,
and the fresh input copy. It is not a stage-only peak or a ru_maxrss difference.
The current RSS just before timing and startup/preparation high-water marks help
interpret small workloads whose peak is dominated by startup. One cold call is
timed; use pytest-benchmark for repeatable timing distributions.

Prepared cells: --scope contact-cells --case dense --mode contact_aware
Use --size and --dtype for cell datasets. This calls the production module, not
the legacy overlap helper.
Prepared finalization: --scope finalize-cells --case compound_pruned --mode contact_aware
"""

import argparse
import contextlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def rss():
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS unavailable; this runner requires Linux")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["overlap", "extraction", "model", "contact-cells", "finalize-cells"], default="overlap")
    parser.add_argument("--case", choices=["single_stack", "unconformity", "sparse", "dense",
                                         "no_overlap", "multiple_horizons", "fault",
                                         "compound_supported", "compound_pruned"], default="unconformity")
    parser.add_argument("--resolution", type=int, choices=[4, 8], default=4)
    parser.add_argument("--mode", choices=["none", "pretty", "watertight", "contact_aware"], default="pretty")
    parser.add_argument("--size", type=int, choices=[32, 256, 8192], default=256)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.scope == "finalize-cells":
        if args.case not in ("compound_supported", "compound_pruned") or args.mode != "contact_aware":
            parser.error("finalize-cells requires a compound dataset and --mode contact_aware")
    elif args.scope == "contact-cells":
        if args.case not in ("sparse", "dense", "no_overlap", "multiple_horizons", "fault") or args.mode != "contact_aware":
            parser.error("contact-cells requires a cell dataset and --mode contact_aware")
    else:
        if args.case not in ("single_stack", "unconformity"):
            parser.error("Extraction/model/overlap scopes require a model dataset")
        if args.dtype != "float64":
            parser.error("--dtype float32 requires contact-cells or finalize-cells")
        if args.scope == "overlap" and args.mode == "contact_aware":
            parser.error("contact_aware does not use legacy overlap; select --scope contact-cells")
    if not args.worker:
        env = os.environ.copy()
        env.update(DEFAULT_BACKEND="numpy", DEFAULT_PYKEOPS="False", DEFAULT_TENSOR_DTYPE="float64",
                   USE_GPU="False", DEBUG_MODE="False", GEMPY_TEST_PLOTS="False", MPLBACKEND="Agg",
                   NOT_MAKE_INPUT_DEEP_COPY="False", GEMPY_FLAT_STACKS="False",
                   OPTIMIZE_MEMORY="True", SET_RAW_ARRAYS_IN_SOLUTION="True",
                   ONLY_LITH_SOLUTION="False", LINE_PROFILER_ENABLED="False",
                   DUAL_CONTOURING_MULTITHREAD="False", GEMPY_SKIP_TRIANGULATION="0",
                   DUAL_CONTOURING_FAULT_OVERLAP_THREADING="False",
                   DUAL_CONTOURING_VERTEX_OVERLAP=args.mode,
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
                       env=env, check=True)
        return

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    process_start_rss = rss()
    with contextlib.redirect_stdout(sys.stderr):
        if args.scope == "finalize-cells":
            from tests.benchmark.test_benchmark_finalize_cells import prepare_finalize_cells, add_finalize_counts
            from gempy_engine.modules.dual_contouring.contact_cells import finalize_cell_vertices
        elif args.scope == "contact-cells":
            from tests.benchmark.test_benchmark_contact_cells import (
                prepare_contact_cells, add_contact_cell_counts,
            )
            # Import production before the startup measurement, not inside timing.
            from gempy_engine.modules.dual_contouring.contact_cells import reconcile_cell_vertices
        else:
            from tests.benchmark.test_benchmark_contacts import (
                BackendTensor, AvailableBackends, DualContouringOverlap, dc,
                prepare, add_mesh_counts,
            )
            BackendTensor._change_backend(AvailableBackends.numpy, use_gpu=False,
                                          use_pykeops=False, dtype="float64")
            dc.DUAL_CONTOURING_VERTEX_OVERLAP = DualContouringOverlap[args.mode]
        startup_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        startup_rss = rss()
        if args.scope == "finalize-cells":
            target, setup, info = prepare_finalize_cells(args.case, args.size, args.dtype)
        elif args.scope == "contact-cells":
            target, setup, info = prepare_contact_cells(args.case, args.size, args.dtype)
        else:
            target, setup, info = prepare(args.scope, args.case, args.resolution, args.mode)
        call_args, call_kwargs = setup()
        prepared_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        pre_call_rss = rss()
        start = time.perf_counter()
        result = target(*call_args, **call_kwargs)
        elapsed = time.perf_counter() - start
        peak_total = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        post_call_rss = rss()
        if args.scope == "finalize-cells":
            add_finalize_counts(info, result)
        elif args.scope == "contact-cells":
            add_contact_cell_counts(info, result, call_args)
        else:
            meshes = result.dc_meshes if args.scope == "model" else result
            add_mesh_counts(info, meshes, "output")
            if args.scope == "model":
                info["leaf_cells"] = len(result.octrees_output[-1].grid.octree_grid.values)
    info.update(elapsed_seconds=elapsed, process_start_rss_bytes=process_start_rss,
                startup_peak_rss_bytes=startup_peak, prepared_peak_rss_bytes=prepared_peak,
                 pre_call_rss_bytes=pre_call_rss, peak_total_rss_bytes=peak_total,
                 startup_rss_bytes=startup_rss, post_call_rss_bytes=post_call_rss,
                memory_note="Lifetime process peak including startup/preparation/copies; not stage peak.")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
