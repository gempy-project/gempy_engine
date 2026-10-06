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
    parser.add_argument("--scope", choices=["overlap", "extraction", "model"], default="overlap")
    parser.add_argument("--case", choices=["single_stack", "unconformity"], default="unconformity")
    parser.add_argument("--resolution", type=int, choices=[4, 8], default=4)
    parser.add_argument("--mode", choices=["none", "pretty", "watertight"], default="pretty")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
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
        from tests.benchmark.test_benchmark_contacts import (
            BackendTensor, AvailableBackends, DualContouringOverlap, dc,
            prepare, add_mesh_counts,
        )

        BackendTensor._change_backend(AvailableBackends.numpy, use_gpu=False,
                                      use_pykeops=False, dtype="float64")
        dc.DUAL_CONTOURING_VERTEX_OVERLAP = DualContouringOverlap[args.mode]
        startup_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        target, setup, info = prepare(args.scope, args.case, args.resolution, args.mode)
        call_args, call_kwargs = setup()
        prepared_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        pre_call_rss = rss()
        start = time.perf_counter()
        result = target(*call_args, **call_kwargs)
        elapsed = time.perf_counter() - start
        peak_total = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        meshes = result.dc_meshes if args.scope == "model" else result
        add_mesh_counts(info, meshes, "output")
        if args.scope == "model":
            info["leaf_cells"] = len(result.octrees_output[-1].grid.octree_grid.values)
    info.update(elapsed_seconds=elapsed, process_start_rss_bytes=process_start_rss,
                startup_peak_rss_bytes=startup_peak, prepared_peak_rss_bytes=prepared_peak,
                pre_call_rss_bytes=pre_call_rss, peak_total_rss_bytes=peak_total,
                memory_note="Lifetime process peak including startup/preparation/copies; not stage peak.")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
