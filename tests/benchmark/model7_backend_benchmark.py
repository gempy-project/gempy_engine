"""Opt-in Model 7 extraction timings across backends (one fresh process per backend).

Example:
    /home/leguark/.venv/2025/bin/python tests/benchmark/model7_backend_benchmark.py \
        --warmups 2 --repeats 7 --output /tmp/model7-backends.json

Each run is a full public ``compute_model`` (interpolation + extraction). The
extraction timer wraps the production dispatcher with CUDA synchronisation.
Unlike model7_matched_benchmark.py, extraction is NOT measured on one shared
solved octree, and NumPy threads are not pinned: these are end-to-end backend
comparisons. Modes that reject a backend are recorded as rejected, not retried.
Production KeOps evaluation differs slightly from dense kernels, so KeOps meshes
may differ from the other backends; triangle counts are recorded per run.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BACKENDS = ('numpy', 'torch_cpu', 'torch_gpu', 'keops_gpu')
MODES = ('none', 'pretty', 'joint_contacts')


def worker(backend, args):
    os.environ.update(DEFAULT_TENSOR_DTYPE='float64', DUAL_CONTOURING_VERTEX_OVERLAP='none',
                      DEFAULT_BACKEND='numpy' if backend == 'numpy' else 'PYTORCH',
                      DEFAULT_PYKEOPS='True' if backend == 'keops_gpu' else 'False')
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root)]
    import numpy as np
    import gempy_engine.API.model.model_api as model_api
    from gempy_engine.config import AvailableBackends
    from gempy_engine.core.backend_tensor import BackendTensor as BT

    if backend == 'numpy':
        BT._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype='float64', grads=False)
    else:
        BT._change_backend(AvailableBackends.PYTORCH, use_gpu=backend.endswith('gpu'),
                           use_pykeops=backend == 'keops_gpu', dtype='float64', grads=False)
    try:
        import torch
        synchronize = torch.cuda.synchronize if BT.use_gpu else (lambda: None)
    except ImportError:
        synchronize = lambda: None

    dispatcher = model_api.dual_contouring_multi_scalar
    clock = {}

    def timed(*a, **k):
        synchronize()
        start = time.perf_counter()
        result = dispatcher(*a, **k)
        synchronize()
        clock['extraction'] = time.perf_counter() - start
        return result

    model_api.dual_contouring_multi_scalar = timed
    report = {}
    for mode in MODES:
        rows = dict(extraction=[], total=[], triangles=None)
        try:
            for round_index in range(args.warmups + args.repeats):
                model = make_model(args.root_resolution, args.depth, args.min_level, mode)
                synchronize()
                start = time.perf_counter()
                solution = model_api.compute_model(*model)
                synchronize()
                total = time.perf_counter() - start
                triangles = [int(len(m.edges)) for m in solution.dc_meshes]
                if rows['triangles'] is None:
                    rows['triangles'] = triangles
                elif triangles != rows['triangles']:
                    raise RuntimeError(f'{mode} triangle counts changed between runs')
                if round_index >= args.warmups:
                    rows['extraction'].append(clock['extraction'])
                    rows['total'].append(total)
        except (ValueError, NotImplementedError) as error:
            report[mode] = dict(status='rejected', error=str(error))
            continue
        report[mode] = dict(status='ok', triangles=rows['triangles'],
                            extraction_median=float(np.median(rows['extraction'])),
                            extraction_p95=float(np.percentile(rows['extraction'], 95)),
                            total_median=float(np.median(rows['total'])),
                            total_p95=float(np.percentile(rows['total'], 95)),
                            extraction_seconds=rows['extraction'], total_seconds=rows['total'])
    return report


def make_model(root_resolution, depth, min_level, mode):
    """Fresh Model 7 inputs on a ``root_resolution``^3 root grid with the given octree depth."""
    import numpy as np
    from gempy_engine.core.data.regular_grid import RegularGrid
    from tests.fixtures.model7_combination import model7_combination_factory

    inputs, options, descriptor = model7_combination_factory(number_octree_levels=depth, mesh_extraction=True)
    extent = inputs.grid.octree_grid.orthogonal_extent
    extent = (extent.detach().cpu().numpy() if hasattr(extent, 'detach') else np.asarray(extent)).copy()
    inputs.grid.octree_grid = RegularGrid(extent, [root_resolution] * 3)
    evaluation = options.evaluation_options
    evaluation.number_octree_levels = depth
    evaluation.number_octree_levels_surface = depth
    evaluation.octree_min_level = min_level
    evaluation.mesh_extraction_overlap = mode
    return inputs, options, descriptor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-resolution', type=int, default=9)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--min-level', type=int, default=0)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--backends', nargs='+', choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument('--worker', choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(worker(args.worker, args)))
        return 0
    results = {}
    for backend in args.backends:
        command = [sys.executable, __file__, '--worker', backend,
                   '--root-resolution', str(args.root_resolution), '--depth', str(args.depth),
                   '--min-level', str(args.min_level), '--warmups', str(args.warmups),
                   '--repeats', str(args.repeats)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            results[backend] = dict(status='failed', stderr=completed.stderr[-2000:])
            continue
        results[backend] = json.loads(completed.stdout.strip().splitlines()[-1])
    text = json.dumps(dict(arguments={k: v for k, v in vars(args).items() if k not in ('worker', 'output')},
                           results=results), indent=2)
    if args.output is not None:
        args.output.write_text(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
