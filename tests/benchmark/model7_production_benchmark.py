"""Standalone production Model7 benchmark; run each configuration in a fresh process.

Example (from the engine repository):
    /home/leguark/.venv/2025/bin/python tests/benchmark/model7_production_benchmark.py \
        --backend numpy --device cpu --dtype float64 --mode pretty --depth 2

Model timing includes the entire public compute_model lifecycle, including its
internal copies, solution construction and cache/GPU cleanup. Extraction timing
includes production dual_contouring_multi_scalar and matching cache/GPU cleanup.
Fixture construction, extraction input deepcopy, validation and accounting are
outside timing. RSS is Linux process-lifetime RSS, including imports and copies,
not a stage-only memory measurement. CUDA peaks include prepared input tensors.
No production functions are replaced in timed calls. Untimed extraction setup
suppresses only mesh extraction, keeping corner evaluation enabled.
"""

import argparse
import contextlib
import copy
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch


def rss_bytes(field='VmRSS'):
    with open('/proc/self/status', encoding='ascii') as status:
        for line in status:
            if line.startswith(field + ':'):
                return int(line.split()[1]) * 1024
    raise RuntimeError('This benchmark requires Linux /proc/self/status')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['numpy', 'torch'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dtype', choices=['float64', 'float32'], default='float64')
    parser.add_argument('--mode', choices=['none', 'pretty'], required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--scope', choices=['extraction', 'model', 'both'], default='both')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.backend == 'numpy' and args.device != 'cpu':
        parser.error('NumPy only supports --device cpu')
    if args.depth < 2:
        parser.error('--depth must be at least 2 for production surface extraction')
    if args.warmups < 0 or args.repeats < 1:
        parser.error('--warmups must be nonnegative and --repeats must be positive')
    if 'gempy_engine' in sys.modules or 'numpy' in sys.modules or 'torch' in sys.modules:
        raise RuntimeError('Run this standalone script in a fresh process before numerical imports')

    environment = dict(
        DEFAULT_BACKEND='numpy' if args.backend == 'numpy' else 'PYTORCH',
        DEFAULT_PYKEOPS='False', PYKEOPS_SOLVER='False',
        DEFAULT_TENSOR_DTYPE=args.dtype, USE_GPU=str(args.device == 'cuda'),
        DEBUG_MODE='False', GEMPY_TEST_PLOTS='False', MPLBACKEND='Agg',
        NOT_MAKE_INPUT_DEEP_COPY='False', GEMPY_FLAT_STACKS='False',
        OPTIMIZE_MEMORY='True', SET_RAW_ARRAYS_IN_SOLUTION='True',
        ONLY_LITH_SOLUTION='False', LINE_PROFILER_ENABLED='False',
        DUAL_CONTOURING_MULTITHREAD='False', GEMPY_SKIP_TRIANGULATION='0',
        DUAL_CONTOURING_FAULT_OVERLAP_THREADING='False',
        DUAL_CONTOURING_VERTEX_OVERLAP=args.mode,
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', BLIS_NUM_THREADS='1',
    )
    os.environ.update(environment)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    process_start_rss = rss_bytes()

    # Production progress messages go to stderr; stdout is a single JSON document.
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import gempy_engine
        from gempy_engine import config
        from gempy_engine.config import AvailableBackends, DualContouringOverlap
        from gempy_engine.core.backend_tensor import BackendTensor
        from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache
        from tests.fixtures.model7_combination import model7_combination_factory

        model_api = importlib.import_module('gempy_engine.API.model.model_api')
        dc = importlib.import_module('gempy_engine.API.dual_contouring.multi_scalar_dual_contouring')
        torch = None
        if args.backend == 'torch':
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.set_default_dtype(getattr(torch, args.dtype))
            if args.device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable; refusing CPU fallback')
        backend = AvailableBackends.numpy if args.backend == 'numpy' else AvailableBackends.PYTORCH
        BackendTensor._change_backend(backend, use_gpu=args.device == 'cuda',
                                      use_pykeops=False, dtype=args.dtype, grads=False)
        BackendTensor.pykeops_enabled = False

        def check_settings():
            if (config.DUAL_CONTOURING_VERTEX_OVERLAP != DualContouringOverlap[args.mode]
                    or dc.DUAL_CONTOURING_VERTEX_OVERLAP != DualContouringOverlap[args.mode]):
                raise RuntimeError('Requested overlap mode was not applied at import time')
            if (BackendTensor.engine_backend != backend or BackendTensor.dtype != args.dtype
                    or BackendTensor.use_gpu != (args.device == 'cuda')
                    or BackendTensor.use_pykeops or BackendTensor.pykeops_enabled):
                raise RuntimeError('Backend/device/dtype/no-KeOps settings changed')

        def synchronize():
            if args.device == 'cuda':
                torch.cuda.synchronize()

        def make_model():
            model = model7_combination_factory(number_octree_levels=args.depth, mesh_extraction=True)
            _, options, _ = model
            options.debug = False
            options.kernel_options.compute_condition_number = False
            evaluation = options.evaluation_options
            evaluation.number_octree_levels = args.depth
            evaluation.number_octree_levels_surface = args.depth
            evaluation.verbose = False
            if (evaluation.number_octree_levels != args.depth
                    or evaluation.number_octree_levels_surface != args.depth):
                raise RuntimeError('Requested overall/surface depth was not applied')
            return model

        def workload(octrees):
            if len(octrees) != args.depth:
                raise RuntimeError('Actual octree level count differs from requested depth')
            counts = [len(level.grid.octree_grid.values) for level in octrees]
            # Each refined parent is replaced by exactly eight children. Leaves
            # include unrefined cells at all coarser levels, not just the last one.
            refined = [int(np.asarray(BackendTensor.t.to_numpy(
                level.grid.octree_grid.active_cells)).sum()) for level in octrees[1:]]
            if any(children != 8 * parents for children, parents in zip(counts[1:], refined)):
                raise RuntimeError('Unexpected octree parent/child accounting')
            return dict(cells_per_level=counts, refined_parents_per_transition=refined,
                        actual_leaf_cells=sum(counts) - sum(refined),
                        final_level_cells=counts[-1], effective_levels=len(octrees),
                        scalar_dtypes_per_level=[sorted({str(output.exported_fields.scalar_field.dtype)
                                                       for output in level.outputs}) for level in octrees])

        def mesh_info(meshes):
            if not meshes or not any(len(mesh.edges) for mesh in meshes):
                raise RuntimeError('Production extraction returned no triangles')
            per_mesh = []
            for mesh in meshes:
                vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.edges)
                if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                    raise RuntimeError('Invalid/nonfinite mesh vertices')
                if (triangles.ndim != 2 or triangles.shape[1] != 3
                        or not np.issubdtype(triangles.dtype, np.integer)
                        or (triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)))):
                    raise RuntimeError('Invalid mesh triangle indices')
                per_mesh.append(dict(vertices=len(vertices), triangles=len(triangles),
                                     vertex_dtype=str(vertices.dtype)))
            return dict(vertices=sum(m['vertices'] for m in per_mesh),
                        triangles=sum(m['triangles'] for m in per_mesh), meshes=per_mesh,
                        finite_vertices=True, valid_triangle_indices=True)

        def parity(meshes, reference):
            return len(meshes) == len(reference) and all(
                mesh.vertices.shape == vertices.shape
                and np.allclose(mesh.vertices, vertices, rtol=1e-5, atol=1e-5)
                and np.array_equal(mesh.edges, triangles)
                for mesh, (vertices, triangles) in zip(meshes, reference))

        def extract(descriptor, interpolation_input, options, octrees):
            try:
                WeightCache.initialize_cache_dir()
                options.temp_interpolation_values.start_computation_ts = int(time.time())
                options.temp_interpolation_values.current_octree_level = args.depth - 1
                return dc.dual_contouring_multi_scalar(descriptor, interpolation_input, options, octrees)
            finally:
                options.temp_interpolation_values.start_computation_ts = -1
                WeightCache.clear_cache()
                BackendTensor.clear_gpu_memory()

        check_settings()
        startup_memory = dict(process_start_rss_bytes=process_start_rss,
                              imports_rss_bytes=rss_bytes(),
                              imports_peak_rss_bytes=rss_bytes('VmHWM'))
        example = make_model()
        interpolation_input, options, descriptor = example
        evaluation = options.evaluation_options
        settings = dict(backend=args.backend, device=args.device, dtype=args.dtype,
                        mode=args.mode, effective_mode=dc.DUAL_CONTOURING_VERTEX_OVERLAP.name,
                        depth=args.depth, overall_levels=evaluation.number_octree_levels,
                        surface_levels=evaluation.number_octree_levels_surface,
                        root_resolution=np.asarray(BackendTensor.t.to_numpy(
                            interpolation_input.grid.octree_grid.resolution)).tolist(),
                        root_cells=len(interpolation_input.grid.octree_grid.values),
                        root_note='Unmodified Model7 factory root: 2 x 2 x 2, eight cells.',
                        cache_mode=options.cache_mode.name,
                        refinement_mode=evaluation.octree_refinement_mode.value,
                        masking=evaluation.mesh_extraction_masking_options.name,
                        extent_capping=evaluation.mesh_extraction_extent_capping.value,
                        triangulation=evaluation.triangulation_method.value,
                        deduplicate_corners=evaluation.deduplicate_octree_corners,
                        keops=False, environment=environment,
                        torch_threads=None if torch is None else torch.get_num_threads(),
                        torch_interop_threads=None if torch is None else torch.get_num_interop_threads(),
                        thread_note='BLAS/Torch threads pinned to one; production pretty QEF '
                                    'still uses an unconditional ThreadPoolExecutor with default workers.')
        versions = dict(engine=gempy_engine.__version__, numpy=np.__version__,
                        torch=None if torch is None else torch.__version__,
                        python=sys.version, engine_path=gempy_engine.__file__)
        if args.device == 'cuda':
            versions.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
        del example, interpolation_input, options, descriptor, evaluation

        results = {}
        model_reference = None
        for scope in (['model', 'extraction'] if args.scope == 'both' else [args.scope]):
            baseline = None
            baseline_workload = None
            if scope == 'extraction':
                # Mesh extraction must remain enabled during interpolation so
                # solved octrees include corner fields, exactly as in full compute.
                interpolation_input, options, descriptor = make_model()
                with patch.object(model_api, 'dual_contouring_multi_scalar', return_value=None):
                    solved = model_api.compute_model(interpolation_input, options, descriptor)
                baseline = (descriptor, interpolation_input, options, solved.octrees_output)
                baseline_workload = workload(solved.octrees_output)
                del solved, interpolation_input, options, descriptor
            timings = []
            memory = []
            reference = None
            for iteration in range(args.warmups + args.repeats):
                gc.collect()
                call_args = make_model() if scope == 'model' else copy.deepcopy(baseline)
                check_settings()
                synchronize()
                before = dict(pre_call_rss_bytes=rss_bytes(),
                              prepared_peak_rss_bytes=rss_bytes('VmHWM'))
                if args.device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                    before['gpu_pre_call_allocated_bytes'] = torch.cuda.memory_allocated()
                    before['gpu_pre_call_reserved_bytes'] = torch.cuda.memory_reserved()
                start = time.perf_counter()
                result = model_api.compute_model(*call_args) if scope == 'model' else extract(*call_args)
                synchronize()
                elapsed = time.perf_counter() - start
                before.update(post_call_rss_bytes=rss_bytes(),
                              peak_total_rss_bytes=rss_bytes('VmHWM'))
                if args.device == 'cuda':
                    before.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                check_settings()
                meshes = result.dc_meshes if scope == 'model' else result
                counts = mesh_info(meshes)
                run_workload = workload(result.octrees_output) if scope == 'model' else baseline_workload
                if reference is None:
                    reference = [(mesh.vertices.copy(), mesh.edges.copy()) for mesh in meshes]
                    expected_counts, expected_workload = counts, run_workload
                elif not parity(meshes, reference) or counts != expected_counts or run_workload != expected_workload:
                    raise RuntimeError('Output changed between fresh runs (possible mutation/nondeterminism)')
                if scope == 'extraction' and model_reference is not None and not parity(meshes, model_reference):
                    raise RuntimeError('Extraction-only output differs from full-compute baseline')
                if iteration >= args.warmups:
                    timings.append(elapsed)
                    memory.append(before)
                del result, meshes, call_args
            results[scope] = dict(raw_seconds=timings, median_seconds=float(np.median(timings)),
                                  p95_seconds=float(np.percentile(timings, 95)),
                                  memory_per_repeat=memory, workload=expected_workload,
                                  output=expected_counts,
                                  repeat_parity=True if args.warmups + args.repeats > 1 else None,
                                  repeat_parity_comparisons=args.warmups + args.repeats - 1,
                                  parity_note='Ordered triangle indices exact; vertices allclose rtol=atol=1e-5.',
                                  model_baseline_parity=True if scope == 'extraction' and model_reference is not None else None)
            if scope == 'model':
                model_reference = reference
            del baseline, reference
        report = dict(settings=settings, versions=versions, warmups=args.warmups,
                      repeats=args.repeats, scope=args.scope, startup_memory=startup_memory,
                      results=results,
                      memory_note='RSS peak is Linux /proc/self/status VmHWM since exec '
                                  '(not potentially inherited ru_maxrss), including imports, '
                                  'preparation, copies, validation and prior scopes; not stage peak. '
                                  'CUDA allocator peaks reset per call but include live prepared tensors '
                                  'and extraction baseline; exclude non-PyTorch GPU allocations.',
                      timing_note='Preparation/validation excluded; production cache and GPU cleanup '
                                  'included. Extraction starts with cold weights cache and may solve '
                                  'edge-gradient weights; only solved octrees are reused. '
                                  'CUDA synchronized before and after timing. '
                                  'p95 uses NumPy linear interpolation; one repeat is only a pilot.')
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output is not None:
        args.output.write_text(text, encoding='ascii')
    print(text, end='')


if __name__ == '__main__':
    main()
