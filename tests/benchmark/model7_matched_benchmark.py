"""Opt-in matched solved-octree Model7 benchmark (run in a fresh process).

Example:
    /home/leguark/.venv/2025/bin/python tests/benchmark/model7_matched_benchmark.py \
        --scope extraction --warmups 0 --repeats 1 --output /tmp/model7.json

Extraction times only the real production dispatcher, inside compute_model's
live cache lifecycle. Copies, cache restoration, validation and reports are not
timed. Model times unpatched compute_model, including its normal cleanup.
Matching solved inputs does NOT imply equivalent mesh coverage/field semantics.
"""

import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time
from collections import Counter
from unittest.mock import patch


MODES = ('joint', 'none', 'pretty', 'joint_contacts')
ORDERS = tuple(itertools.permutations(MODES))


def rss_bytes(field):
    with open('/proc/self/status', encoding='ascii') as stream:
        for line in stream:
            if line.startswith(field + ':'):
                return int(line.split()[1]) * 1024
    raise RuntimeError(f'Missing Linux RSS field {field}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-resolution', type=int, default=9)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--min-level', type=int, default=0)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--scope', choices=('extraction', 'model', 'both'), default='both')
    parser.add_argument('--output', type=Path, help='JSON destination; stdout is also JSON')
    args = parser.parse_args(argv)
    if args.root_resolution < 2 or args.depth < 2 or not 0 <= args.min_level < args.depth:
        parser.error('root-resolution >= 2, depth >= 2, and 0 <= min-level < depth required')
    if args.warmups < 0 or args.repeats < 1:
        parser.error('warmups >= 0 and repeats >= 1 required')
    if args.output is not None and not args.output.parent.is_dir():
        parser.error('Output parent directory must already exist')
    if any(name in sys.modules for name in ('numpy', 'torch', 'gempy_engine')):
        raise RuntimeError('Run standalone in a fresh process before numerical imports')
    environment = dict(
        DEFAULT_BACKEND='numpy', DEFAULT_TENSOR_DTYPE='float64', USE_GPU='False',
        CUDA_VISIBLE_DEVICES='', DEFAULT_PYKEOPS='False', PYKEOPS_SOLVER='False',
        DEBUG_MODE='False', GEMPY_TEST_PLOTS='False', MPLBACKEND='Agg',
        NOT_MAKE_INPUT_DEEP_COPY='False', GEMPY_FLAT_STACKS='False',
        OPTIMIZE_MEMORY='True', SET_RAW_ARRAYS_IN_SOLUTION='True',
        ONLY_LITH_SOLUTION='False', LINE_PROFILER_ENABLED='False',
        DUAL_CONTOURING_MULTITHREAD='False', GEMPY_SKIP_TRIANGULATION='0',
        DUAL_CONTOURING_FAULT_OVERLAP_THREADING='False',
        DUAL_CONTOURING_VERTEX_OVERLAP='none',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', BLIS_NUM_THREADS='1',
    )
    os.environ.update(environment)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    wall_start = time.perf_counter()
    startup_rss = rss_bytes('VmRSS')
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import gempy_engine
        from gempy_engine.config import AvailableBackends, resolve_dual_contouring_overlap
        from gempy_engine.core.data import InterpolationOptions
        from gempy_engine.core.backend_tensor import BackendTensor as BT
        from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache
        from examples.adaptive_joint_model7 import make_model, hierarchy_report, joint_seams
        from gempy_engine.API.dual_contouring.adaptive_joint_extraction import _collect_joint_leaves
        from types import SimpleNamespace

        model_api = importlib.import_module('gempy_engine.API.model.model_api')
        scalar_field = importlib.import_module('gempy_engine.API.interp_single._interp_scalar_field')
        cache_policy = importlib.import_module('gempy_engine.modules.weights_cache.weight_cache_policy')
        production_extract = model_api.dual_contouring_multi_scalar
        BT._change_backend(AvailableBackends.numpy, use_gpu=False,
                           use_pykeops=False, dtype='float64', grads=False)
        BT.pykeops_enabled = False
        if WeightCache.memory_cache:
            raise RuntimeError('Fresh process unexpectedly has cached weights')
        if sys.getprofile() is not None:
            raise RuntimeError('Run without an existing profiler; timed calls must be uninstrumented')

        def check_backend():
            if (BT.engine_backend != AvailableBackends.numpy or BT.dtype != 'float64'
                    or BT.use_gpu or BT.use_pykeops or BT.pykeops_enabled or BT.COMPUTE_GRADS):
                raise RuntimeError('NumPy CPU float64/no-KeOps configuration changed')

        def arrays(value, path='', ancestors=()):
            # Walk stored data only, never lazy properties; paths survive deepcopy.
            if isinstance(value, np.ndarray):
                yield path, value
            elif id(value) not in ancestors:
                ancestors = ancestors + (id(value),)
                if isinstance(value, dict):
                    items = sorted(value.items(), key=lambda item: str(item[0]))
                elif isinstance(value, (list, tuple)):
                    items = enumerate(value)
                elif hasattr(value, '__dict__') and not isinstance(value, type):
                    items = sorted(vars(value).items())
                else:
                    return
                for key, child in items:
                    yield from arrays(child, f'{path}/{key}', ancestors)

        def fingerprint(value):
            rows = {}
            for path, array in arrays(value):
                if array.dtype.hasobject:
                    raise RuntimeError(f'Object array cannot be byte-fingerprinted: {path}')
                rows[path] = dict(shape=list(array.shape), dtype=str(array.dtype),
                                  sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest(),
                                  readonly=not array.flags.writeable)
            digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode('ascii')).hexdigest()
            return dict(sha256=digest, arrays=rows)

        def clone(value):
            result = copy.deepcopy(value)
            originals = dict(arrays(value))
            for path, array in arrays(result):
                # NumPy deepcopy otherwise silently makes retained solves writable.
                array.flags.writeable = originals[path].flags.writeable
                if np.shares_memory(array, originals[path]):
                    raise RuntimeError(f'Deepcopy shares array storage: {path}')
            if fingerprint(result) != fingerprint(value):
                raise RuntimeError('Deepcopy changed array bytes or flags')
            return result

        def cache_signature(cache):
            # Include every payload field, not only arrays: entry hashes carry
            # solver/kernel/timestamp provenance and must never be rewritten.
            def payload(value):
                if isinstance(value, np.ndarray):
                    return fingerprint(value)
                if isinstance(value, dict):
                    if any(not isinstance(key, str) for key in value):
                        raise RuntimeError('Unsupported non-string cache metadata key')
                    return {key: payload(child) for key, child in sorted(value.items())}
                if isinstance(value, (list, tuple)):
                    return dict(type=type(value).__name__, values=[payload(child) for child in value])
                if value is None or type(value) in (str, bool, int, float):
                    return dict(type=type(value).__name__, value=value)
                raise RuntimeError(f'Unsupported cache metadata type: {type(value).__name__}')

            entries = payload(cache)
            digest = hashlib.sha256(json.dumps(entries, sort_keys=True, allow_nan=False).encode('ascii')).hexdigest()
            return dict(sha256=digest, entries=entries)

        def effective_settings(options, mode, descriptor):
            effective = resolve_dual_contouring_overlap(
                options.evaluation_options.mesh_extraction_overlap,
                model_api.dc_dispatch.DUAL_CONTOURING_VERTEX_OVERLAP).name
            if effective != mode:
                raise RuntimeError('Requested extraction mode did not resolve correctly')
            overrides = descriptor.stack_structure.interpolation_options_per_stack or ()
            if any(override is not None and override.cache_mode != InterpolationOptions.CacheMode.IN_MEMORY_CACHE
                   for override in overrides) or options.cache_mode != InterpolationOptions.CacheMode.IN_MEMORY_CACHE:
                raise RuntimeError('Benchmark requires IN_MEMORY_CACHE for every stack; disk state is not restored')
            return dict(requested_overlap=mode, effective_overlap=effective, cache_mode=options.cache_mode.name,
                        evaluation_fields={key: repr(value) for key, value in vars(options.evaluation_options).items()},
                        kernel_fields={key: repr(value) for key, value in vars(options.kernel_options).items()})

        def fixture(mode):
            inputs, options, descriptor = make_model(
                args.root_resolution, args.depth, args.min_level, mode)
            options.debug = False
            options.kernel_options.compute_condition_number = False
            options.evaluation_options.verbose = False
            options.cache_mode = InterpolationOptions.CacheMode.IN_MEMORY_CACHE
            for override in descriptor.stack_structure.interpolation_options_per_stack or ():
                if override is not None:
                    override.cache_mode = InterpolationOptions.CacheMode.IN_MEMORY_CACHE
            effective_settings(options, mode, descriptor)
            if len(inputs.surface_points.sp_coords) != 102 or len(inputs.orientations.dip_positions) != 8:
                raise RuntimeError('Original Model7 observations changed')
            return inputs, options, descriptor

        def workload(levels):
            report = hierarchy_report(SimpleNamespace(octrees_output=levels))
            if len(levels) != args.depth:
                raise RuntimeError('Unexpected solved octree depth')
            counts = report['level_cell_counts']
            refined = [int(np.count_nonzero(level.grid.octree_grid.active_cells))
                       for level in levels[1:]]
            if any(n != 8 * p for n, p in zip(counts[1:], refined)):
                raise RuntimeError('Invalid parent/child counts')
            report.update(actual_leaf_cells=sum(counts) - sum(refined),
                          refined_parents=refined, final_level_cells=counts[-1])
            origins, spans, shape, extent, samples, _ = _collect_joint_leaves(
                levels, len(levels[0].outputs))
            order = np.lexsort((spans, origins[:, 2], origins[:, 1], origins[:, 0]))
            report['canonical_leaf_identity_and_fields_sha256'] = fingerprint(
                (origins[order], spans[order], shape, extent, samples[:, order]))['sha256']
            return report

        references = {}
        solved_digest = None
        results = {}

        def observe(scope, mode, meshes, levels, seconds, round_index, audit):
            nonlocal solved_digest
            check_backend()
            solved = fingerprint(levels)
            if solved_digest is None:
                solved_digest = solved
            elif solved != solved_digest:
                raise RuntimeError('Solved hierarchy/node fields/masks/weights differ across calls')
            if len(meshes) != 4 or not any(len(mesh.edges) for mesh in meshes):
                raise RuntimeError('Expected four nonempty Model7 surface exports')
            surfaces = []
            for mesh in meshes:
                vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.edges)
                if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                    raise RuntimeError('Invalid/nonfinite mesh vertices')
                if (faces.ndim != 2 or faces.shape[1] != 3 or
                        not np.issubdtype(faces.dtype, np.integer) or
                        faces.size and (faces.min() < 0 or faces.max() >= len(vertices))):
                    raise RuntimeError('Invalid triangle indices')
                surfaces.append(dict(stack=mesh.stack_index, surface=mesh.surface_index,
                                     vertices=len(vertices), triangles=len(faces),
                                     oriented_geometry=fingerprint((vertices, faces))['sha256'],
                                     triangle_banks=dict(Counter(map(str, mesh.joint_face_bank_ids)))
                                     if mode == 'joint' else None))
            output = dict(surfaces=surfaces, vertices=sum(s['vertices'] for s in surfaces),
                          triangles=sum(s['triangles'] for s in surfaces))
            if mode == 'joint_contacts':
                _, seams, _ = joint_seams(meshes)
                contact = meshes[0].contact_report
                output.update(seam_edges=len(seams),
                              coarsefine_seam_edge_count=int(contact['coarsefine_seam_edge_count']),
                              borrowed_vertex_count=contact['borrowed_vertex_count'],
                              fault_overridden_junction_cells=len(contact['fault_overridden_junction_cells']),
                              no_uniform_fallback=contact['no_uniform_fallback'],
                              no_post_reconciliation=contact['no_post_reconciliation'])
            if mode == 'joint':
                _, seams, banks = joint_seams(meshes)
                contact = meshes[0].contact_report
                output.update(seam_edges=len(seams), seam_banks=banks,
                              coarsefine_seam_edge_count=int(contact['coarsefine_seam_edge_count']),
                              cross_bank_id_joins=contact['cross_bank_id_joins'],
                              no_uniform_fallback=contact['no_uniform_fallback'],
                              no_post_reconciliation=contact['no_post_reconciliation'])
            reference = references.get(mode)
            if reference is None:
                references[mode] = clone(meshes)
            else:
                for mesh, expected in zip(meshes, reference):
                    np.testing.assert_allclose(mesh.vertices, expected.vertices, rtol=1e-5, atol=1e-5)
                    np.testing.assert_array_equal(mesh.edges, expected.edges)
                    if mode == 'joint':
                        for field in ('joint_vertex_ids', 'joint_face_bank_ids', 'joint_bank_ids', 'joint_seam_edges'):
                            np.testing.assert_array_equal(getattr(mesh, field), getattr(expected, field))
                    if mode == 'joint_contacts':
                        for field in ('joint_vertex_ids', 'joint_seam_edges'):
                            np.testing.assert_array_equal(getattr(mesh, field), getattr(expected, field))
            if round_index is None:
                return output
            entry = results[scope].setdefault(mode, dict(raw_seconds=[], warmup_seconds=[],
                                                       output=output, hierarchy=workload(levels),
                                                       calls=[], parity_comparisons=0))
            # Counts and oriented byte digests must also be stable, not just allclose.
            if output != entry['output']:
                raise RuntimeError(f'{mode} output counts/digests changed')
            entry['parity_comparisons'] += int(reference is not None)
            entry['warmup_seconds' if round_index < args.warmups else 'raw_seconds'].append(seconds)
            entry['calls'].append(dict(round=round_index, warmup=round_index < args.warmups,
                                       solved_sha256=solved['sha256'], **audit))

        scopes = ('extraction', 'model') if args.scope == 'both' else (args.scope,)
        cache_snapshot_report = None
        extraction_validation = {}
        for scope in scopes:
            results[scope] = {}
            if scope == 'extraction':
                def live_extraction(**baseline):
                    nonlocal cache_snapshot_report
                    baseline_before = fingerprint(baseline)
                    live_cache = WeightCache.memory_cache
                    if not live_cache:
                        raise RuntimeError('Missing live production weight cache')
                    for index, output in enumerate(baseline['octree_list'][-1].outputs):
                        actual = output.weights
                        entry = live_cache[f"{baseline['options'].cache_model_name}.{index}"]
                        if (actual.flags.writeable or np.shares_memory(actual, entry['weights']) or
                                actual.dtype != entry['weights'].dtype or
                                actual.shape != entry['weights'].shape or not np.isfinite(actual).all() or
                                actual.tobytes() != entry['weights'].tobytes()):
                            raise RuntimeError('Actual retained solve/cache provenance differs')
                    live_signature = cache_signature(live_cache)
                    cache = clone(live_cache)
                    if cache_signature(cache) != live_signature:
                        raise RuntimeError('Cache snapshot metadata or payload changed during copying')
                    cache_snapshot_report = dict(signature=live_signature, live_provenance_nonalias_checked=True,
                                                 temp_context=vars(baseline['options'].temp_interpolation_values).copy())

                    for mode in MODES:
                        call = clone(baseline)
                        call['options'].evaluation_options.mesh_extraction_overlap = mode
                        settings = effective_settings(call['options'], mode, call['data_descriptor'])
                        WeightCache.memory_cache = clone(cache)
                        signature_before = cache_signature(WeightCache.memory_cache)
                        before = fingerprint(call['octree_list'])
                        audit = dict(actual_solver_calls=0,
                                     cache_route_counts={route.name: 0 for route in cache_policy.WeightCacheRoute},
                                     cache_returns=[])

                        def profile(frame, event, result):
                            if event == 'call' and frame.f_code is scalar_field._solve_interpolation_result.__code__:
                                audit['actual_solver_calls'] += 1
                            elif event == 'return' and frame.f_code is cache_policy.resolve_weight_cache.__code__:
                                if result is None:
                                    raise RuntimeError('Cache resolver did not return a decision')
                                audit['cache_route_counts'][result.route.name] += 1
                                audit['cache_returns'].append(dict(stack=frame.f_locals['stack_number'],
                                                                   key=result.key, fingerprint=result.fingerprint,
                                                                   route=result.route.name))

                        check_backend()
                        sys.setprofile(profile)
                        try:
                            meshes = production_extract(**call)
                        finally:
                            sys.setprofile(None)
                        if (audit['actual_solver_calls'] or
                                audit['cache_route_counts']['CACHED'] != len(call['octree_list'][-1].outputs) or
                                any(count for route, count in audit['cache_route_counts'].items() if route != 'CACHED')):
                            raise RuntimeError(f'{mode} untimed extraction did not exclusively reuse three fixed solves: {audit}')
                        signature_after = cache_signature(WeightCache.memory_cache)
                        if signature_after != signature_before or signature_before != live_signature:
                            raise RuntimeError(f'{mode} untimed extraction changed cache entries/hash/payload')
                        if fingerprint(call['octree_list']) != before or fingerprint(baseline) != baseline_before:
                            raise RuntimeError('Untimed extraction mutated solved arrays or baseline')
                        output = observe(scope, mode, meshes, call['octree_list'], None, None, {})
                        extraction_validation[mode] = dict(
                            **audit, settings=settings, output=output, cache_signature_before=signature_before,
                            cache_signature_after=signature_after, cache_unchanged=True,
                            solved_before_sha256=before['sha256'], solved_after_sha256=fingerprint(call['octree_list'])['sha256'],
                            profile_scope='Separate untimed production extraction; calling-thread solve/resolver code events.')
                        del meshes, call
                    last_meshes = None
                    for round_index in range(args.warmups + args.repeats):
                        for mode in ORDERS[round_index % len(ORDERS)]:
                            call = clone(baseline)
                            call['options'].evaluation_options.mesh_extraction_overlap = mode
                            settings = effective_settings(call['options'], mode, call['data_descriptor'])
                            WeightCache.memory_cache = clone(cache)
                            cache_before = cache_signature(WeightCache.memory_cache)
                            before = fingerprint(call['octree_list'])
                            check_backend()
                            start = time.perf_counter()
                            meshes = production_extract(**call)
                            seconds = time.perf_counter() - start
                            cache_after = cache_signature(WeightCache.memory_cache)
                            if cache_after != cache_before or cache_before != live_signature:
                                raise RuntimeError(f'{mode} timed extraction changed cache entries/hash/payload')
                            after = fingerprint(call['octree_list'])
                            if before != after or fingerprint(baseline) != baseline_before:
                                raise RuntimeError('Extraction mutated solved arrays or original baseline')
                            observe(scope, mode, meshes, call['octree_list'], seconds, round_index,
                                    dict(clone_before_sha256=before['sha256'], clone_after_sha256=after['sha256'],
                                         original_before_sha256=baseline_before['sha256'],
                                         original_after_sha256=fingerprint(baseline)['sha256'],
                                         settings=settings, cache_unchanged=True,
                                         cache_before_sha256=cache_before['sha256'],
                                         cache_after_sha256=cache_after['sha256']))
                            last_meshes = meshes
                    # No per-call clears: normal compute_model finally owns cleanup.
                    return last_meshes

                model = fixture('joint')
                with patch.object(model_api, 'dual_contouring_multi_scalar', side_effect=live_extraction):
                    solution = model_api.compute_model(*model)
                del solution, model
            else:
                for round_index in range(args.warmups + args.repeats):
                    for mode in ORDERS[round_index % len(ORDERS)]:
                        model = fixture(mode)
                        settings = effective_settings(model[1], mode, model[2])
                        observation_before = fingerprint((model[0].surface_points, model[0].orientations))
                        check_backend()
                        start = time.perf_counter()
                        solution = model_api.compute_model(*model)
                        seconds = time.perf_counter() - start
                        if fingerprint((model[0].surface_points, model[0].orientations)) != observation_before:
                            raise RuntimeError('Model computation changed observations')
                        observe(scope, mode, solution.dc_meshes, solution.octrees_output, seconds, round_index,
                                dict(settings=settings, observations_preserved=True,
                                     cache_cleared=not WeightCache.memory_cache))
                        del solution, model
            if WeightCache.memory_cache:
                raise RuntimeError('Production finally did not clear memory cache')
            for entry in results[scope].values():
                entry.update(median_seconds=float(np.median(entry['raw_seconds'])),
                             p95_seconds=float(np.percentile(entry['raw_seconds'], 95)))
            results[scope]['ratios'] = {
                f'{numerator}_over_{denominator}': results[scope][numerator]['median_seconds'] /
                results[scope][denominator]['median_seconds']
                for numerator, denominator in (('joint', 'none'), ('joint', 'pretty'), ('pretty', 'none'),
                                               ('joint_contacts', 'none'), ('joint_contacts', 'pretty'),
                                               ('joint_contacts', 'joint'))}

        with open('/proc/cpuinfo', encoding='ascii') as stream:
            cpu_name = next((line.split(':', 1)[1].strip() for line in stream
                             if line.startswith('model name')), platform.processor())
        libraries = {}
        for name in ('numpy', 'scipy', 'pandas', 'pydantic', 'torch', 'threadpoolctl'):
            try:
                libraries[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                libraries[name] = None
        settings_model = fixture('joint')
        if 'pyvista' in sys.modules:
            raise RuntimeError('Benchmark unexpectedly imported PyVista')
        try:
            from threadpoolctl import threadpool_info
        except ImportError:
            numerical_pools = None
        else:
            numerical_pools = threadpool_info()
            if any(pool['num_threads'] != 1 for pool in numerical_pools):
                raise RuntimeError('A loaded numerical thread pool is not pinned to one')
        report = dict(
            status='validated', arguments=vars(args) | {'output': None if args.output is None else str(args.output)},
            backend='NumPy CPU float64; no GPU; no KeOps', environment=environment,
            evaluation_options=repr(settings_model[1].evaluation_options),
            kernel_options=repr(settings_model[1].kernel_options), cache_mode=settings_model[1].cache_mode.name,
            host=dict(cpu=cpu_name, logical_cpus=os.cpu_count(), platform=platform.platform(),
                      hostname=platform.node(), interpreter=sys.executable, python=sys.version,
                      libraries=libraries, engine_version=gempy_engine.__version__, engine_path=gempy_engine.__file__),
            numerical_thread_pools=numerical_pools, pyvista_imported=False,
            orders=[dict(round=i, warmup=i < args.warmups, modes=list(ORDERS[i % len(ORDERS)]))
                    for i in range(args.warmups + args.repeats)],
            order_note=f'{len(ORDERS)} deterministic mode permutations cycle; default rounds are not position-balanced.',
            thread_note='Numerical library threads pinned to one. No concurrent benchmark calls. '
                        'Production pretty uses an unconditional default-worker QEF ThreadPoolExecutor.',
            semantics=dict(joint='All mixed-level leaves; bank-conditioned fault skins/shared seams.',
                           none='Final-level cells only; ordinary smooth-fault fields.',
                           pretty='Final-level cells only; ordinary smooth-fault fields plus overlap QEF/averaging.',
                           joint_contacts='All mixed-level leaves; production smooth-fault fields; joint contact '
                                          'seams; one fault mesh whose vertices affected horizons borrow.'),
            parity_note='Within-mode oriented triangles/IDs exact, vertices allclose rtol=atol=1e-5; '
                        'byte digests additionally required stable. No cross-mode mesh equivalence asserted.',
            timing_note='Extraction production call only, including all field/gradient queries. '
                        'One solve setup; live cache snapshots restored outside timers, no extra per-call cleanup. '
                        'Separate untimed extraction per mode profiles cache routes and actual solver calls; '
                        'no profiler or solver/resolver replacements in measured calls. '
                        'Model unpatched compute_model includes cleanup; fixture construction excluded. '
                        'Copies, checks and reporting excluded. p95 uses NumPy linear interpolation.',
            solved_fingerprint=solved_digest, live_cache_snapshot=cache_snapshot_report,
            extraction_validation=extraction_validation,
            results=results, final_memory_cache_empty=not WeightCache.memory_cache,
            memory=dict(startup_rss_bytes=startup_rss, final_rss_bytes=rss_bytes('VmRSS'),
                        peak_since_exec_bytes=rss_bytes('VmHWM')),
            memory_note='Linux VmHWM since exec includes imports, copies, validation and all prior scopes; not stage RSS.',
            wall_seconds=time.perf_counter() - wall_start,
        )
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output is not None:
        args.output.write_text(text, encoding='ascii')
    print(text, end='')


if __name__ == '__main__':
    main()
