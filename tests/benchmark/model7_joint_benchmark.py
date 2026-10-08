"""Standalone Model7 bank-skin benchmark (no plotting or reference generation).

Run with the project virtualenv. Numerical libraries are pinned to one thread
before import. Split timers reuse solved captures/prepared fields; end-to-end
uses a fresh case each round. Fixture CSV/input/grid preparation is outside all
scope timers to align with model-compute timing. The adapter does not query
fields, so there is no alternate extraction route here.
"""

import argparse
from collections import Counter
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
from time import perf_counter

# Set, rather than default, so an inherited multithreaded environment cannot win.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
for _name, _value in {
    'DEFAULT_BACKEND': 'numpy', 'DEFAULT_PYKEOPS': 'False',
    'DEFAULT_TENSOR_DTYPE': 'float64', 'PYKEOPS_SOLVER': 'False',
    'DEBUG_MODE': 'False', 'GEMPY_FLAT_STACKS': 'False',
    'NOT_MAKE_INPUT_DEEP_COPY': 'False', 'LINE_PROFILER_ENABLED': 'False',
}.items():
    os.environ[_name] = _value

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

with redirect_stdout(sys.stderr):
    from gempy_engine.API.dual_contouring.model7_bank_experiment import (
        capture_model7, extract_bank_skin, query_bank,
    )
    from gempy_engine.config import AvailableBackends
    from gempy_engine.core.backend_tensor import BackendTensor
    from gempy_engine.core.data.generic_grid import GenericGrid
    from tests.fixtures.model7_combination import model7_combination_factory


def prepare(resolution):
    inputs, options, descriptor = model7_combination_factory(mesh_extraction=False)
    points = inputs.surface_points.sp_coords.copy()
    positions = inputs.orientations.dip_positions.copy()
    gradients = inputs.orientations.dip_gradients.copy()
    extent = inputs.grid.octree_grid.orthogonal_extent
    axes = tuple(np.linspace(extent[2*a], extent[2*a+1], resolution + 1) for a in range(3))
    xyz = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    center_axes = tuple((a[:-1] + a[1:]) / 2 for a in axes)
    centers = np.stack(np.meshgrid(*center_axes, indexing='ij'), axis=-1)
    normal = gradients[0] / np.linalg.norm(gradients[0])
    probes = points[:3].mean(axis=0) + np.linspace(-1e-5, 1e-5, 101)[:, None]*normal
    queries = np.concatenate((xyz.reshape(-1, 3), centers.reshape(-1, 3)))
    # Keep the example's solve/capture custom-grid contract, including probes.
    inputs.grid.custom_grid = GenericGrid(np.concatenate((queries, probes)))
    options.evaluation_options.compute_scalar_gradient = True
    return dict(inputs=inputs, options=options, descriptor=descriptor, axes=axes,
                queries=queries, resolution=resolution, originals=(points, positions, gradients))


def sample(case, solution, captures):
    levels = [np.array(o.scalar_field_at_sp, copy=True) for o in solution.root_output.outputs]
    fault, fault_gradient = query_bank(captures[0], case['queries'])
    n = (case['resolution'] + 1)**3
    shape = (case['resolution'] + 1,)*3
    banks = []
    for bank in (0, 1):
        values, gradients = [fault], [fault_gradient]
        for stack in (1, 2):
            scalar, gradient = query_bank(captures[stack], case['queries'], bank)
            values.append(scalar)
            gradients.append(gradient)
        banks.append(dict(
            samples=np.array([v[:n].reshape(shape) for v in values]),
            gradients=np.array([g[:n].reshape((*shape, 3)) for g in gradients]),
            centers=np.array([v[n:].reshape((case['resolution'],)*3) for v in values]),
        ))
    return levels, banks


def extract(case, levels, banks):
    return [extract_bank_skin(case['axes'], b['samples'], b['gradients'], levels, bank,
                              center_samples=b['centers'], include_reference=False)
            for bank, b in enumerate(banks)]


def pipeline(resolution, split=False, case=None):
    if case is None:
        case = prepare(resolution)
    start = perf_counter()
    solution, captures = capture_model7(case['inputs'], case['options'], case['descriptor'])
    solved = perf_counter()
    levels, banks = sample(case, solution, captures)
    sampled = perf_counter()
    meshes = extract(case, levels, banks)
    extracted = perf_counter()
    times = dict(solve_capture=solved-start, bank_corner_gradient_sampling=sampled-solved,
                 both_bank_extraction=extracted-sampled) if split else {}
    return (case, solution, captures, levels, banks, meshes), times


def validate(result):
    """Actual-engine geometry and seam checks, called only after all timers stop."""
    case, solution, captures, levels, banks, meshes = result
    inputs = case['inputs']
    for actual, original in zip((inputs.surface_points.sp_coords, inputs.orientations.dip_positions,
                                 inputs.orientations.dip_gradients), case['originals']):
        np.testing.assert_array_equal(actual, original)
    n = (case['resolution'] + 1)**3
    parity = []
    separator = banks[0]['samples'][0].ravel() - levels[0][0]
    for bank, samples in enumerate(banks):
        for stack in (1, 2):
            output = solution.root_output.outputs[stack]
            slc = output.grid.custom_grid_slice
            drift = captures[stack][0].fault_internal.fault_values_everywhere[0, slc][:n]
            np.testing.assert_array_equal(drift > .5, separator < 0)
            far = np.abs(drift-bank) < 1e-12
            if not far.any():
                raise RuntimeError('No saturated samples for production bank parity')
            scalar = np.asarray(output.exported_fields.scalar_field[slc])[:n]
            gradient = np.stack([g[slc][:n] for g in (output.exported_fields.gx_field,
                                output.exported_fields.gy_field, output.exported_fields.gz_field)], axis=-1)
            errors = [float(np.max(np.abs(samples['samples'][stack].ravel()[far]-scalar[far]))),
                      float(np.max(np.abs(samples['gradients'][stack].reshape(-1, 3)[far]-gradient[far])))]
            if max(errors) > 1e-12:
                raise RuntimeError('Production bank scalar/gradient parity failed')
            parity.append(dict(bank=bank, stack=stack, saturated_nodes=int(far.sum()),
                               scalar_max_error=errors[0], gradient_max_error=errors[1]))
    reports = []
    for bank, mesh in enumerate(meshes):
        if 'reference' in mesh:
            raise RuntimeError('Unexpected reference generation')
        vertices = mesh['vertices']
        if vertices.dtype != np.float64 or not np.isfinite(vertices).all():
            raise RuntimeError('Invalid mesh vertices')
        shared = Counter()
        defects, side_errors, winding, erosion = [], [], [], []
        per_surface_edges = []
        for surface, faces in enumerate(mesh['faces']):
            if not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
                raise RuntimeError('Empty or invalid surface faces')
            edges = Counter(tuple(sorted((int(a), int(b)))) for face in faces
                            for a, b in zip(face, np.roll(face, -1)))
            per_surface_edges.append(edges)
            shared.update(edges.keys())
            xyz = vertices[faces]
            cross = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
            defects.append(dict(degenerate=int(np.sum(np.linalg.norm(cross, axis=1) < 1e-12)),
                                duplicate=len(faces)-len(np.unique(np.sort(faces, axis=1), axis=0)),
                                nonmanifold_edges=sum(count > 2 for count in edges.values())))
            used = vertices[np.unique(faces)]
            fault, _ = query_bank(captures[0], used)
            side_errors.append(0 if surface == 0 else int(np.sum(
                (-1 if bank == 0 else 1)*(fault-levels[0][0]) > 1e-10)))
            if surface >= 2:
                values, _ = query_bank(captures[1], used, bank)
                erosion.append(float(np.max(values-levels[1][0])))
            stack = (0, 1, 2, 2)[surface]
            _, normals = query_bank(captures[stack], xyz.mean(axis=1), None if stack == 0 else bank)
            if surface == 0 and bank == 0:
                normals = -normals
            winding.append(int(np.sum(np.sum(cross*normals, axis=1) <= 0)))
        seams = set(map(tuple, np.sort(mesh['seam_edges'], axis=1)))
        shared_match = {edge for edge, count in shared.items() if count > 1} == seams
        incidence_ok = all(sorted(edges[edge] for edges in per_surface_edges) == [0, 0, 1, 2]
                           for edge in seams)
        passed = (shared_match and incidence_ok and not any(any(d.values()) for d in defects)
                  and not any(side_errors) and not any(winding) and max(erosion) <= 1e-8)
        reports.append(dict(bank=bank, passed=passed, vertices=len(vertices),
                            triangle_counts=[len(f) for f in mesh['faces']], seams=len(seams),
                            defects=defects, bank_side_violations=side_errors, winding_violations=winding,
                            older_surface_max_erosion_residual=erosion,
                            shared_edges_match=shared_match, seam_incidence_valid=incidence_ok,
                            diagnostics=mesh['diagnostics']))
        if not passed:
            raise RuntimeError(f'Bank {bank} validation failed: {reports[-1]}')
    if case['resolution'] == 14:
        if [r['triangle_counts'] for r in reports] != [[442, 78, 156, 182], [442, 234, 676, 878]] or \
                [r['seams'] for r in reports] != [52, 91]:
            raise RuntimeError('Resolution 14 differs from verified Model7 mesh counts')
    return dict(passed=True, observations_preserved=True, production_bank_parity=parity, banks=reports,
                reference_parity='not run: no reference generated',
                interior='center checked, not an interior topology certificate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution', type=int, default=14)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scope', choices=('all', 'splits', 'end-to-end'), default='all',
                        help='all runs independent split and fresh-input end-to-end pipelines')
    args = parser.parse_args()
    if not 2 <= args.resolution <= 16 or args.repeats < 1 or args.warmups < 0:
        parser.error('require resolution 2..16, repeats >= 1 and warmups >= 0')
    if args.output and not args.output.parent.is_dir():
        parser.error('output parent must already exist')
    raw = {}
    last = {}
    with redirect_stdout(sys.stderr), threadpool_limits(limits=1):
        BackendTensor._change_backend(AvailableBackends.numpy, use_gpu=False,
                                      use_pykeops=False, dtype='float64')
        for iteration in range(args.warmups + args.repeats):
            print(f'Model7 {iteration + 1}/{args.warmups + args.repeats}: {args.scope}', file=sys.stderr, flush=True)
            if args.scope in ('all', 'splits'):
                result, times = pipeline(args.resolution, split=True)
                last['splits'] = result
                if iteration >= args.warmups:
                    for key, value in times.items():
                        raw.setdefault(key, []).append(value)
            if args.scope in ('all', 'end-to-end'):
                case = prepare(args.resolution)
                start = perf_counter()
                result, _ = pipeline(args.resolution, case=case)
                elapsed = perf_counter() - start
                last['end-to-end'] = result
                if iteration >= args.warmups:
                    raw.setdefault('end_to_end_fresh_inputs', []).append(elapsed)
        pools = threadpool_info()
        if any(pool['num_threads'] != 1 for pool in pools):
            raise RuntimeError('Numerical thread pool is not single-threaded')
        validation = {scope: validate(result) for scope, result in last.items()}
    report = dict(benchmark='model7_joint_bank_skins', resolution=args.resolution,
                  cells=args.resolution**3, corner_nodes=(args.resolution+1)**3,
                  center_queries=args.resolution**3, solve_only_extra_probes=101,
                  backend='NumPy CPU float64', keops=False, numerical_threads=1,
                  thread_pools=pools, warmups=args.warmups, repeats=args.repeats, scope=args.scope,
                  timings_seconds={key: dict(median=float(np.median(values)),
                                             p95=float(np.percentile(values, 95)), raw_samples=values)
                                   for key, values in raw.items()}, validation=validation,
                  timing_contract={
                      'solve_capture': 'unmodified three-level solve, custom-grid evaluation, gradients, capture copies and API guards; no production mesh',
                      'bank_corner_gradient_sampling': 'one independent and four forced-bank queries on corners AND centers, plus array preparation; excludes probes',
                      'both_bank_extraction': 'prepared fields through extract_bank_skin(include_reference=False), including adapter checks and tags',
                      'end_to_end_fresh_inputs': 'solve/capture + sampling + both-bank extraction on a fresh case each round; fixture CSV/input/grid preparation before timer; separate run, not sum of medians',
                      'intrinsic_overhead': 'extractor topology/ownership/incidence guards and diagnostic metadata cannot be disabled by API and remain timed',
                      'excluded': 'fixture CSV/input/grid preparation in all scopes, imports, warmups, external validation, parity/quality reports, plotting, reference generation, JSON serialization',
                      'validation': 'last measured result per scope only, outside all timers',
                  }, caveats=[
                      'Uniform bounded experiment, at most 4096 cells; some resolutions (including 16) may reject topology.',
                      'Two disjoint bank-side fault skins, NOT a single conforming fault mesh; finite/multiple faults and adaptive grids unsupported.',
                      'Single process, process-local non-thread-safe capture; numerical libraries limited to one thread.',
                      'p95 uses NumPy linear percentile; one-repeat pilot has median == p95 and is not a performance estimate.',
                  ])
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
