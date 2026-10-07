"""Six joint-DC cases: actual accepted meshes and guide-only expected rejections."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def failure_cases():
    unit = (np.linspace(0., 1., 9),) * 3
    branches = (np.array([-1., 0., 1., 2.]), np.array([-1., 0., 1., 2.]),
                np.array([0., .5, 1., 1.5]))
    return [
        dict(name='control', title='1. Supported onlap: control', axes=unit,
             fields=[lambda p: p[..., 0] - .25 * p[..., 2] - .4,
                     lambda p: .25 * p[..., 0] + p[..., 2] - .6],
             gradients=[(1., 0., -.25), (.25, 0., 1.)], ownership=[(1, 1), None],
             relations=['ONLAP', 'BASEMENT'], expected=None,
             explanation='Affine two-surface contact; actual DC mesh shown.'),
        dict(name='curved', title='2. Curved erosion boundary', axes=unit,
             fields=[lambda p: p[..., 2] - .43 - .25 * (p[..., 0] - .5) ** 2
                     - .15 * (p[..., 1] - .5) ** 2, lambda p: p[..., 0] - .57],
             gradients=[lambda p: np.stack((-.5 * (p[..., 0] - .5),
                                           -.3 * (p[..., 1] - .5), np.ones(p.shape[:-1])), axis=-1),
                        (1., 0., 0.)], ownership=[None, (0, -1)],
             relations=['ERODE', 'BASEMENT'], expected=None,
             explanation='Approximate joint DC: Hermite QEF on a corner-affine seam.\n'
                         'Actual mesh and seam, not exact analytic projection.'),
        dict(name='grid_aligned', title='3. Contact on a grid face', axes=unit,
             fields=[lambda p: p[..., 0] + p[..., 2] - 1.2,
                     lambda p: p[..., 0] - p[..., 2] - .2],
             gradients=[(1., 0., 1.), (1., 0., -1.)], ownership=[None, (0, -1)],
             relations=['ERODE', 'BASEMENT'], expected='grid_edge_junction',
             explanation='The seam is x=.7, z=.5: on an interior grid face.\n'
                         'Needs a sided boundary-incidence rule, not a bad geometry.'),
        dict(name='multiway', title='4. Three-way geological junction', axes=unit,
             fields=[lambda p: p[..., 0] - .57, lambda p: p[..., 1] - .53,
                     lambda p: p[..., 2] - .43],
             gradients=[(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)],
             ownership=[None, (0, -1), (1, -1)], relations=['ERODE', 'ERODE', 'BASEMENT'],
             expected='unsupported_multiway_junction',
             explanation='Two contact pairs compete inside the same cell.\n'
                         'Requires a richer junction representation.'),
        dict(name='scalar_scale', title='5. Same branches; scalar multiplied by 1e-6', axes=branches,
             fields=[lambda p: 1e-6 * ((p[..., 0] - .5) * (p[..., 1] - .5) - .1)],
             gradients=[lambda p: 1e-6 * np.stack((p[..., 1] - .5, p[..., 0] - .5,
                                                   np.zeros(p.shape[:-1])), axis=-1)],
             ownership=[None], relations=['BASEMENT'], expected='ambiguous_face_tie',
             explanation='The zero set is unchanged; the unscaled case passes.\n'
                         'Absolute saddle tolerance misclassifies this scale.', comparison='scalar'),
        dict(name='coordinate_scale', title='6. Same plane; coordinates multiplied by 1e-6',
             axes=tuple(a * 1e-6 for a in unit),
             fields=[lambda p: p[..., 2] - .43e-6], gradients=[(0., 0., 1.)],
             ownership=[None], relations=['BASEMENT'], expected='degenerate_dual_face',
             explanation='Magnified to unit size for display; unit-domain case passes.\n'
                         'Absolute area tolerance rejects small valid triangles.', comparison='coordinate'),
    ]


def extract_case(case, *, include_reference=False, unsafe_diagnostics=False):
    from gempy_engine.API.dual_contouring.topology_aware_extraction import extract_topology_aware
    from gempy_engine.core.data.stack_relation_type import StackRelationType as R

    xyz = np.stack(np.meshgrid(*case['axes'], indexing='ij'), axis=-1)
    fields = np.array([f(xyz) for f in case['fields']])
    gradients = np.array([g(xyz) if callable(g) else np.broadcast_to(g, xyz.shape)
                          for g in case['gradients']])
    ownership = np.array([np.ones(xyz.shape[:-1]) if pair is None else pair[1] * fields[pair[0]]
                          for pair in case['ownership']])
    n = len(fields)
    return extract_topology_aware(
        case['axes'], fields, list(range(n)), [0] * n,
        [getattr(R, r) for r in case['relations']], [[0.]] * n,
        ownership=ownership, gradient_samples=gradients, include_reference=include_reference,
        unsafe_diagnostics=unsafe_diagnostics,
    )


def evaluate_case(case, *, unsafe_diagnostics=False):
    result, error = None, None
    try:
        result = extract_case(case, unsafe_diagnostics=unsafe_diagnostics)
    except ValueError as exception:
        error = str(exception)
    expected = case['expected']
    if not unsafe_diagnostics and ((expected is None and error is not None) or (expected is not None and
                                                  (error is None or not error.startswith(expected + ':')))):
        raise RuntimeError(f'{case["name"]}: expected {expected!r}, got {error!r}')
    comparison = None
    if case.get('comparison') == 'scalar':
        unscaled = dict(case, fields=[lambda p: (p[..., 0] - .5) * (p[..., 1] - .5) - .1],
                        gradients=[lambda p: np.stack((p[..., 1] - .5, p[..., 0] - .5,
                                                       np.zeros(p.shape[:-1])), axis=-1)])
        comparison = extract_case(unscaled)
    elif case.get('comparison') == 'coordinate':
        unscaled = dict(case, axes=tuple(a / 1e-6 for a in case['axes']),
                        fields=[lambda p: p[..., 2] - .43])
        comparison = extract_case(unscaled)
    report = dict(name=case['name'], status='rejected' if error else 'accepted', error=error,
                  expected=expected, explanation=case['explanation'],
                  triangles=None if result is None else sum(map(len, result['faces'])),
                  per_surface_triangles=None if result is None else list(map(len, result['faces'])),
                  shared_seam_edge_count=None if result is None else len(result['seam_edges']),
                   equivalent_unscaled_case='accepted' if comparison is not None else None)
    if unsafe_diagnostics and result is not None:
        from collections import Counter
        warnings = Counter(result['diagnostics']['unsafe_warnings'])
        defects = []
        edge_counts = []
        for faces in result['faces']:
            points = result['vertices'][faces]
            cross = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
            scale = np.max(np.sum((points - np.roll(points, 1, axis=1)) ** 2, axis=2), axis=1)
            edges = Counter(tuple(sorted((int(a), int(b)))) for triangle in faces
                            for a, b in zip(triangle, np.roll(triangle, -1)))
            edge_counts.append(edges)
            defects.append(dict(degenerate_triangles=int(np.count_nonzero(cross <= 1e-12 * scale)),
                                duplicate_triangles=len(faces) - len({tuple(sorted(t)) for t in faces}),
                                nonmanifold_edges=sum(count > 2 for count in edges.values())))
        shared = set()
        shared_triangles = set()
        for i, edges in enumerate(edge_counts):
            for j, other in enumerate(edge_counts[i + 1:], start=i + 1):
                shared.update(set(edges) & set(other))
                shared_triangles.update({tuple(sorted(t)) for t in result['faces'][i]} &
                                        {tuple(sorted(t)) for t in result['faces'][j]})
        # Display actual shared-ID edges, not unvalidated expected seam evidence.
        result['seam_edges'] = np.array(sorted(shared), dtype=int).reshape(-1, 2)
        report.update(status='unchecked', warnings=dict(warnings), defects=defects,
                      shared_seam_edge_count=len(shared), shared_triangle_count=len(shared_triangles))
    return result, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--off-screen', action='store_true')
    parser.add_argument('--check-only', action='store_true', help='Run all cases without importing PyVista')
    parser.add_argument('--unsafe-diagnostics', action='store_true', help='Bypass rejection guards; emit raw defective meshes')
    parser.add_argument('--screenshot', type=Path)
    args = parser.parse_args()
    if args.screenshot is not None and not args.screenshot.parent.is_dir():
        parser.error('Screenshot parent directory must exist')
    from gempy_engine.config import AvailableBackends
    from gempy_engine.core.backend_tensor import BackendTensor
    BackendTensor._change_backend(AvailableBackends.numpy, use_gpu=False,
                                  use_pykeops=False, dtype='float64', grads=False)
    cases = failure_cases()
    evaluated = [evaluate_case(case, unsafe_diagnostics=args.unsafe_diagnostics) for case in cases]
    print(json.dumps([report for _, report in evaluated], indent=2), flush=True)
    if args.check_only:
        return

    import pyvista as pv
    plotter = pv.Plotter(shape=(2, 3), window_size=(1920, 1120), off_screen=args.off_screen,
                         title='Joint Dual Contouring | UNCHECKED Diagnostic Gallery' if args.unsafe_diagnostics
                         else 'Joint Dual Contouring | Six-Case Gallery')
    plotter.set_background('#17212b', all_renderers=True)
    colors = ['#51b4ec', '#ffb363', '#76d5a0']
    for index, (case, (result, report)) in enumerate(zip(cases, evaluated)):
        plotter.subplot(index // 3, index % 3)
        plotter.add_text(case['title'], position=(.02, .94), viewport=True, font_size=10, color='white')
        low = np.array([a[0] for a in case['axes']])
        width = np.array([a[-1] - a[0] for a in case['axes']])
        if result is not None:
            for surface, faces in enumerate(result['faces']):
                poly = pv.PolyData((result['vertices'] - low) / width,
                                   np.column_stack((np.full(len(faces), 3), faces)).ravel())
                plotter.add_mesh(poly, color=colors[surface], show_edges=True, edge_color='#34404a')
            seam = result['seam_edges']
            if len(seam):
                poly = pv.PolyData((result['vertices'] - low) / width,
                                   lines=np.column_stack((np.full(len(seam), 2), seam)).ravel())
                plotter.add_mesh(poly, color='white', line_width=5)
        else:
            # Dense VTK contours are illustrative guides, never fallback extractor output.
            axes = [np.linspace(a[0], a[-1], 41) for a in case['axes']]
            xyz = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
            for surface, field in enumerate(case['fields']):
                points = (xyz - low) / width
                grid = pv.StructuredGrid(points[..., 0], points[..., 1], points[..., 2])
                grid['field'] = field(xyz).ravel(order='F')
                guide = grid.contour([0.], scalars='field')
                plotter.add_mesh(guide, color=colors[surface], opacity=.65, smooth_shading=True)
            if case['name'] == 'grid_aligned':
                plotter.add_mesh(pv.Line((.7, 0, .5), (.7, 1, .5)), color='white', line_width=6)
                plotter.add_mesh(pv.Plane(center=(.5, .5, .5), direction=(0, 0, 1),
                                         i_resolution=8, j_resolution=8), style='wireframe',
                                 color='#9bafc0', opacity=.35)
            if case['name'] == 'multiway':
                plotter.add_points(np.array([[.57, .53, .43]]), color='white', point_size=14,
                                   render_points_as_spheres=True)
        status = ('UNCHECKED: raw dual mesh' if args.unsafe_diagnostics else 'ACCEPTED: actual dual mesh') if result is not None else 'REJECTED: no extractor mesh'
        plotter.add_text(status, position=(.02, .875), viewport=True, font_size=10,
                         color='#76d5a0' if result is not None else '#ff8799')
        reason = '' if report['error'] is None else report['error'].split(':', 1)[0] + '\n'
        info = reason + case['explanation']
        if args.unsafe_diagnostics and result is not None:
            info = 'Checks bypassed; no repair. White: actual shared-ID edges.\n' + '\n'.join(
                f'S{s}: {d["degenerate_triangles"]} degenerate, {d["duplicate_triangles"]} duplicate, '
                f'{d["nonmanifold_edges"]} nonmanifold edges' for s, d in enumerate(report['defects']))
            info += f'\nCross-surface shared triangles: {report["shared_triangle_count"]}'
            info += f'\nBypassed warnings: {sum(report["warnings"].values())}; details in log.'
        if result is None:
            info += '\nColored surfaces: unmasked analytic guides ONLY.'
        plotter.add_text(info, position=(.02, .025), viewport=True, font_size=9, color='#d0dde8')
        plotter.add_axes()
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.camera.zoom(.72)
    print('READY: opening unchecked gallery.' if args.unsafe_diagnostics else
          'READY: failures verified; opening gallery. Guides are NOT extracted meshes.', flush=True)
    plotter.show(screenshot=str(args.screenshot) if args.screenshot else None)


if __name__ == '__main__':
    main()
