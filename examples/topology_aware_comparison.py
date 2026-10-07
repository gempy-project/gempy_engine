"""CPU planar onlap or curved erosion: production QEF versus approximate joint DC.

This isolates vertex placement and junction incidence, not full compute_model,
adaptive octrees, production masking, or contact_aware post-processing.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NORMALS = np.array([[1., 0., -.25], [.25, 0., 1.]])
LEVELS = np.array([.4, .6])


def shared_edges(mesh):
    """Actual common mesh edges, not nearby points or analytic seam segments."""
    edges = []
    for faces in mesh['faces']:
        edges.append({tuple(sorted((int(a), int(b)))) for face in faces
                      for a, b in zip(face, np.roll(face, -1))})
    return np.array(sorted(set.intersection(*edges)), dtype=int).reshape(-1, 2)


def quality_report(mesh):
    from gempy_engine.modules.dual_contouring.topology_extraction import triangle_quality

    faces = np.concatenate(mesh['faces'])
    affected = np.concatenate(mesh['affected_faces'])
    return {region: triangle_quality(mesh['vertices'], faces[mask])
            for region, mask in [('total', np.ones(len(faces), dtype=bool)),
                                 ('contact', affected), ('noncontact', ~affected)]}


def parity_report(joint, reference):
    reference_ids = {key: i for i, key in enumerate(reference['vertex_keys'])}
    reports = []
    for surface, (faces, ref_faces, affected, ref_affected) in enumerate(zip(
            joint['faces'], reference['faces'], joint['affected_faces'], reference['affected_faces'])):
        masks_equal = np.array_equal(affected, ref_affected)
        mapped = np.array([[reference_ids[joint['vertex_keys'][i]] for i in face]
                           for face in faces[~affected]], dtype=int).reshape(-1, 3)
        reports.append(dict(
            surface=surface, noncontact_triangle_count=int((~affected).sum()),
            affected_masks_exact=masks_equal,
            connectivity_exact=np.array_equal(mapped, ref_faces[~ref_affected]),
            geometry_exact=np.array_equal(joint['vertices'][faces[~affected]],
                                          reference['vertices'][ref_faces[~ref_affected]]),
        ))
    regular_ids = [i for i, key in enumerate(joint['vertex_keys']) if key[0] == 'regular']
    regular_exact = all(np.array_equal(joint['vertices'][i],
                                      reference['vertices'][reference_ids[joint['vertex_keys'][i]]])
                        for i in regular_ids)
    return dict(method='Exact array equality; oriented connectivity remapped through vertex_keys',
                surfaces=reports, retained_regular_vertex_count=len(regular_ids),
                retained_regular_vertices_exact=regular_exact,
                passed=regular_exact and all(r['affected_masks_exact'] and r['connectivity_exact']
                                             and r['geometry_exact'] for r in reports))


def residual_report(mesh, fields):
    """Analytic absolute scalar residuals at unique, surface-referenced vertices."""
    reports = []
    for surface, (field, faces, affected) in enumerate(zip(
            fields, mesh['faces'], mesh['affected_faces'])):
        regions = {}
        for region, ids in [('total', np.unique(faces)),
                            ('contact', np.unique(faces[affected])),
                            ('shared_seam', np.intersect1d(np.unique(shared_edges(mesh)),
                                                         np.unique(faces)))]:
            values = np.abs(field(mesh['vertices'][ids]))
            regions[region] = dict(
                count=len(ids), mean=float(values.mean()) if len(ids) else None,
                rms=float(np.sqrt(np.mean(values ** 2))) if len(ids) else None,
                percentiles=dict(zip(['0', '5', '50', '95', '100'],
                                     np.percentile(values, [0, 5, 50, 95, 100]).tolist()))
                if len(ids) else None,
            )
        reports.append(dict(surface=surface, **regions))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['planar', 'curved'], default='planar')
    parser.add_argument('--resolution', type=int, default=8, help='Uniform cells per axis (2 to 16)')
    parser.add_argument('--off-screen', action='store_true')
    parser.add_argument('--screenshot', type=Path)
    parser.add_argument('--analytic-seam', action='store_true', help='Optional cyan analytic guide')
    args = parser.parse_args()
    suffix = '-curved' if args.case == 'curved' else ''
    if args.screenshot is None:
        args.screenshot = Path(f'/tmp/opencode/topology-dual{suffix}-comparison.png')
    if not 2 <= args.resolution <= 16:
        parser.error('--resolution must be between 2 and 16 (maximum 4096 cells)')
    if not args.screenshot.parent.is_dir():
        parser.error(f'Screenshot parent directory does not exist: {args.screenshot.parent}')
    try:
        import pyvista as pv
    except ImportError:
        parser.error('Viewer requires optional PyVista')

    from gempy_engine.API.dual_contouring.topology_aware_extraction import extract_topology_aware
    from gempy_engine.core.data.stack_relation_type import StackRelationType as R

    axes = (np.linspace(0., 1., args.resolution + 1),) * 3
    xyz = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    if args.case == 'curved':
        from examples.topology_aware_failures import failure_cases

        case = next(c for c in failure_cases() if c['name'] == 'curved')
        fields = case['fields']
        samples = np.array([field(xyz) for field in fields])
        gradients = np.array([g(xyz) if callable(g) else np.broadcast_to(g, xyz.shape)
                              for g in case['gradients']])
        ownership = np.array([np.ones(xyz.shape[:-1]) if pair is None else pair[1] * samples[pair[0]]
                              for pair in case['ownership']])
        relations, levels = [R.ERODE, R.BASEMENT], np.zeros(2)
        ownership_description = 'surface 0: 1; surface 1: -f0'
    else:
        fields = [lambda p, n=n, level=level: p @ n - level
                  for n, level in zip(NORMALS, LEVELS)]
        samples = np.einsum('...j,sj->s...', xyz, NORMALS)
        gradients = np.broadcast_to(NORMALS[:, None, None, None, :], (*samples.shape, 3))
        ownership = np.ones_like(samples)
        ownership[0] = samples[1] - LEVELS[1]
        relations, levels = [R.ONLAP, R.BASEMENT], LEVELS
        ownership_description = 'surface 0: f1 - 0.6; surface 1: 1'
    centers = (axes[0][:-1] + axes[0][1:]) / 2
    center_xyz = np.stack(np.meshgrid(centers, centers, centers, indexing='ij'), axis=-1)
    interior = np.array([field(center_xyz) for field in fields])
    joint = extract_topology_aware(
        axes, samples, [0, 1], [0, 0], relations, [[levels[0]], [levels[1]]],
        ownership=ownership, gradient_samples=gradients, interior_samples=interior,
        include_reference=True,
    )
    # The API reference literally calls production generate_dual_contouring_vertices;
    # it reuses the joint plans' retained primal edges and production QUADS 1-3 split.
    reference = joint['reference']
    parity = parity_report(joint, reference)
    joint_seam = shared_edges(joint)
    planned_seam = np.sort(joint['seam_edges'], axis=1)
    seam_parity = set(map(tuple, joint_seam)) == set(map(tuple, planned_seam))
    metrics = dict(
        scope='Identical Hermite data, retained primal-edge support, production QUADS 1-3 split; '
              'not full compute_model or contact_aware post-processing',
        case=args.case, resolution=args.resolution, cell_count=args.resolution ** 3,
        normals=NORMALS.tolist() if args.case == 'planar' else 'Analytic gallery gradients',
        levels=levels.tolist(), ownership=ownership_description,
        contact_region='affected_faces: all triangles incident to junction representatives, '
                       'including corresponding reference triangles',
        quality_definition=dict(angle='Minimum interior angle per triangle, degrees',
                                aspect='longest edge / smallest altitude = L^2 / (2A)',
                                percentile_levels=[0, 5, 50, 95, 100]),
        baseline=quality_report(reference), joint=quality_report(joint),
        analytic_absolute_scalar_residual=dict(
            definition='Unique vertices referenced by each surface; contact uses affected_faces; '
                       'shared_seam uses actual common mesh edges. '
                       'Approximate joint DC, not exact analytic projection',
            percentile_levels=[0, 5, 50, 95, 100],
            baseline=residual_report(reference, fields), joint=residual_report(joint, fields)),
        outside_affected_parity=parity,
        shared_id_seam=dict(baseline_edge_count=len(shared_edges(reference)),
                            joint_edge_count=len(joint_seam),
                            matches_extractor_seam_edges=seam_parity),
        diagnostics=joint['diagnostics'],
    )
    metrics_path = Path(f'/tmp/opencode/topology-dual{suffix}-quality.json')
    payload = json.dumps(metrics, indent=2, allow_nan=False)
    metrics_path.write_text(payload + '\n')
    print(payload, flush=True)
    print(f'Metrics: {metrics_path}', flush=True)
    if not parity['passed'] or not seam_parity:
        raise RuntimeError('Exact noncontact parity or shared-ID seam validation failed')

    plotter = pv.Plotter(shape=(1, 2), window_size=(1800, 900), off_screen=args.off_screen)
    plotter.set_background('#17212b', all_renderers=True)
    panels = [('Existing DC QEF / identical support', reference, metrics['baseline']),
              ('Approximate joint dual / identical support', joint, metrics['joint'])]
    for index, (title, mesh, quality) in enumerate(panels):
        plotter.subplot(0, index)
        plotter.add_text(title, position='upper_left', font_size=15, color='white')
        for surface, (faces, affected) in enumerate(zip(mesh['faces'], mesh['affected_faces'])):
            for mask, color in [(~affected, ['#899ba8', '#b4a99b'][surface]),
                                (affected, ['#51b4ec', '#ffb363'][surface])]:
                selected = faces[mask]
                if not len(selected):
                    continue
                poly = pv.PolyData(mesh['vertices'],
                                   np.column_stack((np.full(len(selected), 3), selected)).ravel())
                plotter.add_mesh(poly, color=color, show_edges=True, edge_color='#34404a',
                                 opacity=1., lighting=False)
        seam_edges = shared_edges(mesh)
        if len(seam_edges):
            seam_poly = pv.PolyData(
                mesh['vertices'],
                lines=np.column_stack((np.full(len(seam_edges), 2), seam_edges)).ravel(),
            )
            plotter.add_mesh(seam_poly, color='white', line_width=6, lighting=False,
                             render_lines_as_tubes=True)
            plotter.add_points(mesh['vertices'][np.unique(seam_edges)], color='white', point_size=8,
                               render_points_as_spheres=True)
        if args.analytic_seam:
            if args.case == 'curved':
                y = np.linspace(0., 1., 101)
                seam = np.column_stack((np.full_like(y, .57), y,
                                        .43 + .25 * (.57 - .5) ** 2 + .15 * (y - .5) ** 2))
            else:
                seam_x = .55 / 1.0625
                seam = np.array([[seam_x, 0., .6 - .25 * seam_x],
                                 [seam_x, 1., .6 - .25 * seam_x]])
            plotter.add_mesh(pv.lines_from_points(seam), color='#32e6ef', line_width=2,
                             lighting=False)
        contact = quality['contact']
        info = (f'{args.case}: {args.resolution}^3 uniform cells; identical Hermite data / retained primal edges\n'
                'Production QUADS 1-3 split; open complete-quad crop\n'
                'Blue/orange: junction-incident faces; gray/tint: regular faces\n'
                f'White: actual shared-ID seam ({len(seam_edges)} edges)\n'
                f'Triangles: {quality["total"]["count"]} total / {contact["count"]} contact / '
                f'{quality["noncontact"]["count"]} noncontact\n'
                f'Contact minimum angle: {contact["minimum_angle_degrees"]:.3f} deg\n'
                f'Outside affected region: exact geometry + connectivity parity = {parity["passed"]}\n'
                'Scope: QEF/junction incidence only, not full compute_model')
        if args.analytic_seam:
            info += '\nCyan: optional analytic guide, not a quality metric'
        plotter.add_text(info, position='lower_left', font_size=10, color='#d0dde8')
        plotter.add_axes()
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.camera.zoom(.8)
    plotter.link_views()
    plotter.show(screenshot=str(args.screenshot))
    print(f'Screenshot: {args.screenshot}', flush=True)


if __name__ == '__main__':
    main()
