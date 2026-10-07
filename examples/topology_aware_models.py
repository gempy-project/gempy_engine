"""Real engine erosion fixtures, uniformly resampled (not adaptive extraction)."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_case(case='complex', resolution=16):
    from gempy_engine.API.model.model_api import compute_model
    from gempy_engine.API.dual_contouring.topology_aware_extraction import extract_topology_aware
    from gempy_engine.core.data.generic_grid import GenericGrid
    from examples.topology_aware_comparison import parity_report, quality_report
    from gempy_engine.core.data.stack_relation_type import StackRelationType as R
    from gempy_engine.config import DUAL_CONTOURING_VERTEX_OVERLAP

    if not 2 <= resolution <= 16:
        raise ValueError('resolution must be between 2 and 16')
    if case == 'unconformity':
        from tests.fixtures.simple_geometries import unconformity
        original = unconformity.__wrapped__()
    elif case == 'complex':
        from tests.fixtures.simple_models import unconformity_complex_factory
        original = unconformity_complex_factory()
    else:
        raise ValueError(f'Unknown case: {case}')
    interpolation_input, options, descriptor = copy.deepcopy(original)
    extent = interpolation_input.grid.octree_grid.orthogonal_extent
    axes = tuple(np.linspace(extent[2*a], extent[2*a+1], resolution + 1) for a in range(3))
    xyz = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    centers = tuple((a[:-1] + a[1:]) / 2 for a in axes)
    center_xyz = np.stack(np.meshgrid(*centers, indexing='ij'), axis=-1)
    node_count = xyz.size // 3
    interpolation_input.grid.custom_grid = GenericGrid(np.concatenate((xyz.reshape(-1, 3), center_xyz.reshape(-1, 3))))
    options.evaluation_options.compute_scalar_gradient = True
    start = time.perf_counter()
    solution = compute_model(interpolation_input, options, descriptor)
    outputs = solution.root_output.outputs
    levels = [np.array(o.scalar_field_at_sp, dtype=float, copy=True) for o in outputs]
    relations = descriptor.stack_structure.masking_descriptor
    stack_samples, stack_gradients, stack_centers, actual_masks = [], [], [], []
    for output in outputs:
        slc = output.grid.custom_grid_slice
        exported = output.exported_fields
        values = np.array(exported.scalar_field[slc], dtype=float, copy=True)
        gradient = np.stack([np.array(g[slc], dtype=float, copy=True) for g in
                             (exported.gx_field, exported.gy_field, exported.gz_field)], axis=-1)
        if len(values) != node_count + resolution**3 or not np.isfinite(gradient).all():
            raise RuntimeError('Invalid custom-grid scalar/gradient export')
        stack_samples.append(values[:node_count].reshape(xyz.shape[:-1]))
        stack_centers.append(values[node_count:].reshape(center_xyz.shape[:-1]))
        stack_gradients.append(gradient[:node_count].reshape(xyz.shape))
        actual_masks.append(np.array(output.squeezed_mask_array[slc][:node_count], dtype=bool).reshape(xyz.shape[:-1]))
    signed = [s - v.min() for s, v in zip(stack_samples, levels)]
    # Mirror _lithology_mask's ERODE/BASEMENT logic with signed inequalities.
    if any(r not in (R.ERODE, R.BASEMENT) for r in relations):
        raise ValueError('This example only adapts erosion piles')
    mask_fields = [s if r is R.ERODE else np.ones_like(s) for s, r in zip(signed, relations)]
    volume_ownership, external_ownership = [], []
    for stack in range(len(outputs)):
        external = [mask_fields[-1]] if stack == 0 else [-s for s in mask_fields[:stack]]
        external_ownership.append(np.minimum.reduce(external))
        volume_ownership.append(np.minimum(mask_fields[stack], external_ownership[-1]))
    mask_parity = [bool(np.array_equal(s > 0, m)) for s, m in zip(volume_ownership, actual_masks)]
    if not all(mask_parity):
        raise RuntimeError('Signed model ownership disagrees with actual squeezed masks')
    groups, indices, samples, gradients, ownership, interior = [], [], [], [], [], []
    for stack, values in enumerate(levels):
        for local, level in enumerate(values):
            groups.append(stack)
            indices.append(local)
            samples.append(stack_samples[stack])
            gradients.append(stack_gradients[stack])
            ownership.append(external_ownership[stack])
            interior.append(stack_centers[stack] - level)
    report = dict(case=case, resolution=resolution, scope='Real kriging; uniform node resampling, not adaptive octree extraction',
                  stack_count=len(outputs), surface_count=len(groups), isovalues=[v.tolist() for v in levels],
                  mask_parity=mask_parity, ownership='Signed erosion constraints; self-stack threshold excluded only for interface extraction',
                  production_triangles=[len(m.edges) for m in solution.dc_meshes],
                  production_overlap=DUAL_CONTOURING_VERTEX_OVERLAP.name,
                  compute_seconds=time.perf_counter()-start)
    data = dict(axes=axes, scalar_samples=np.array(samples), surface_to_stack=groups, surface_indices=indices,
                stack_relations=relations, isovalues=levels, ownership=np.array(ownership),
                gradient_samples=np.array(gradients), interior_samples=np.array(interior))
    try:
        joint = extract_topology_aware(**data, include_reference=True, sampled_junctions=True)
    except ValueError as error:
        joint = None
        report.update(status='rejected', rejection=str(error))
    else:
        reference = joint['reference']
        parity = parity_report(joint, reference)
        incidence = []
        for a, b in joint['seam_edges']:
            incidence.append([sum(int(a in face and b in face) for face in faces) for faces in joint['faces']])
        edges = Counter(edge for faces in joint['faces'] for edge in
                        {tuple(sorted((int(a), int(b)))) for face in faces for a, b in zip(face, np.roll(face, -1))})
        actual_seams = {edge for edge, count in edges.items() if count > 1}
        reference_edges = Counter(edge for faces in reference['faces'] for edge in
                                  {tuple(sorted((int(a), int(b)))) for face in faces for a, b in zip(face, np.roll(face, -1))})
        if actual_seams != set(map(tuple, np.sort(joint['seam_edges'], axis=1))):
            raise RuntimeError('Actual shared mesh edges disagree with planned seams')
        defects = []
        for faces in joint['faces']:
            points = joint['vertices'][faces]
            area = np.linalg.norm(np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]), axis=1)
            defects.append(dict(degenerate_triangles=int(np.sum(area < 1e-12)),
                                duplicate_triangles=len(faces)-len(np.unique(np.sort(faces, axis=1), axis=0))))
        report.update(status='supported', triangle_counts=[len(f) for f in joint['faces']],
                      reference_triangle_counts=[len(f) for f in reference['faces']],
                      shared_seam_edges=len(joint['seam_edges']), seam_incidence=incidence,
                      actual_shared_edges_match=True, reference_shared_seam_edges=sum(c > 1 for c in reference_edges.values()), defects=defects,
                      noncontact_parity=parity, quality=quality_report(joint),
                      reference_quality=quality_report(reference), diagnostics=joint['diagnostics'])
        if not parity['passed'] or any(any(d.values()) for d in defects):
            raise RuntimeError('Noncontact parity or triangle validation failed')
    return report, joint, solution, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['unconformity', 'complex'], default='complex')
    parser.add_argument('--resolution', type=int, default=16)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--off-screen', action='store_true')
    parser.add_argument('--screenshot', type=Path)
    args = parser.parse_args()
    screenshot = args.screenshot or Path(f'/tmp/opencode/topology-real-{args.case}.png')
    if not args.check_only and not screenshot.parent.is_dir():
        parser.error('Screenshot parent must exist')
    report, joint, solution, data = run_case(args.case, args.resolution)
    payload = json.dumps(report, indent=2, allow_nan=False)
    print(payload, flush=True)
    if args.check_only:
        return
    screenshot.with_suffix('.json').write_text(payload + '\n')
    import pyvista as pv
    plotter = pv.Plotter(shape=(1, 2), off_screen=args.off_screen, window_size=(1800, 900),
                         title=f'Topology-Aware DC | Real {args.case} Model')
    colors = ['#56b4e9', '#e69f00', '#009e73', '#cc79a7']
    for panel in range(2):
        plotter.subplot(0, panel)
        plotter.set_background('#17212b')
        if joint is not None:
            mesh = joint['reference'] if panel == 0 else joint
            title = 'Independent same-data production QEF' if panel == 0 else 'Joint sampled-face DC'
            for i, faces in enumerate(mesh['faces']):
                poly = pv.PolyData(mesh['vertices'], np.column_stack((np.full(len(faces), 3), faces)).ravel())
                plotter.add_mesh(poly, color=colors[i], show_edges=True, lighting=False)
            if panel and len(joint['seam_edges']):
                seam = pv.PolyData(joint['vertices'], lines=np.column_stack((np.full(len(joint['seam_edges']), 2), joint['seam_edges'])).ravel())
                plotter.add_mesh(seam, color='white', line_width=5)
        else:
            title = 'Actual production DC (not joint fallback)' if panel == 0 else 'Actual stack-0 scalar nodes (no extraction)'
            if panel == 0:
                for i, mesh in enumerate(solution.dc_meshes):
                    if not len(mesh.edges):
                        continue
                    poly = pv.PolyData(np.asarray(mesh.vertices), np.column_stack((np.full(len(mesh.edges), 3), mesh.edges)).ravel())
                    plotter.add_mesh(poly, color=colors[i], show_edges=True)
            else:
                xyz = np.stack(np.meshgrid(*data['axes'], indexing='ij'), axis=-1).reshape(-1, 3)
                plotter.add_points(xyz, scalars=data['scalar_samples'][0].ravel(), point_size=7)
        plotter.add_text(title, font_size=14, color='white')
        plotter.add_text(f'{args.case}: {report["stack_count"]} stacks / {report["surface_count"]} surfaces\n'
                         f'Uniform resampling {args.resolution}^3; masks verified\n'
                         + (f'Triangles: {report["triangle_counts"]}; shared edges: {report["shared_seam_edges"] if panel else report["reference_shared_seam_edges"]}\n'
                            'White: actual shared seam; exact noncontact parity' if joint is not None
                            else f'Production triangles: {report["production_triangles"]}\nJoint REJECTED: {report["rejection"]}'),
                         position='lower_left', font_size=10, color='white')
        plotter.add_axes()
        plotter.view_isometric()
        plotter.reset_camera()
    plotter.link_views()
    print(f'READY: {args.case} comparison; topology extraction {report["status"]}.', flush=True)
    plotter.show(screenshot=str(screenshot))
    print(f'Screenshot: {screenshot}')


if __name__ == '__main__':
    main()
