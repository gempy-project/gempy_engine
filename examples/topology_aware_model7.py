"""Actual g07_combination fault + unconformity, with explicit bank-conditioned queries."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_case(resolution=14):
    from gempy_engine.config import AvailableBackends
    from gempy_engine.core.backend_tensor import BackendTensor
    from gempy_engine.core.data.generic_grid import GenericGrid
    from gempy_engine.API.dual_contouring.model7_bank_experiment import (
        capture_model7, query_bank, extract_bank_skin,
    )
    from gempy_engine.modules.dual_contouring.topology_extraction import CORNERS, sampled_face_segment
    from examples.topology_aware_comparison import parity_report, quality_report
    from tests.fixtures.model7_combination import model7_combination_factory

    if not 2 <= resolution <= 16:
        raise ValueError('resolution must be between 2 and 16 (bounded uniform experiment)')
    BackendTensor._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype='float64')
    print(f'Active backend: {BackendTensor.engine_backend.name}; CPU; {BackendTensor.dtype}; KeOps={BackendTensor.use_pykeops}', flush=True)
    inputs, options, descriptor = model7_combination_factory(mesh_extraction=True)
    original_points = inputs.surface_points.sp_coords.copy()
    original_orientations = inputs.orientations.dip_positions.copy()
    extent = inputs.grid.octree_grid.orthogonal_extent
    axes = tuple(np.linspace(extent[2*a], extent[2*a+1], resolution + 1) for a in range(3))
    xyz = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    center_axes = tuple((a[:-1] + a[1:]) / 2 for a in axes)
    centers = np.stack(np.meshgrid(*center_axes, indexing='ij'), axis=-1)
    fault_observations = original_points[:3]
    normal = inputs.orientations.dip_gradients[0].copy()
    normal /= np.linalg.norm(normal)
    probes = fault_observations.mean(axis=0) + np.linspace(-1e-5, 1e-5, 101)[:, None]*normal
    queries = np.concatenate((xyz.reshape(-1, 3), centers.reshape(-1, 3), probes))
    node_count = xyz.size // 3
    inputs.grid.custom_grid = GenericGrid(queries)
    options.evaluation_options.compute_scalar_gradient = True
    solution, captures = capture_model7(inputs, options, descriptor)
    outputs = solution.root_output.outputs
    levels = [np.array(o.scalar_field_at_sp, copy=True) for o in outputs]
    fault_values, fault_gradients = query_bank(captures[0], queries)
    separator = fault_values - levels[0][0]
    banks, sampling_metrics, topology_metrics, meshes, rejections, validation = [], [], [], [], [], []
    node_indices = np.indices((resolution,)*3).reshape(3, -1).T[:, None, :] + CORNERS
    index = tuple(node_indices[:, :, a] for a in range(3))
    for bank in (0, 1):
        values, gradients = [fault_values], [fault_gradients]
        parity = []
        for stack in (1, 2):
            scalar, gradient = query_bank(captures[stack], queries, bank)
            values.append(scalar)
            gradients.append(gradient)
            output = outputs[stack]
            slc = output.grid.custom_grid_slice
            drift = captures[stack][0].fault_internal.fault_values_everywhere[0, slc]
            if not np.array_equal(drift[:node_count] > .5, separator[:node_count] < 0):
                raise RuntimeError('Geometric bank sign disagrees with production fault drift')
            expected_drift = 1/(1+np.exp(np.clip(options.sigmoid_slope*separator, -50, 50)))
            segmentation_error = float(np.max(np.abs(expected_drift-drift)))
            if segmentation_error > 1e-12:
                raise RuntimeError('Shifted production drift disagrees with smooth descending sigmoid')
            actual = np.array(output.exported_fields.scalar_field[slc])
            actual_gradient = np.stack([g[slc] for g in (output.exported_fields.gx_field,
                                                       output.exported_fields.gy_field, output.exported_fields.gz_field)], axis=-1)
            far = np.abs(drift - bank) < 1e-12
            if not far.any():
                raise RuntimeError('No saturated production drift samples for bank parity')
            parity.append(dict(stack=stack, saturated_queries=int(far.sum()),
                               scalar_max_error=float(np.max(np.abs(scalar[far]-actual[far]))),
                               gradient_max_error=float(np.max(np.abs(gradient[far]-actual_gradient[far]))),
                               drift_min=float(drift.min()), drift_max=float(drift.max()),
                               shifted_sigmoid_max_error=segmentation_error,
                               smooth_transition_queries=int(np.sum((drift > 1e-12) & (drift < 1-1e-12)))))
        samples = np.array([v[:node_count].reshape(xyz.shape[:-1]) for v in values])
        normals = np.array([g[:node_count].reshape(xyz.shape) for g in gradients])
        center_samples = np.array([v[node_count:node_count+resolution**3].reshape((resolution,)*3) for v in values])
        banks.append(dict(samples=samples, gradients=normals))
        sampling_metrics.append(parity)
        fields = np.array([samples[0]-levels[0][0], samples[1]-levels[1][0],
                           samples[2]-levels[2][0], samples[2]-levels[2][1]])
        corners = fields[(slice(None), *index)]
        crossed = (corners.min(axis=2) < 0) & (corners.max(axis=2) > 0)
        contacts = []
        for surface in (1, 2, 3):
            candidate = np.flatnonzero(crossed[0] & crossed[surface])
            segments, errors = 0, {}
            for cell in candidate:
                try:
                    segment = sampled_face_segment(corners[[0, surface], cell])
                except ValueError as error:
                    errors[str(error)] = errors.get(str(error), 0) + 1
                else:
                    segments += int(segment is not None)
            contacts.append(dict(surface=surface, candidate_cells=len(candidate),
                                 isolated_shared_face_segments=segments, rejected_face_evidence=errors))
        topology_metrics.append(dict(bank=bank, fault_horizon_contacts=contacts,
                                     cells_with_multiple_horizons_on_fault=int(np.sum(crossed[0] & (crossed[1:].sum(axis=0) > 1))),
                                     fault_erosion_older_competing_cells=int(np.sum(crossed[0] & crossed[1] & crossed[2:].any(axis=0)))))
        try:
            mesh = extract_bank_skin(axes, samples, normals, levels, bank, center_samples=center_samples)
        except ValueError as error:
            meshes.append(None)
            rejections.append(str(error))
            validation.append(None)
        else:
            meshes.append(mesh)
            rejections.append(None)
            incidence = []
            for a, b in mesh['seam_edges']:
                incidence.append([sum(int(a in face and b in face) for face in faces) for faces in mesh['faces']])
            defects, bank_violations, winding, erosion_residuals = [], [], [], []
            actual_shared = Counter(edge for faces in mesh['faces'] for edge in
                                    {tuple(sorted((int(a), int(b)))) for face in faces
                                     for a, b in zip(face, np.roll(face, -1))})
            if {e for e, count in actual_shared.items() if count > 1} != set(map(tuple, np.sort(mesh['seam_edges'], axis=1))):
                raise RuntimeError('Actual shared surface edges disagree with planned seam incidence')
            for surface, faces in enumerate(mesh['faces']):
                positions = mesh['vertices'][faces]
                cross = np.cross(positions[:, 1]-positions[:, 0], positions[:, 2]-positions[:, 0])
                defects.append(dict(degenerate=int(np.sum(np.linalg.norm(cross, axis=1) < 1e-12)),
                                    duplicate=len(faces)-len(np.unique(np.sort(faces, axis=1), axis=0))))
                used = np.unique(faces)
                actual_fault, _ = query_bank(captures[0], mesh['vertices'][used])
                signed_bank = (-1 if bank == 0 else 1) * (actual_fault-levels[0][0])
                bank_violations.append(0 if surface == 0 else int(np.sum(signed_bank > 1e-10)))
                if surface >= 2:
                    actual_erosion, _ = query_bank(captures[1], mesh['vertices'][used], bank)
                    erosion_residuals.append(float(np.max(actual_erosion-levels[1][0])))
                stack = [0, 1, 2, 2][surface]
                _, actual_normals = query_bank(captures[stack], positions.mean(axis=1), None if stack == 0 else bank)
                if surface == 0 and bank == 0:
                    actual_normals = -actual_normals
                winding.append(int(np.sum(np.sum(cross*actual_normals, axis=1) <= 0)))
            parity = parity_report(mesh, mesh['reference'])
            validation.append(dict(defects=defects, bank_side_violations=bank_violations,
                                   actual_engine_winding_violations=winding, seam_incidence=incidence,
                                   older_surface_max_erosion_residual=erosion_residuals,
                                   actual_erosion_residual_tolerance=1e-8,
                                   actual_shared_edges_match=True,
                                   noncontact_parity=parity, quality=quality_report(mesh),
                                   reference_quality=quality_report(mesh['reference']),
                                   diagnostics=mesh['diagnostics'],
                                   reference_triangle_counts=[len(f) for f in mesh['reference']['faces']]))
            if any(any(d.values()) for d in defects) or any(bank_violations) or any(winding) or \
                    max(erosion_residuals) > 1e-8 or not parity['passed']:
                raise RuntimeError(f'Bank {bank} validation failed: defects={defects}, bank={bank_violations}, '
                                   f'winding={winding}, erosion={erosion_residuals}, parity={parity["passed"]}')
    if not np.array_equal(original_points, inputs.surface_points.sp_coords) or \
            not np.array_equal(original_orientations, inputs.orientations.dip_positions):
        raise RuntimeError('Original observations changed')
    shifts = [float(np.max(np.abs(banks[0]['samples'][s]-banks[1]['samples'][s]))) for s in (1, 2)]
    report = dict(source='https://docs.gempy.org/examples/geometries/g07_combination.html',
                  status='supported' if all(m is not None for m in meshes) else 'unsupported_actual_model7',
                  backend='numpy CPU float64', resolution=resolution, world_extent=[0, 2500, 0, 1000, 0, 1000],
                  observations=102, orientations=8, original_observations_preserved=True,
                  stacks=['FAULT', 'ERODE', 'BASEMENT'], fault_relations=[[0, 1], [0, 2]],
                  bank_signs={'0': 'positive fault scalar; shifted drift 0', '1': 'negative fault scalar; shifted drift 1'},
                  fault_segmentation='shifted descending sigmoid; same reference-point drift as production',
                  surface_count=4, bank_surface_count=8, isovalues=[v.tolist() for v in levels],
                  production_triangles=[len(m.edges) for m in solution.dc_meshes],
                  bank_sampling=sampling_metrics, dependent_bank_max_difference=shifts,
                  topology_evidence=topology_metrics, bank_rejections=rejections,
                  joint_triangle_counts=[None if m is None else [len(f) for f in m['faces']] for m in meshes],
                  joint_seam_counts=[None if m is None else len(m['seam_edges']) for m in meshes],
                  validation=validation,
                  scope='uniform samples; bank-side fault skins, NOT a single conforming fault mesh',
                  unchanged_solved_weights=True, unchanged_reference_fault_drifts=True,
                  query_coordinate_displacement=0, post_repair=False)
    data = dict(axes=axes, xyz=xyz, separator=separator[:node_count].reshape(xyz.shape[:-1]),
                banks=banks, captures=captures, levels=levels)
    return report, meshes, solution, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution', type=int, default=14)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--off-screen', action='store_true')
    parser.add_argument('--screenshot', type=Path, default=Path('/tmp/opencode/topology-model7.png'))
    args = parser.parse_args()
    if not args.check_only and not args.screenshot.parent.is_dir():
        parser.error('Screenshot parent must already exist')
    report, meshes, solution, data = run_case(args.resolution)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if args.check_only:
        return
    args.screenshot.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    import pyvista as pv
    plotter = pv.Plotter(shape=(1, 2), off_screen=args.off_screen, window_size=(1800, 900),
                         title='Topology-Aware DC | Model 7 Fault Banks')
    colors = ['#cf68d9', '#56b4e9', '#e69f00', '#009e73']
    supported = report['status'] == 'supported'
    for panel in (0, 1):
        plotter.subplot(0, panel)
        plotter.set_background('#17212b')
        if supported:
            for bank, joint in enumerate(meshes):
                mesh = joint['reference'] if panel == 0 else joint
                for surface, faces in enumerate(mesh['faces']):
                    poly = pv.PolyData(mesh['vertices'], np.column_stack((np.full(len(faces), 3), faces)).ravel())
                    plotter.add_mesh(poly, color=colors[surface], opacity=.45 if surface == 0 else 1.,
                                     show_edges=bool(bank), lighting=False)
                if panel and len(joint['seam_edges']):
                    seam = pv.PolyData(joint['vertices'], lines=np.column_stack((np.full(len(joint['seam_edges']), 2), joint['seam_edges'])).ravel())
                    plotter.add_mesh(seam, color='white', line_width=5)
            title = 'Independent same-bank QEF' if panel == 0 else 'Joint bank-side fault skins (not single fault mesh)'
            detail = 'Solid bank 0 / edged bank 1; white actual shared seams; purple separator'
        elif panel == 0:
            for i, mesh in enumerate(solution.dc_meshes):
                if not len(mesh.edges):
                    continue
                poly = pv.PolyData(np.asarray(mesh.vertices), np.column_stack((np.full(len(mesh.edges), 3), mesh.edges)).ravel())
                plotter.add_mesh(poly, color=colors[i], opacity=.45 if i == 0 else 1., show_edges=True)
            title = 'Actual production faulted Model7 (not joint fallback)'
            detail = f'Original 102 points / 8 orientations; triangles {report["production_triangles"]}'
        else:
            points = data['xyz'].reshape(-1, 3)
            bank = (data['separator'].ravel() < 0).astype(int)
            plotter.add_points(points, scalars=bank, cmap=['#56b4e9', '#e69f00'], point_size=4,
                               render_points_as_spheres=True, show_scalar_bar=False)
            title = 'Joint UNSUPPORTED; geometric bank sample indicators'
            detail = '\n'.join(f'Bank {b}: {error}' for b, error in enumerate(report['bank_rejections']))
        plotter.add_text(title, font_size=13, color='white')
        plotter.add_text(detail, position='lower_left', font_size=10, color='white')
        plotter.add_axes()
        plotter.view_isometric()
        plotter.reset_camera()
    plotter.link_views()
    print(f'READY: Model 7 comparison; bank extraction {report["status"]}.', flush=True)
    plotter.show(screenshot=str(args.screenshot))
    print(f'Screenshot: {args.screenshot}', flush=True)


if __name__ == '__main__':
    main()
