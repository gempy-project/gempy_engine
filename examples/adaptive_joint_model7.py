"""Real Model 7 through production compute_model joint extraction.

Run directly in PyCharm for an interactive viewer, or use --check-only.
The default root 9 / depth 2 / min-level 0 uses verified natural mixed leaves.
Root 7 / depth 2 / min-level 2 is a verified fully refined baseline; min-level
1 also fully refines, while min-level 0 rejects with unsupported_hanging_branch.
No uniform fallback or mesh repair is attempted. Mixed leaves do not imply
coarse/fine CONTACT seams: the default has zero such seam edges.
--mode joint_contacts uses joint erosion/onlap contacts with pretty's fault
vertex merge (one fault mesh, horizons ending on fault vertices, no banks).
--compare shows ordinary smooth-fault none/pretty beside bank-conditioned joint
skins and joint_contacts. These are different field semantics, not a
like-for-like benchmark.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _host(values):
    """NumPy copy of a NumPy array or a (possibly CUDA) Torch tensor."""
    return values.detach().cpu().numpy() if hasattr(values, 'detach') else np.asarray(values)


def make_model(root_resolution, depth, min_level, mode):
    from gempy_engine.core.data.regular_grid import RegularGrid
    from tests.fixtures.model7_combination import model7_combination_factory

    inputs, options, descriptor = model7_combination_factory(
        number_octree_levels=depth, mesh_extraction=True)
    extent = _host(inputs.grid.octree_grid.orthogonal_extent).copy()
    inputs.grid.octree_grid = RegularGrid(extent, [root_resolution] * 3)
    evaluation = options.evaluation_options
    evaluation.number_octree_levels = depth
    evaluation.number_octree_levels_surface = depth
    evaluation.octree_min_level = min_level
    evaluation.mesh_extraction_overlap = mode
    return inputs, options, descriptor


def hierarchy_report(solution):
    levels = solution.octrees_output
    root_shape = np.asarray(levels[0].grid.octree_grid.regular_grid_shape, dtype=int)
    spans = Counter()
    counts = []
    for i, level in enumerate(levels):
        count = len(level.grid.octree_grid.values)
        counts.append(count)
        refined = (int(np.count_nonzero(levels[i + 1].grid.octree_grid.active_cells))
                   if i + 1 < len(levels) else 0)
        spans[2 ** (len(levels) - 1 - i)] += count - refined
    spans = {str(span): count for span, count in sorted(spans.items()) if count}
    return dict(root_shape=root_shape.tolist(),
                finest_lattice_shape=(root_shape * 2 ** (len(levels) - 1)).tolist(),
                level_cell_counts=counts, leaf_span_counts=spans,
                mixed_leaves=len(spans) > 1,
                covered_finest_cells=sum(int(s) ** 3 * n for s, n in spans.items()))


def joint_seams(meshes):
    # Seam endpoints are GLOBAL IDs, not indices into any one surface mesh.
    positions = {}
    banks = {}
    incidence = []
    for mesh in meshes:
        bank_ids = (np.full(len(mesh.vertices), -1) if mesh.joint_bank_ids is None
                    else mesh.joint_bank_ids)  # joint_contacts has no banks
        for gid, vertex, bank in zip(mesh.joint_vertex_ids, mesh.vertices, bank_ids):
            gid = int(gid)
            if gid in positions and not np.array_equal(positions[gid], vertex):
                raise ValueError(f'Inconsistent shared vertex {gid}')
            positions[gid] = vertex
            if banks.setdefault(gid, int(bank)) != int(bank):
                raise ValueError('Cross-bank vertex ID join')
        incidence.append(Counter(tuple(sorted((int(a), int(b))))
                                 for face in mesh.joint_vertex_ids[mesh.edges]
                                 for a, b in zip(face, np.roll(face, -1))))
    edges = np.asarray(meshes[0].joint_seam_edges, dtype=np.int64).reshape(-1, 2)
    for mesh in meshes[1:]:
        if not np.array_equal(edges, mesh.joint_seam_edges):
            raise ValueError('Exported seam lists differ')
    for a, b in edges:
        edge = tuple(sorted((int(a), int(b))))
        if a not in positions or b not in positions or banks[int(a)] != banks[int(b)]:
            raise ValueError('Unresolved or cross-bank seam')
        if sorted(row[edge] for row in incidence if row[edge]) != [1, 2]:
            raise ValueError(f'Invalid actual seam incidence: {edge}')
    return positions, edges, dict(Counter(str(banks[int(a)]) for a, _ in edges))


def run_case(root_resolution=9, depth=2, min_level=0, mode='joint'):
    from gempy_engine.API.model.model_api import compute_model
    from tests.fixtures.model7_combination import MODEL7_SURFACES

    inputs, options, descriptor = make_model(root_resolution, depth, min_level, mode)
    points = inputs.surface_points.sp_coords.copy()
    orientations = inputs.orientations.dip_positions.copy()
    gradients = inputs.orientations.dip_gradients.copy()
    print(f'Computing {mode}: root={root_resolution}, depth={depth}, min-level={min_level}', flush=True)
    solution = compute_model(inputs, options, descriptor)
    for before, after in ((points, inputs.surface_points.sp_coords),
                          (orientations, inputs.orientations.dip_positions),
                          (gradients, inputs.orientations.dip_gradients)):
        np.testing.assert_array_equal(before, after)
    meshes = solution.dc_meshes
    if len(points) != 102 or len(orientations) != 8 or len(meshes) != 4:
        raise ValueError('Original Model 7 observation/export counts changed')
    surfaces = []
    for name, mesh in zip(MODEL7_SURFACES, meshes):
        if not np.isfinite(mesh.vertices).all():
            raise ValueError('Nonfinite mesh vertices')
        surfaces.append(dict(name=name, stack=mesh.stack_index, surface=mesh.surface_index,
                             exported_surface=mesh.exported_surface_index, isovalue=mesh.isovalue,
                             vertices=len(mesh.vertices), triangles=len(mesh.edges),
                             triangle_banks=(dict(Counter(map(str, mesh.joint_face_bank_ids)))
                                             if mode == 'joint' else None)))
    report = dict(status='supported', mode=mode, backend='NumPy CPU float64; no KeOps',
                  source='tests.fixtures.model7_combination.model7_combination_factory',
                  source_points=len(points), source_orientations=len(orientations),
                  observations_preserved=True, world_extent=[0, 2500, 0, 1000, 0, 1000],
                  engine_extent=np.asarray(inputs.grid.octree_grid.orthogonal_extent).tolist(),
                  transform='(world_xyz - [1150, 500, 525]) / 4600 (original factory)',
                  root_resolution=root_resolution, depth=depth, min_level=min_level,
                  hierarchy=hierarchy_report(solution), surfaces=surfaces)
    if mode == 'joint':
        _, edges, seam_banks = joint_seams(meshes)
        contact = meshes[0].contact_report
        report.update(interface=contact['interface'], seam_edges=len(edges), seam_banks=seam_banks,
                      coarsefine_seam_edge_count=int(contact['coarsefine_seam_edge_count']),
                      bank_labels={'0': 'positive fault scalar / fixed drift 0',
                                   '1': 'negative fault scalar / fixed drift 1'},
                      no_uniform_fallback=contact['no_uniform_fallback'],
                      no_post_reconciliation=contact['no_post_reconciliation'],
                      cross_bank_id_joins=contact['cross_bank_id_joins'])
    elif mode == 'joint_contacts':
        _, edges, _ = joint_seams(meshes)
        contact = meshes[0].contact_report
        report.update(interface=contact['interface'], seam_edges=len(edges),
                      **{key: contact[key] for key in (
                          'fault_cell_count', 'borrowed_vertex_count', 'filtered_fault_triangle_count',
                          'weighted_fault_vertex_count', 'fault_overridden_junction_cells',
                          'fallback_region_leaf_count', 'coarsefine_seam_edge_count')})
    else:
        report['interface'] = 'ordinary production smooth-fault fields'
    return report, solution


def render(cases, off_screen=False, screenshot=None):
    import pyvista as pv

    plotter = pv.Plotter(shape=(1, len(cases)), off_screen=off_screen,
                         window_size=(700 * len(cases), 800), title='Production Model 7')
    colors = ['#cf68d9', '#56b4e9', '#e69f00', '#009e73']
    for panel, (report, solution) in enumerate(cases):
        plotter.subplot(0, panel)
        plotter.set_background('#17212b')
        joint = report['mode'] == 'joint'
        for surface, mesh in enumerate(solution.dc_meshes):
            for bank in ((0, 1) if joint else (None,)):
                faces = mesh.edges[mesh.joint_face_bank_ids == bank] if joint else mesh.edges
                if not len(faces):
                    continue
                poly = pv.PolyData(mesh.vertices, np.column_stack((np.full(len(faces), 3), faces)).ravel())
                plotter.add_mesh(poly, color=colors[surface], opacity=.4 if surface == 0 else 1.,
                                 show_edges=bank == 1, lighting=False)
        if report['mode'] in ('joint', 'joint_contacts'):
            positions, edges, _ = joint_seams(solution.dc_meshes)
            if len(edges):
                xyz = np.array([positions[int(gid)] for edge in edges for gid in edge])
                lines = np.column_stack((np.full(len(edges), 2), np.arange(len(xyz)).reshape(-1, 2)))
                plotter.add_mesh(pv.PolyData(xyz, lines=lines.ravel()), color='white', line_width=4)
        title = ('Joint: bank-conditioned, two fault skins' if joint else
                 'joint_contacts: joint contacts, one fault mesh' if report['mode'] == 'joint_contacts' else
                 f'{report["mode"]}: ordinary smooth-fault fields')
        plotter.add_text(title, font_size=12, color='white')
        plotter.add_text('Solid bank 0 / edged bank 1; white shared seams' if joint else
                         'Horizons end on fault vertices; white shared seams'
                         if report['mode'] == 'joint_contacts' else
                         'Different field semantics; not a like-for-like benchmark',
                         position='lower_left', font_size=10, color='white')
        plotter.add_axes()
        plotter.view_isometric()
        plotter.reset_camera()
    plotter.link_views()
    plotter.show(screenshot=None if screenshot is None else str(screenshot))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-resolution', type=int, default=9)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--min-level', type=int, default=0)
    parser.add_argument('--check-only', action='store_true', help='validate without importing PyVista')
    parser.add_argument('--mode', choices=('joint', 'joint_contacts'), default='joint')
    parser.add_argument('--compare', action='store_true', help='four production modes, fresh models')
    parser.add_argument('--off-screen', action='store_true')
    parser.add_argument('--screenshot', type=Path)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if args.root_resolution < 2 or args.depth < 2 or args.min_level < 0:
        parser.error('root-resolution >= 2, depth >= 2 and min-level >= 0 required')
    for path in (args.json, args.screenshot):
        if path is not None and not path.parent.is_dir():
            parser.error(f'Output parent must already exist: {path.parent}')

    from gempy_engine.config import AvailableBackends
    from gempy_engine.core.backend_tensor import BackendTensor
    BackendTensor._change_backend(AvailableBackends.numpy, use_gpu=False,
                                  use_pykeops=False, dtype='float64', grads=False)
    print('Backend: NumPy CPU float64; KeOps disabled', flush=True)
    cases = []
    try:
        # Joint first: a rejection must never become an ordinary-mode fallback.
        cases.append(run_case(args.root_resolution, args.depth, args.min_level, args.mode))
        if args.compare:
            other = 'joint_contacts' if args.mode == 'joint' else 'joint'
            cases = [run_case(args.root_resolution, args.depth, args.min_level, mode)
                     for mode in ('none', 'pretty')] + cases + [
                        run_case(args.root_resolution, args.depth, args.min_level, other)]
    except (ValueError, NotImplementedError) as error:
        report = dict(status='rejected', error=str(error), root_shape=[args.root_resolution] * 3,
                      finest_lattice_shape=[args.root_resolution * 2 ** (args.depth - 1)] * 3,
                      depth=args.depth, min_level=args.min_level, no_uniform_fallback=True,
                      leaf_span_counts=None, coarsefine_seam_edge_count=None,
                      diagnostic='Extraction rejected; no solution returned. Leaf/seam counts unavailable.')
        print(f'REJECTED: {error}', file=sys.stderr, flush=True)
        exit_code = 1
    else:
        report = cases[0][0] if not args.compare else dict(
            comparison='different field semantics, not a like-for-like benchmark',
            cases=[r for r, _ in cases])
        exit_code = 0
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    print(text, flush=True)
    if args.json is not None:
        args.json.write_text(text)
    if exit_code == 0 and not args.check_only:
        render(cases, args.off_screen, args.screenshot)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
