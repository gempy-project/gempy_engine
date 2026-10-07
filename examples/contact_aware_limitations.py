"""Manual production-path limitations gallery; CUDA compute is opt-in via --gpu."""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['DUAL_CONTOURING_VERTEX_OVERLAP'] = 'contact_aware'
os.environ['DEFAULT_BACKEND'] = 'numpy'
os.environ['DEFAULT_PYKEOPS'] = 'False'
os.environ['DUAL_CONTOURING_MULTITHREAD'] = 'False'

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.options.evaluation_options import MeshExtractionMaskingOptions
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine.core.data.stacks_structure import StacksStructure


def analytic_model(normals, levels, relations, resolution, devices):
    functions = []
    for normal, values in zip(normals, levels):
        normal = BackendTensor.t.array(normal, dtype=BackendTensor.dtype)

        def field(xyz, n=normal):
            devices.add(str(xyz.device) if hasattr(xyz, 'device') else 'cpu')
            return xyz @ n

        functions.append(CustomInterpolationFunctions(
            scalar_field_at_surface_points=BackendTensor.t.array(values, dtype=BackendTensor.dtype),
            implicit_function=field,
            gx_function=lambda xyz, n=normal: xyz[:, 0] * 0 + n[0],
            gy_function=lambda xyz, n=normal: xyz[:, 0] * 0 + n[1],
            gz_function=lambda xyz, n=normal: xyz[:, 0] * 0 + n[2],
        ))
    n = len(functions)
    descriptor = InputDataDescriptor(
        TensorsStructure(np.array([], dtype=int)),
        StacksStructure(np.zeros(n, dtype=int), np.zeros(n, dtype=int),
                        np.array([len(v) for v in levels]), relations,
                        faults_relations=np.zeros((n, n), dtype=bool),
                        interp_functions_per_stack=functions),
    )
    root, depth = (16, 3) if resolution == 64 else (resolution // 2, 2)
    inputs = InterpolationInput(
        SurfacePoints(np.empty((0, 3))), Orientations(np.empty((0, 3)), np.empty((0, 3))),
        EngineGrid(octree_grid=RegularGrid(np.array([0., 1., 0., 1., 0., 1.]), [root] * 3)),
    )
    options = InterpolationOptions.from_args(10., 1.)
    options.evaluation_options.number_octree_levels = depth
    options.evaluation_options.number_octree_levels_surface = depth
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
    return compute_model(inputs, options, descriptor).dc_meshes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='Use PyTorch CUDA, without PyKeOps')
    parser.add_argument('--off-screen', action='store_true')
    parser.add_argument('--screenshot')
    args = parser.parse_args()
    device = 'NumPy CPU'
    if args.gpu:
        import torch
        if not torch.cuda.is_available():
            parser.error('--gpu requested but CUDA is unavailable; no silent CPU fallback')
        device = torch.cuda.get_device_name(0)
    BackendTensor._change_backend(AvailableBackends.PYTORCH if args.gpu else AvailableBackends.numpy,
                                  use_gpu=args.gpu, use_pykeops=False, dtype='float64', grads=False)
    devices = set()
    cases = [
        ('1. Interfaces are NOT closed solids', [(0., 0., 1.), (1., 0., .3)],
         [[.53], [.62]], [R.ERODE, R.BASEMENT], 8, 'open'),
        ('2. Coarse onlap: approximate seam', [(1., 0., -.25), (.25, 0., 1.)],
         [[.4], [.6]], [R.ONLAP, R.BASEMENT], 8, 'onlap'),
        ('3. Fine onlap: still not exact', [(1., 0., -.25), (.25, 0., 1.)],
         [[.4], [.6]], [R.ONLAP, R.BASEMENT], 64, 'onlap'),
        ('4. Coarse isolation: competing contacts', [(0., 0., 1.), (1., 0., .2)],
         [[.43, .46], [.59]], [R.ERODE, R.BASEMENT], 8, 'isolation'),
        ('5. Fine isolation: separated horizons', [(0., 0., 1.), (1., 0., .2)],
         [[.43, .46], [.59]], [R.ERODE, R.BASEMENT], 64, 'isolation'),
    ]
    results = []
    for title, normals, levels, relations, resolution, kind in cases:
        print(f'Computing {title} ({resolution} cells/axis) on {device}', flush=True)
        meshes = analytic_model(normals, levels, relations, resolution, devices)
        results.append((title, meshes, kind, resolution))

    from tests.fixtures.complex_geometries import one_fault_model
    inputs, descriptor, options = one_fault_model.__wrapped__()
    options.evaluation_options.number_octree_levels = 3
    options.evaluation_options.number_octree_levels_surface = 3
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
    print(f'Computing real kriging fault model on {device}', flush=True)
    fault_meshes = compute_model(inputs, options, descriptor).dc_meshes
    if args.gpu:
        assert devices == {'cuda:0'}, f'Analytic field evaluated on unexpected devices: {devices}'
        assert all(m.vertices_tensor.is_cuda for _, meshes, _, _ in results for m in meshes)
        assert all(m.vertices_tensor.is_cuda for m in fault_meshes)
    results.append(('6. Fault smoke: finite tips unverified', fault_meshes, 'fault', 0))
    print(f'Field evaluation devices: {sorted(devices)}; mesh/QEF source tensors verified', flush=True)

    plotter = pv.Plotter(shape=(2, 3), window_size=(1920, 1120), off_screen=args.off_screen,
                         title=f'GemPy contact_aware | Limitations | {device}')
    plotter.set_background('#17212b', all_renderers=True)
    colors = ['#51b4ec', '#ffb363', '#76d5a0', '#cf9eec', '#f1d66e']
    overlays, actors = [], []
    for index, (title, meshes, kind, resolution) in enumerate(results):
        plotter.subplot(index // 3, index % 3)
        plotter.add_text(title, position=(.02, .94), viewport=True, font_size=12, color='white')
        markers, boundary_count = [], 0
        for mesh_index, mesh in enumerate(meshes):
            assert np.isfinite(mesh.vertices).all()
            faces = np.column_stack((np.full(len(mesh.edges), 3), mesh.edges)).ravel()
            surface = pv.PolyData(mesh.vertices, faces)
            color = '#f87979' if kind == 'fault' and mesh.stack_index == 0 else colors[mesh_index % len(colors)]
            actors.append(plotter.add_mesh(surface, color=color, opacity=.82, show_edges=True,
                                            edge_color='#34404a', line_width=.5,
                                            label=f'G{mesh.stack_index + 1}/S{mesh.surface_index + 1}'))
            boundary = surface.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                                      manifold_edges=False, non_manifold_edges=False)
            boundary_count += boundary.n_cells
            if boundary.n_cells:
                overlays.append(plotter.add_mesh(boundary, color='#ff668c', line_width=4,
                                                 lighting=False))
            active = np.unique(mesh.edges.ravel())
            ids = mesh.contact_report['contact_ids']
            shared = active[ids[active] >= 0]
            if len(shared):
                markers.append(mesh.vertices[shared])
        if markers:
            points = np.unique(np.concatenate(markers), axis=0)
            overlays.append(plotter.add_points(points, color='white', point_size=6,
                                               render_points_as_spheres=True, lighting=False))
        info = f'{boundary_count} per-interface boundary edges (seams included)'
        if kind == 'open':
            info += '\nPink edges expose the uncapped exterior perimeter.\nNo side walls, top/bottom caps or unit ownership yet.'
            plotter.add_mesh(pv.Box(bounds=(0, 1, 0, 1, 0, 1)), style='wireframe',
                             color='#94a5b7', opacity=.3, line_width=1)
        elif kind == 'onlap':
            x = .55 / 1.0625
            seam = np.array([[x, 0, .6 - .25 * x], [x, 1, .6 - .25 * x]])
            overlays.append(plotter.add_mesh(pv.Line(*seam), color='#32e6ef', line_width=6,
                                             lighting=False))
            errors, bends = [], []
            for mesh in meshes:
                active = np.unique(mesh.edges.ravel())
                shared = active[mesh.contact_report['contact_ids'][active] >= 0]
                if len(shared):
                    errors.extend(np.linalg.norm(mesh.vertices[shared][:, [0, 2]] - seam[0, [0, 2]], axis=1))
                    if mesh.stack_index == 1:
                        bends.extend(np.abs(mesh.vertices[shared] @ np.array([.25, 0, 1]) - .6) / np.sqrt(1.0625))
            error = max(errors, default=0.)
            info = (f'{resolution} cells/axis | max seam offset {error:.6f}\n'
                    f'Max substrate displacement {max(bends, default=0.):.6f}\n'
                    'Cyan = exact analytic seam; white = shared mesh points.\n'
                    'Shared means may bend the interface. Zoom to compare.')
            assert error > 0, 'This example should demonstrate an approximate contact'
        elif kind == 'isolation':
            upper = next(m for m in meshes if (m.stack_index, m.surface_index) == (0, 1))
            older = next(m for m in meshes if m.stack_index == 1)
            a, b = upper.contact_report['contact_ids'], older.contact_report['contact_ids']
            shared_ids = np.intersect1d(a[a >= 0], b[b >= 0])
            conflicts = meshes[0].contact_report['conflict_count']
            info = (f'{resolution} cells/axis | upper G1/S2-G2 contacts: {len(shared_ids)}\n'
                    f'{conflicts} provisional candidates rejected (not crack counts).\n'
                    'G1/S1 and G1/S2 never merge with each other.\n'
                    'A coarse cell cannot resolve every nearby surface.')
            if resolution == 64:
                assert not len(shared_ids), 'Fine isolation example unexpectedly touches'
        else:
            info = ('Real kriging, octree and contact extraction completed.\n'
                    f'Compute: {device}\n'
                    'Unverified: finite-fault tips, arbitrary junctions, full suite.\n'
                    'Not a closed volume or simulation mesh; no speed claim.')
        # VTK interprets pipes as table-column separators, not literal text.
        plotter.add_text(info.replace(' | ', '; '), position=(.02, .025), viewport=True,
                         font_size=10, color='#d0dde8')
        plotter.add_text('Pink: boundary; White: shared contacts\nH: overlays; E: edges; O: opacity; R: reset',
                         position=(.02, .865), viewport=True, font_size=9, color='#b6c6d5')
        plotter.add_axes()
        plotter.view_isometric()
        plotter.reset_camera()
        print(f'{title}: {info.replace(chr(10), " | ")}', flush=True)

    def toggle_overlays():
        for actor in overlays:
            actor.visibility = not actor.visibility
        plotter.render()

    def toggle_edges():
        for actor in actors:
            actor.prop.show_edges = not actor.prop.show_edges
        plotter.render()

    def toggle_opacity():
        for actor in actors:
            actor.prop.opacity = 1. if actor.prop.opacity < 1. else .55
        plotter.render()

    plotter.add_key_event('h', toggle_overlays)
    plotter.add_key_event('e', toggle_edges)
    plotter.add_key_event('o', toggle_opacity)
    print('READY: gallery computed; contact reconciliation is CPU/NumPy, even with CUDA fields/QEF.', flush=True)
    plotter.show(screenshot=args.screenshot, auto_close=False)
    for line in plotter.render_window.ReportCapabilities().splitlines():
        if 'OpenGL' in line and ('vendor' in line or 'renderer' in line or 'version' in line):
            print(f'VTK rendering: {line}', flush=True)
    plotter.close()


if __name__ == '__main__':
    main()
