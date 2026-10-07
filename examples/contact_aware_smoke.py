"""Five manual-only production-path examples for contact-aware mesh inspection."""

import argparse
import importlib
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
from gempy_engine.config import AvailableBackends, DualContouringOverlap
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--off-screen', action='store_true')
    parser.add_argument('--screenshot', default=None)
    parser.add_argument('--isolation-resolution', type=int, choices=[8, 16, 32, 64], default=64)
    parser.add_argument('--onlap-resolution', type=int, choices=[8, 16, 32, 64], default=64)
    parser.add_argument('--measure-onlap', action='store_true', help='Compute only onlap diagnostics, without opening a window')
    args = parser.parse_args()
    BackendTensor._change_backend(AvailableBackends.numpy, use_gpu=False,
                                  use_pykeops=False, dtype='float64', grads=False)
    dc = importlib.import_module('gempy_engine.API.dual_contouring.multi_scalar_dual_contouring')
    dc.DUAL_CONTOURING_VERTEX_OVERLAP = DualContouringOverlap.contact_aware

    examples = [
        ('1. Planar erosion', 'A tilted horizon terminates against a younger plane.',
         [(0, 0, 1), (1, 0, .3)], [[.53], [.62]], [0, 0], [R.ERODE, R.BASEMENT]),
        ('2. Onlap', 'The upper horizon meets an inclined substrate.',
         [(1, 0, -.25), (.25, 0, 1)], [[.4], [.6]], [0, 0], [R.ONLAP, R.BASEMENT]),
        ('3. Curved three-way contact', 'Curved erosion boundary and two intersecting groups.',
         [(0, 0, 1), (1, 0, .15), (0, 1, .15)], [[.56], [.57], [.58]],
         [.65, 0, 0], [R.ERODE, R.ERODE, R.BASEMENT]),
        ('4. Same-group isolation', 'Two close horizons must not collapse through another group.',
         [(0, 0, 1), (1, 0, .2)], [[.43, .46], [.59]], [0, 0], [R.ERODE, R.BASEMENT]),
    ]
    if args.measure_onlap:
        examples = [examples[1]]
    results = []
    for title, description, normals, levels, curvature, relations in examples:
        isolation = title.startswith('4.')
        onlap = title.startswith('2.')
        resolution = args.isolation_resolution if isolation else args.onlap_resolution if onlap else 8
        root_resolution, octree_levels = ((16, 3) if resolution == 64 else (resolution // 2, 2))
        functions = []
        for normal, values, bend in zip(normals, levels, curvature):
            normal = np.asarray(normal, dtype=float)
            functions.append(CustomInterpolationFunctions(
                scalar_field_at_surface_points=np.array(values),
                implicit_function=lambda xyz, n=normal, b=bend: xyz @ n + b * (xyz[:, 0] - .5) ** 2,
                gx_function=lambda xyz, n=normal, b=bend: n[0] + 2 * b * (xyz[:, 0] - .5),
                gy_function=lambda xyz, n=normal: np.full(len(xyz), n[1]),
                gz_function=lambda xyz, n=normal: np.full(len(xyz), n[2]),
            ))
        n = len(functions)
        descriptor = InputDataDescriptor(
            TensorsStructure(np.array([], dtype=int)),
            StacksStructure(np.zeros(n, dtype=int), np.zeros(n, dtype=int),
                            np.array([len(values) for values in levels]), relations,
                            faults_relations=np.zeros((n, n), dtype=bool),
                            interp_functions_per_stack=functions),
        )
        inputs = InterpolationInput(
            SurfacePoints(np.empty((0, 3))), Orientations(np.empty((0, 3)), np.empty((0, 3))),
            EngineGrid(octree_grid=RegularGrid(np.array([0., 1., 0., 1., 0., 1.]), [root_resolution] * 3)),
        )
        options = InterpolationOptions.from_args(10., 1.)
        options.evaluation_options.number_octree_levels = octree_levels
        options.evaluation_options.number_octree_levels_surface = octree_levels
        options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
        print(f'Computing {title}', flush=True)
        meshes = compute_model(inputs, options, descriptor).dc_meshes
        if onlap:
            substrate_error = 0.
            band_radius = 0.
            unshared_movement = 0.
            unshared_plane_error = 0.
            line_xz = np.array([.55 / 1.0625, .6 - .25 * (.55 / 1.0625)])
            for mesh in meshes:
                shared = mesh.contact_report['contact_ids'] >= 0
                active = np.zeros(len(mesh.vertices), dtype=bool)
                active[np.unique(mesh.edges.ravel())] = True
                inactive_shared = int(np.count_nonzero(shared & ~active))
                print(f'Onlap G{mesh.stack_index + 1}/S{mesh.surface_index + 1}: '
                      f'inactive shared vertices={inactive_shared}', flush=True)
                assert inactive_shared == 0, 'Shared onlap vertices are not referenced by final triangles'
                original = np.asarray(mesh.vertices_tensor)
                normal = np.asarray(normals[mesh.stack_index])
                residual = np.abs(mesh.vertices @ normal - levels[mesh.stack_index][0]) / np.linalg.norm(normal)
                if np.any(shared):
                    points = mesh.vertices[shared]
                    interior = points[(points[:, 1] > 0) & (points[:, 1] < 1)]
                    if len(interior):
                        row_y = interior[np.argmin(np.abs(interior[:, 1] - .5)), 1]
                        row_xz = interior[np.isclose(interior[:, 1], row_y)][:, [0, 2]]
                        print(f'  Shared row y={row_y:.9f}, x,z={np.unique(row_xz, axis=0).tolist()}', flush=True)
                    band_radius = max(band_radius, float(np.linalg.norm(mesh.vertices[shared][:, [0, 2]] - line_xz, axis=1).max()))
                    if mesh.stack_index == 1:
                        substrate_error = float(residual[shared].max())
                if np.any(~shared):
                    unshared_movement = max(unshared_movement, float(np.linalg.norm(mesh.vertices[~shared] - original[~shared], axis=1).max()))
                    unshared_plane_error = max(unshared_plane_error, float(residual[~shared].max()))
            assert unshared_movement == 0, 'Unshared onlap vertices were moved'
            print(f'Onlap {resolution}: max substrate plane distance={substrate_error:.9f}, '
                  f'max contact-line distance={band_radius:.9f}, unshared movement={unshared_movement:.3e}, '
                  f'unshared plane distance={unshared_plane_error:.3e}', flush=True)
            title = f'2. Onlap ({resolution})'
            description = f'Max substrate bend {substrate_error:.5f}; unshared vertices unchanged.'
        if isolation:
            upper = next(mesh for mesh in meshes if (mesh.stack_index, mesh.surface_index) == (0, 1))
            older = next(mesh for mesh in meshes if mesh.stack_index == 1)
            upper_cells = upper.dc_data.left_right_codes[upper.dc_data.valid_voxels]
            older_cells = older.dc_data.left_right_codes[older.dc_data.valid_voxels]
            shared_cells = set(map(tuple, upper_cells)) & set(map(tuple, older_cells))
            upper_ids = upper.contact_report['contact_ids']
            older_ids = older.contact_report['contact_ids']
            shared_ids = np.intersect1d(upper_ids[upper_ids >= 0], older_ids[older_ids >= 0])
            gap = float(upper.vertices[:, 2].min() - older.vertices[:, 2].max())
            print(f'Isolation {resolution}: G1/S2-G2 shared cells={len(shared_cells)}, '
                  f'shared contact IDs={len(shared_ids)}, vertical gap={gap:.6f}', flush=True)
            if resolution == 64:
                assert not shared_cells and not len(shared_ids) and gap > 0, 'Fine isolation surfaces still touch'
            title = f'4. Same-group isolation ({resolution})'
            description = f'G1/S2-G2: {len(shared_ids)} shared contacts; vertical gap {gap:.4f}.'
        results.append((title, description, meshes, False))

    if args.measure_onlap:
        return

    # Reuse the real kriging fixture's underlying factory, without pytest execution.
    from tests.fixtures.complex_geometries import one_fault_model
    inputs, descriptor, options = one_fault_model.__wrapped__()
    options.evaluation_options.number_octree_levels = 3
    options.evaluation_options.number_octree_levels_surface = 3
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
    print('Computing 5. Faulted stratigraphy (real kriging model)', flush=True)
    results.append(('5. Faulted stratigraphy', 'Real kriging model: red fault, displaced layers, erosion.',
                    compute_model(inputs, options, descriptor).dc_meshes, True))

    plotter = pv.Plotter(shape=(2, 3), window_size=(1680, 1000),
                         title='GemPy contact_aware | Five Smoke Geometries', off_screen=args.off_screen)
    plotter.set_background('#18212b', all_renderers=True)
    colors = ['#48a9e6', '#ffb15a', '#68cf9b', '#d894ec', '#f4d66c', '#bdc9d5']
    surface_actors, contact_actors = [], []
    for index, (title, description, meshes, fault_model) in enumerate(results):
        plotter.subplot(index // 3, index % 3)
        plotter.add_text(title, position='upper_left', font_size=13, color='white')
        plotter.add_text(description, position='lower_left', font_size=9, color='#c9d5df')
        markers = []
        for mesh_index, mesh in enumerate(meshes):
            if not np.isfinite(mesh.vertices).all():
                raise ValueError(f'{title}: nonfinite vertices')
            if len(mesh.edges) and not np.all((mesh.edges >= 0) & (mesh.edges < len(mesh.vertices))):
                raise ValueError(f'{title}: invalid triangle indices')
            color = '#f56a6a' if fault_model and mesh.stack_index == 0 else colors[mesh_index % len(colors)]
            name = ('Fault' if fault_model and mesh.stack_index == 0 else
                    f'G{mesh.stack_index + 1} / S{mesh.surface_index + 1}')
            if len(mesh.edges):
                faces = np.column_stack((np.full(len(mesh.edges), 3), mesh.edges)).ravel()
                actor = plotter.add_mesh(pv.PolyData(mesh.vertices, faces), color=color,
                                         opacity=.88, show_edges=True, edge_color='#29343e',
                                         line_width=.5, label=name, smooth_shading=False)
                surface_actors.append(actor)
            ids = mesh.contact_report['contact_ids']
            active = np.unique(mesh.edges.ravel()) if len(mesh.edges) else np.empty(0, dtype=int)
            shared = active[ids[active] >= 0]
            if len(shared):
                markers.append(mesh.vertices[shared])
        if markers:
            points = np.unique(np.concatenate(markers), axis=0)
            actor = plotter.add_points(points, color='white', point_size=6,
                                       render_points_as_spheres=True, lighting=False)
            contact_actors.append(actor)
        contacts = meshes[0].contact_report['contact_count'] if meshes else 0
        conflicts = meshes[0].contact_report['conflict_count'] if meshes else 0
        plotter.add_text(f'{contacts} shared cell sets | {conflicts} rejected candidates',
                         position='upper_right', font_size=9, color='#c9d5df')
        plotter.add_legend(size=(.44, .19), bcolor='#18212b', face='rectangle', loc='lower right')
        plotter.add_axes()
        plotter.view_isometric()
        plotter.reset_camera()
        print(f'{title}: {len(meshes)} surfaces, {sum(len(m.edges) for m in meshes)} triangles, '
              f'{contacts} shared sets, {conflicts} rejected candidates', flush=True)

    plotter.subplot(1, 2)
    plotter.add_text('Visual Smoke Test', position='upper_left', font_size=21, color='white')
    plotter.add_text(
        'Mode: contact_aware\n\n'
        'White dots: shared cell contacts\n'
        'Colors: individual geological surfaces\n\n'
        'Drag: rotate the active example\n'
        'Wheel: zoom\n'
        'Shift + drag: pan\n'
        'C: show/hide contact dots\n'
        'E: show/hide triangle edges\n'
        'O: toggle translucent/opaque surfaces\n'
        'R: reset active camera\n\n'
        'Coarse-resolution sticking is accepted.\n'
        'Onlap and isolation use finer grids.\n'
        'Orange G1/S2 must separate from G2;\n'
        'blue G1/S1 still meets G2.\n\n'
        'These are surface interfaces,\nnot closed lithological solids.',
        position=(15, 40), font_size=10, color='#c9d5df',
    )

    def toggle_contacts():
        for actor in contact_actors:
            actor.visibility = not actor.visibility
        plotter.render()

    def toggle_edges():
        for actor in surface_actors:
            actor.prop.show_edges = not actor.prop.show_edges
        plotter.render()

    def toggle_opacity():
        for actor in surface_actors:
            actor.prop.opacity = 1. if actor.prop.opacity < 1. else .55
        plotter.render()

    plotter.add_key_event('c', toggle_contacts)
    plotter.add_key_event('e', toggle_edges)
    plotter.add_key_event('o', toggle_opacity)
    print('READY: opening PyVista window', flush=True)
    plotter.show(screenshot=args.screenshot)
    print('PyVista window closed', flush=True)


if __name__ == '__main__':
    main()
