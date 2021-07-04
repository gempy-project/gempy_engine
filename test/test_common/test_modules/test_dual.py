from gempy_engine.core.data.grid import Grid
import numpy as np
import os
import matplotlib.pyplot as plt
import pytest
from gempy_engine.modules.activator.activator_interface import activate_formation_block
import gempy_engine.integrations.interp_single.interp_single_interface as interp
from gempy_engine.core.data.internal_structs import SolverInput

from gempy_engine.core.data.exported_structs import OctreeLevel
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.integrations.interp_single.interp_single_interface import compute_n_octree_levels, interpolate_single_scalar
from gempy_engine.modules.dual_contouring.dual_contouring_interface import solve_qef_3d, QEF, find_intersection_on_edge
from gempy_engine.modules.octrees_topology.octrees_topology_interface import \
    get_regular_grid_for_level

dir_name = os.path.dirname(__file__)

plot_pyvista = True
try:
    # noinspection PyPackageRequirements
    import pyvista as pv
except ImportError:
    plot_pyvista = False

def test_find_edges_intersection(simple_model, simple_grid_3d_octree):
    
    # region Test find_intersection_on_edge
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = compute_n_octree_levels(2, interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[-1]

    sfsp = last_octree_level.output_corners.scalar_field_at_sp
    sfsp = np.append(sfsp, -0.1)
    xyz_on_edge, valid_edges = find_intersection_on_edge(
        last_octree_level.grid_corners.values,
        last_octree_level.output_corners.exported_fields.scalar_field,
        sfsp        
    )
    
    # endregion

    # region Get Normals

    interpolation_input.grid = Grid(xyz_on_edge)
    output_on_edges = interpolate_single_scalar(interpolation_input, options, data_shape)
    # stack gradients output_on_edges.exported_fields.gx_field
    gradients = np.stack(
        (output_on_edges.exported_fields.gx_field,
        output_on_edges.exported_fields.gy_field,
        output_on_edges.exported_fields.gz_field), axis=0).T[:-7]

    # endregion


    # Compute QEF
    v_mesh = []
    indeces = valid_edges.sum(axis=1).cumsum()
    indices = np.unique(indeces)[:]

    for i in range(len(indices) - 1):
        i_0 = indices[i]
        i_1 = indices[i+1]

        a = xyz_on_edge[i_0:i_1]
        b = gradients[i_0:i_1]
        
        mass_point = np.mean(a, axis=0)
        BIAS_STRENGTH = 0.1

        a = np.vstack((a, mass_point))
        a = np.vstack((a, mass_point))
        a = np.vstack((a, mass_point))
        b = np.vstack((b, np.array([BIAS_STRENGTH, 0, 0])))
        b = np.vstack((b, np.array([0, BIAS_STRENGTH, 0])))
        b = np.vstack((b, np.array([0, 0, BIAS_STRENGTH])))
        
        qef = QEF.make_3d(a, b)
        residual, v = qef.solve()
        

        v_mesh.append(v)

    if plot_pyvista:
        n = 1
        p = pv.Plotter()
        
        # Plot Actual mesh (from marching cubes)
        output_1_centers = last_octree_level.output_centers
        resolution = [20, 20, 20]
        mesh = _compute_actual_mesh(simple_model, ids, grid_0_centers, resolution,
        output_1_centers.scalar_field_at_sp,  output_1_centers.weights)

        p.add_mesh(mesh, opacity=1, silhouette=True)


        # Plot Regular grid Octree
        regular_grid_values = octree_list[n].grid_centers.regular_grid.values_vtk_format
        regular_grid_scalar = get_regular_grid_for_level(octree_list, n)

        shape = octree_list[n].grid_centers.regular_grid_shape
        grid_3d = regular_grid_values.reshape(*(shape + 1), 3).T
        regular_grid_mesh = pv.StructuredGrid(*grid_3d)
        regular_grid_mesh["lith"] = regular_grid_scalar.ravel()
        foo = regular_grid_mesh.threshold([0, 10])

        p.add_mesh(foo, show_edges=True, opacity=.5, cmap="tab10")

        # # Plot corners
        points = octree_list[n].grid_corners.values
        #p.add_mesh(pv.PolyData(xyz_on_edge), color="r", point_size=10.0, render_points_as_spheres=False)


        # Plot gradients
        poly = pv.PolyData(xyz_on_edge)
        poly['vectors'] = gradients
        arrows = poly.glyph(orient='vectors', scale=False, factor=.05)
        p.add_mesh(arrows, color="k", point_size=10.0, render_points_as_spheres=False)

        poly = pv.PolyData(a)
        poly['vectors'] = b

        arrows = poly.glyph(orient='vectors', scale=False, factor=.05)

        p.add_mesh(arrows, color="green", point_size=10.0, render_points_as_spheres=False)


        # Plot QEF
        p.add_mesh(pv.PolyData(v_mesh), color="b", point_size=15.0, render_points_as_spheres=False)

        p.add_axes()
        p.show()

    return xyz_on_edge, gradients


def test_QEF(simple_model, simple_grid_3d_octree):
    xyz_on_edge, gradients = test_find_edges_intersection(simple_model, simple_grid_3d_octree)
    qef = QEF.make_3d(xyz_on_edge, gradients)
    residual, v = qef.solve()
    print(v)




# =======================




def _compute_actual_mesh(simple_model, ids, grid, resolution, scalar_at_surface_points, weights):
    def _compute_high_res_model(data_shape, ids, interp_input, orientations, resolution, scalar_at_surface_points,
                                surface_points, weights):
        from test.fixtures.simple_models import create_regular_grid
        from gempy_engine.core.data.grid import Grid, RegularGrid

        grid_high_res = Grid.from_regular_grid(RegularGrid([0.25, .75, 0.25, .75, 0.25, .75], resolution))
        grid_internal_high_res, ori_internal, sp_internal = interp.input_preprocess(data_shape, grid_high_res,
                                                                                    orientations,
                                                                                    surface_points)
        exported_fields_high_res = interp._evaluate_sys_eq(grid_internal_high_res, interp_input, weights)
        values_block_high_res = activate_formation_block(exported_fields_high_res.scalar_field,
                                                         scalar_at_surface_points,
                                                         ids, sigmoid_slope=50000)
        return values_block_high_res, exported_fields_high_res, grid_high_res.dxdydz

    surface_points = simple_model[0]
    orientations = simple_model[1]
    options = simple_model[2]
    data_shape = simple_model[3]
    grid_internal, ori_internal, sp_internal = interp.input_preprocess(data_shape, grid, orientations,
                                                                       surface_points)
    interp_input = SolverInput(sp_internal, ori_internal, options)
    values_block_high_res, scalar_high_res, dxdydz = _compute_high_res_model(data_shape, ids, interp_input,
                                                                             orientations, resolution,
                                                                             scalar_at_surface_points, surface_points,
                                                                             weights)
    from skimage.measure import marching_cubes
    import pyvista as pv
    vert, edges, _, _ = marching_cubes(scalar_high_res.scalar_field[:-7].reshape(resolution),
                                       scalar_at_surface_points[0],
                                       spacing=dxdydz)
    loc_0 = np.array([0.25, .25, .25]) + np.array(dxdydz) / 2
    vert += np.array(loc_0).reshape(1, 3)
    mesh = pv.PolyData(vert, np.insert(edges, 0, 3, axis=1).ravel())
    return mesh



