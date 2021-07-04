import numpy as np
import os
import matplotlib.pyplot as plt
import pytest
from gempy_engine.modules.activator.activator_interface import activate_formation_block
import gempy_engine.integrations.interp_single.interp_single_interface as interp
from gempy_engine.core.data.internal_structs import SolverInput

from gempy_engine.core.data.exported_structs import OctreeLevel
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.integrations.interp_single.interp_single_interface import compute_n_octree_levels
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
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = compute_n_octree_levels(2, interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[-1]

    sfsp = last_octree_level.output_corners.scalar_field_at_sp
    sfsp = np.append(sfsp, -0.1)
    xyz_on_edge = find_intersection_on_edge(
        last_octree_level.grid_corners.values,
        last_octree_level.output_corners.exported_fields.scalar_field,
        sfsp        
    )

    if plot_pyvista:
        n = 1
        
        output_1_centers = last_octree_level.output_centers
        resolution = [20, 20, 20]
        mesh = _compute_actual_mesh(simple_model, ids, grid_0_centers, resolution,
        output_1_centers.scalar_field_at_sp,  output_1_centers.weights)

        p = pv.Plotter()

        p.add_mesh(mesh, opacity=1, silhouette=True)


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
        p.add_mesh(pv.PolyData(xyz_on_edge), color="r", point_size=10.0, render_points_as_spheres=False)

        p.add_axes()
        p.show()




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







def test_get_edges_values(simple_model, simple_grid_3d_octree):
    spi, ori_i, options, data_shape = simple_model
    ids = np.array([1, 2])
    grid_0_centers = simple_grid_3d_octree
    interpolation_input = InterpolationInput(spi, ori_i, grid_0_centers, ids)

    octree_list = compute_n_octree_levels(2, interpolation_input, options, data_shape)

    last_octree_level: OctreeLevel = octree_list[-1]
    #find_intersection_on_edge(last_octree_level.output_corners)


    if plot_pyvista:
        n = 1

        p = pv.Plotter()
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
        p.add_mesh(pv.PolyData(points), color="r", point_size=10.0, render_points_as_spheres=False)

        p.add_axes()
        p.show()










@pytest.mark.skip(reason="Not Implemented yet")
def test_dual_contouring_from_octree(simple_model_output):
    octrees = simple_model_output.octrees
    last_level: OctreeLevel = octrees[1]

    x = last_level.xyz_coords[0, 0]
    y = last_level.xyz_coords[0, 1]
    z = last_level.xyz_coords[0, 2]


    qef = QEF.make_3d(positions, normals)

    residual, v = qef.solve()

    solve_qef_3d()



@pytest.mark.skip(reason="Not Implemented yet")
def test_dual_contouring(plot=True):
    zx = np.load(dir_name + "/solutions/zx.npy")
    gx = np.load(dir_name + "/solutions/gx.npy")
    gy = np.load(dir_name + "/solutions/gy.npy")
    gz = np.load(dir_name + "/solutions/gz.npy")

    # TODO: Compute normal at edges



    if plot:
        plt.contourf(zx[:-7].reshape(50, 5, 50)[:, 2, :].T, N=40, cmap="autumn")
        plt.quiver(
            gx[:-7].reshape(50, 5, 50)[:, 2, :].T,
            gz[:-7].reshape(50, 5, 50)[:, 2, :].T,
            scale=10
        )

        plt.colorbar()


        plt.show()
