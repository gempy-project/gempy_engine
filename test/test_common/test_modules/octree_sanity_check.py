import numpy
import numpy as np

from gempy_engine.core.data import Orientations, SurfacePoints, InterpolationOptions, TensorsStructure
from gempy_engine.core.data.exported_structs import InterpOutput, OctreeLevel
from gempy_engine.core.data.grid import Grid
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.integrations.interp_single.interp_single_interface import input_preprocess, solve_interpolation, \
    _evaluate_sys_eq, _get_scalar_field_at_surface_points, compute_octree_level_n
from gempy_engine.modules.activator.activator_interface import activate_formation_block
from gempy_engine.modules.octrees_topology.octrees_topology_interface import compute_octree_root
from test.fixtures.simple_models import create_regular_grid, simple_grid_3d_more_points_grid
from test.test_common.test_modules.test_octrees import compute_high_res_model


def simple_model_f():


    numpy.set_printoptions(precision=3, linewidth=200)

    dip_positions = np.array([
        [0.25010, 0.50010, 0.54177],
        [0.66677, 0.50010, 0.62510],
    ])
    sp = np.array([
        [0.25010, 0.50010, 0.37510],
        [0.50010, 0.50010, 0.37510],
        [0.66677, 0.50010, 0.41677],
        [0.70843, 0.50010, 0.47510],
        [0.75010, 0.50010, 0.54177],
        [0.58343, 0.50010, 0.39177],
        [0.73343, 0.50010, 0.50010],
    ])

    nugget_effect_scalar = 0
    spi = SurfacePoints(sp, nugget_effect_scalar)

    dip_gradients = np.array([[0, 0, 1],
                              [-.6, 0, .8]])
    nugget_effect_grad = 0

    range_ = 4.166666666667
    co = 0.1428571429

    ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)

    kri = InterpolationOptions(range_, co, 0, i_res=1, gi_res=1,
                               number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)
    _ = np.ones(3)
    tensor_structure = TensorsStructure(np.array([7]))
    return spi, ori_i, kri, tensor_structure


simple_model = simple_model_f()
resolution = [50, 5, 50]
g, dx, dy, dz = create_regular_grid([0.25, .75, 0.25, .75, 0.25, .75], resolution)
grid = Grid(g, [g.shape[0]], [50, 5, 50], [dx, dy, dz])


surface_points = simple_model[0]
orientations = simple_model[1]
options = simple_model[2]
data_shape = simple_model[3]

ids = np.array([1, 2])
unit_values = ids

grid_internal, ori_internal, sp_internal = input_preprocess(data_shape, grid, orientations,
                                                            surface_points)
interp_input = SolverInput(sp_internal, ori_internal, options)

weights = solve_interpolation(interp_input)

exported_fields = _evaluate_sys_eq(grid_internal, interp_input, weights)

scalar_at_surface_points = _get_scalar_field_at_surface_points(
    exported_fields.scalar_field, data_shape.nspv, surface_points.n_points)

# -----------------
# Export and Masking operations can happen even in parallel
# TODO: [~X] Export block
values_block: np.ndarray = activate_formation_block(exported_fields.scalar_field, scalar_at_surface_points,
                                                    ids, sigmoid_slope=50000)

resolution = [50, 50, 50]
values_block_high_res, scalar_high_res = compute_high_res_model(data_shape, ids, interp_input, orientations,
                                                                resolution, scalar_at_surface_points,
                                                                surface_points, weights)

from skimage.measure import marching_cubes
import pyvista as pv
vert, edges, _, _ = marching_cubes(scalar_high_res.scalar_field[:-7].reshape(resolution), scalar_at_surface_points[0])
mesh = pv.PolyData(vert, np.insert(edges, 0, 3, axis=1).ravel())
mesh.plot()

output = InterpOutput()
output.grid = grid
output.exported_fields = exported_fields
output.weights = weights
output.scalar_field_at_sp = scalar_at_surface_points
output.values_block = values_block

grid = simple_grid_3d_more_points_grid

octree_lvl0 = OctreeLevel(grid.values, output.ids_block_regular_grid,
                          output.exported_fields_regular_grid,
                          is_root=True)

octree_lvl1 = compute_octree_root(octree_lvl0, grid.regular_grid, grid.dxdydz, compute_topology=True)

n_levels = 3  # TODO: Move to options
octree_list = [octree_lvl0, octree_lvl1]
for i in range(2, n_levels):
    next_octree = compute_octree_level_n(octree_list[-1], interp_input, output, unit_values, grid.dxdydz, i)
    octree_list.append(next_octree)

print(octree_list)
