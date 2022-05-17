from matplotlib import pyplot as plt

from gempy_engine.API.interp_manager.interp_manager_api import _compute_mask, _interpolate_all
from gempy_engine.API.interp_single._interp_single_internals import _compute_mask_components, interpolate_all_fields, _interpolate_stack
from gempy_engine.core.data.input_data_descriptor import StackRelationType, TensorsStructure
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_for_level
from ...conftest import plot_pyvista

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
    from ...helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_points, plot_vector
except ImportError:
    plot_pyvista = False


def test_extract_input_subsets(unconformity_complex):
    interpolation_input, options, input_descriptor = unconformity_complex
    stack_structure = input_descriptor.stack_structure
    for i in range(stack_structure.n_stacks):
        stack_structure.stack_number = i
        tensor_struct_i: TensorsStructure = TensorsStructure.from_tensor_structure_subset(input_descriptor, i)
        interpolation_input_i = InterpolationInput.from_interpolation_input_subset(interpolation_input, stack_structure)
        if i == 1:
            assert interpolation_input_i.surface_points.sp_coords.shape[0] == 2
        if i == 2:
            assert interpolation_input_i.surface_points.sp_coords.shape[0] == 3
        print("Iteration {}".format(i))
        print(tensor_struct_i)
        print(interpolation_input_i)
        


def test_compute_several_scalar_fields(unconformity_complex):
    """Plot each individual scalar field"""
    # TODO:
    # interpolation_input, options, structure = unconformity_complex
    # outputs = _interpolate_stack(ro)


def test_compute_mask_components(unconformity, n_oct_levels=3):
    """Plot each individual mask compontent"""
    # TODO:
    # interpolation_input, options, structure = unconformity
    # print(interpolation_input)
    # 
    # options.number_octree_levels = n_oct_levels
    # solutions = _interpolate_stack(structure, interpolation_input, options)
    # 
    # exported_fields_example = solutions[0].octrees_output[-1].output_centers.exported_fields
    # mask_matrices = _compute_mask_components(exported_fields_example, StackRelationType.ERODE)
    # print(mask_matrices)


def test_mask_arrays(unconformity_complex):
    interpolation_input, options, structure = unconformity_complex
    solutions = interpolate_all_fields(interpolation_input, options, structure)

    # resolution = [16, 16, 16]
    # extent = interpolation_input.grid.regular_grid.extent
    # 
    # regular_grid_scalar = get_regular_grid_for_level(solutions.octrees_output, 3).astype("int8")
    # plt.imshow(regular_grid_scalar.reshape(resolution)[:, resolution[1] // 2, :].T, extent=extent[[0, 1, 4, 5]])
    # plt.show()


def test_compute_mask_inner_loop(unconformity, n_oct_levels=4):
    pass
    # interpolation_input, options, structure = unconformity
    # print(interpolation_input)
    # 
    # options.number_octree_levels = n_oct_levels
    # solutions = _interpolate_all(interpolation_input, options, structure)
    # if True:
    #     resolution = [16, 16, 16]
    #     extent = interpolation_input.grid.regular_grid.extent
    # 
    #     regular_grid_scalar = get_regular_grid_for_level(solutions.octrees_output, 3).astype("int8")
    #     plt.imshow(regular_grid_scalar.reshape(resolution)[:, resolution[1] // 2, :].T, extent=extent[[0, 1, 4, 5]])
    #     plt.show()


def test_compute_mask_components_on_all_leaves(unconformity, n_oct_levels=4):
    pass
    # interpolation_input, options, structure = unconformity
    # print(interpolation_input)
    # 
    # options.number_octree_levels = n_oct_levels
    # solutions = _interpolate_stack(structure, interpolation_input, options)
    # 
    # mask_foo = _compute_mask(solutions)
    # 
    # regular_grid_octree = solutions[0].octrees_output[-1].grid_centers.regular_grid
    # regular_grid_resolution = solutions[0].octrees_output[-2].grid_centers.regular_grid.resolution
    # 
    # cross_section = regular_grid_octree.active_cells.reshape(regular_grid_resolution)[:, 0, :]
    # plt.imshow(cross_section)
    # plt.show()
    # pass


def test_masking(unconformity, n_oct_levels=4):
    pass
    # if plot_pyvista or True:
    #     pv.global_theme.show_edges = True
    #     p = pv.Plotter()
    #     plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
    #     plot_points(p, solutions.debug_input_data.surface_points.sp_coords, True)
    # 
    #     xyz, gradients = solutions.debug_input_data.orientations.dip_positions, solutions.debug_input_data.orientations.dip_gradients
    #     poly = pv.PolyData(xyz)
    # 
    #     poly['vectors'] = gradients
    #     arrows = poly.glyph(orient='vectors', scale=True, factor=100)
    # 
    #     p.add_mesh(arrows, color="green", point_size=10.0, render_points_as_spheres=False)
    # 
    #     # TODO: Dual contour meshes look like they are not working
    #     # plot_dc_meshes(p, solutions.dc_meshes[0])
    #     p.show()
