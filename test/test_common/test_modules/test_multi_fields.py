from typing import List

from matplotlib import pyplot as plt

from gempy_engine.API.interp_manager.interp_manager_api import  _interpolate, interpolate_model
from gempy_engine.API.interp_single._interp_single_internals import _compute_mask_components, interpolate_all_fields, _interpolate_stack
from gempy_engine.core.data.exported_structs import InterpOutput, Solutions
from gempy_engine.core.data.input_data_descriptor import StackRelationType, TensorsStructure
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_for_level
from ... import helper_functions_pyvista
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
        if i == 0:
            assert interpolation_input_i.surface_points.sp_coords.shape[0] == 3
        print("Iteration {}".format(i))
        print(tensor_struct_i)
        print(interpolation_input_i)


def test_compute_several_scalar_fields(unconformity_complex):
    """Plot each individual scalar field"""
    # TODO:
    interpolation_input, options, structure = unconformity_complex
    outputs: List[InterpOutput] = _interpolate_stack(structure, interpolation_input, options)

    if True:
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].values_block, grid)
        plot_block(outputs[1].values_block, grid)
        plot_block(outputs[2].values_block, grid)


def plot_block(block, grid):
    resolution = grid.resolution
    extent = grid.extent
    plt.imshow(block.reshape(resolution)[:, resolution[1] // 2, :].T, extent=extent[[0, 1, 4, 5]], origin="lower")
    plt.show()


def test_compute_mask_components_all_erode(unconformity_complex):
    """Plot each individual mask compontent"""
    # TODO:
    interpolation_input, options, structure = unconformity_complex
    outputs: List[InterpOutput] = _interpolate_stack(structure, interpolation_input, options)

    if True:
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].mask_components.mask_lith, grid)
        plot_block(outputs[1].mask_components.mask_lith, grid)
        plot_block(outputs[2].mask_components.mask_lith, grid)

# noinspection PyUnreachableCode
def test_mask_arrays(unconformity_complex):
    interpolation_input, options, structure = unconformity_complex
    outputs: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, structure)
    grid = interpolation_input.grid.regular_grid
    
    grid_0_centers = interpolation_input.grid
    from gempy_engine.modules.octrees_topology._octree_common import _generate_corners
    from gempy_engine.core.data.grid import Grid
    grid_0_corners = Grid(_generate_corners(grid_0_centers.values, grid_0_centers.dxdydz))
    interpolation_input.grid = grid_0_corners

    # TODO [x]: loop all scalars!!
    output_0_corners: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, structure)  # TODO: This is unnecessary for the last level except for Dual contouring
    # TODO: Final block is a (3, 7500) array

    if False:
        plot_block(outputs[0].squeezed_mask_array, grid)      
        plot_block(outputs[1].squeezed_mask_array, grid)
        plot_block(outputs[2].squeezed_mask_array, grid)
    
    mask_1 = output_0_corners[0].squeezed_mask_array.reshape((1, -1, 8)).sum(-1, bool)[0]
    mask_2 = output_0_corners[1].squeezed_mask_array.reshape((1, -1, 8)).sum(-1, bool)[0]
    mask_3 = output_0_corners[2].squeezed_mask_array.reshape((1, -1, 8)).sum(-1, bool)[0]
    if True:

        plot_block(mask_1, grid)
        plot_block(mask_2, grid)
        plot_block(mask_3, grid)

    if True:
        mask_1_f = mask_1 
        mask_2_f = (mask_1_f ^ mask_2) * mask_2
        mask_3_f = (mask_2_f ^ mask_3) * mask_3
        
        
        plot_block(mask_1_f, grid)
        plot_block(mask_2_f, grid)
        plot_block(mask_3_f, grid)


def test_final_block(unconformity_complex):
    interpolation_input, options, structure = unconformity_complex
    outputs: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, structure)

    # TODO: Final block is a (3, 7500) array

    if True:
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].final_block, grid)
        plot_block(outputs[1].final_block, grid)
        plot_block(outputs[2].final_block, grid)


def test_final_exported_fields(unconformity_complex):
    interpolation_input, options, structure = unconformity_complex
    outputs: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, structure)

    if True:
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].final_exported_fields._scalar_field, grid)
        plot_block(outputs[1].final_exported_fields._scalar_field, grid)
        plot_block(outputs[2].final_exported_fields._scalar_field, grid)
    

def test_plot_corners(unconformity_complex, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    solutions: Solutions = interpolate_model(interpolation_input, options, structure)
    output_corners: InterpOutput = solutions.octrees_output[-1].outputs_corners[-1]
    
    vertices = output_corners.grid.values
    helper_functions_pyvista.plot_pyvista(solutions.octrees_output,  v_just_points=vertices)
    

def test_dual_contouring_multiple_independent_fields(unconformity_complex, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    options.debug = True
    
    solutions: Solutions = interpolate_model(interpolation_input, options, structure)
    
    if True:
        
        dc_data = solutions.dc_meshes[1].dc_data # * Scalar field where to show gradients
        intersection_xyz = dc_data.xyz_on_edge
        gradients = dc_data.gradients

        center_mass = dc_data.bias_center_mass
        normals = dc_data.bias_normals

        helper_functions_pyvista.plot_pyvista(solutions.octrees_output,
                                              dc_meshes=solutions.dc_meshes,
                                              #xyz_on_edge=intersection_xyz, gradients=gradients, # * Uncomment for more detailed plots
                                              #a=center_mass, b=normals
                                              )


def test_dual_contouring_multiple_independent_fields_mask(unconformity_complex, n_oct_levels=4):
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.debug_water_tight = True

    solutions: Solutions = interpolate_model(interpolation_input, options, structure)

    if True:
        dc_data = solutions.dc_meshes[0].dc_data  # * Scalar field where to show gradients
        intersection_xyz = dc_data.xyz_on_edge
        gradients = dc_data.gradients

        center_mass = dc_data.bias_center_mass
        normals = dc_data.bias_normals

        helper_functions_pyvista.plot_pyvista(octree_list=solutions.octrees_output,
                                              dc_meshes=solutions.dc_meshes,
                                              #xyz_on_edge=intersection_xyz, gradients=gradients,
                                              #a=center_mass, b=normals,
                                              #vertices=solutions.dc_meshes[0].vertices, delaunay_3d=False
                                              )


def test_dual_contouring_multiple_dependent_fields(unconformity_complex, n_oct_levels=2):
    # * Dependent_dual_contouring seems a bad idea

    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.dependent_dual_contouring = True

    solutions: Solutions = interpolate_model(interpolation_input, options, structure)

    if True:
        dc_data = solutions.dc_meshes[0].dc_data
        intersection_xyz = dc_data.xyz_on_edge
        gradients = dc_data.gradients

        center_mass = dc_data.bias_center_mass
        normals = dc_data.bias_normals

        helper_functions_pyvista.plot_pyvista(solutions.octrees_output, dc_meshes=solutions.dc_meshes,
                                              xyz_on_edge=intersection_xyz, gradients=gradients,
                                              a=center_mass, b=normals
                                              )


def test_final_block_octrees(unconformity_complex, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    solution: Solutions = _interpolate(interpolation_input, options, structure)
    final_block = solution.octrees_output[0].output_centers.final_block
    final_block2 = get_regular_grid_for_level(solution.octrees_output, 1).astype("int8")

    # TODO: Final block is a (3, 7500) array

    if True:
        grid = interpolation_input.grid.regular_grid
        plot_block(final_block, grid)

        grid2 = solution.octrees_output[1].grid_centers.regular_grid
        plot_block(final_block2, grid2)


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
