from gempy_engine.API.interp_manager.interp_manager_api import interpolate_model, _interpolate_stack
from gempy_engine.API.interp_single._interp_single_internals import _compute_mask_components
from gempy_engine.core.data.input_data_descriptor import StackRelationType
from ...conftest import plot_pyvista

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
    from ...helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_points, plot_vector
except ImportError:
    plot_pyvista = False


def test_compute_mask_components(unconformity, n_oct_levels=3):

    interpolation_input, options, structure = unconformity
    print(interpolation_input)

    options.number_octree_levels = n_oct_levels
    solutions = _interpolate_stack(structure, interpolation_input, options)
    
    exported_fields_example = solutions[0].octrees_output[-1].output_centers.exported_fields
    mask_matrices = _compute_mask_components(exported_fields_example, StackRelationType.ERODE)
    print(mask_matrices)


def test_compute_mask_components_on_all_leaves(unconformity, n_oct_levels=3):
    interpolation_input, options, structure = unconformity
    print(interpolation_input)

    options.number_octree_levels = n_oct_levels
    solutions = _interpolate_stack(structure, interpolation_input, options)
    pass





def test_masking(unconformity, n_oct_levels=4):
    

    if plot_pyvista or True:
        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
        plot_points(p, solutions.debug_input_data.surface_points.sp_coords, True)

        xyz, gradients = solutions.debug_input_data.orientations.dip_positions, solutions.debug_input_data.orientations.dip_gradients
        poly = pv.PolyData(xyz)

        poly['vectors'] = gradients
        arrows = poly.glyph(orient='vectors', scale=True, factor=100)

        p.add_mesh(arrows, color="green", point_size=10.0, render_points_as_spheres=False)

        # TODO: Dual contour meshes look like they are not working
        # plot_dc_meshes(p, solutions.dc_meshes[0])
        p.show()
