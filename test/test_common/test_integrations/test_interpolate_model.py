from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.integrations.interp_manager.interp_manager_api import interpolate_model
from test.helper_functions import plot_octree_pyvista, plot_dc_meshes, plot_points, plot_vector

from ...conftest import plot_pyvista

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
except ImportError:
    plot_pyvista = False

class TestInterpolateModelNumpy:
    BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=False)

    def test_interpolate_model(self, simple_model_interpolation_input, n_oct_levels = 3):
        BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=False)

        interpolation_input, options, structure = simple_model_interpolation_input
        print(interpolation_input)

        options.number_octree_levels = n_oct_levels
        solutions = interpolate_model(interpolation_input, options ,structure)

        if plot_pyvista or True:
           # pv.global_theme.show_edges = True
            p = pv.Plotter()
            plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
            plot_dc_meshes(p, solutions.dc_meshes[0])
            p.show()

    def test_interpolate_model_no_octtree(self, simple_model_3_layers_high_res, n_oct_levels = 1):
        BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=False)

        interpolation_input, options, structure = simple_model_3_layers_high_res
        print(interpolation_input)

        options.number_octree_levels = n_oct_levels
        solutions = interpolate_model(interpolation_input, options ,structure)

        if plot_pyvista or True:
           # pv.global_theme.show_edges = True
            p = pv.Plotter()
            plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
            plot_dc_meshes(p, solutions.dc_meshes[0])
            p.show()


    def test_interpolate_model_several_surfaces(self, simple_model_3_layers, n_oct_levels = 3):
        BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=False)

        interpolation_input, options, structure = simple_model_3_layers
        print(interpolation_input)

        options.number_octree_levels = n_oct_levels
        solutions = interpolate_model(interpolation_input, options ,structure)

        if plot_pyvista or False:
            # pv.global_theme.show_edges = True
            p = pv.Plotter()
            plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
            plot_dc_meshes(p, solutions.dc_meshes[0])

            plot_points(p, interpolation_input.surface_points.sp_coords, True)
            plot_vector(p, interpolation_input.orientations.dip_positions,
                        interpolation_input.orientations.dip_gradients)
            p.show()


class TestInterpolateModelTF:
    BackendTensor.change_backend(AvailableBackends.tensorflow, use_gpu=False)
    def test_interpolate_model(self, simple_model_interpolation_input, n_oct_levels=3):
        interpolation_input, options, structure = simple_model_interpolation_input
        print(interpolation_input)

        options.number_octree_levels = n_oct_levels
        solutions = interpolate_model(interpolation_input, options, structure)

        if plot_pyvista or False:
            #pv.global_theme.show_edges = True
            p = pv.Plotter()
            plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
            plot_dc_meshes(p, solutions.dc_meshes[0])
            p.show()

    def test_interpolate_model_no_octtree(self, simple_model_3_layers_high_res, n_oct_levels = 1):
        BackendTensor.change_backend(AvailableBackends.tensorflow, use_gpu=False)

        interpolation_input, options, structure = simple_model_3_layers_high_res
        print(interpolation_input)

        options.number_octree_levels = n_oct_levels
        solutions = interpolate_model(interpolation_input, options ,structure)

        if plot_pyvista or True:
           # pv.global_theme.show_edges = True
            p = pv.Plotter()
            plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
            plot_dc_meshes(p, solutions.dc_meshes[0])
            p.show()


    def test_interpolate_model_several_surfaces(self, simple_model_3_layers, n_oct_levels=3):
        interpolation_input, options, structure = simple_model_3_layers
        print(interpolation_input)

        options.number_octree_levels = n_oct_levels
        solutions = interpolate_model(interpolation_input, options, structure)

        if plot_pyvista or False:
            # pv.global_theme.show_edges = True
            p = pv.Plotter()
            plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
            plot_dc_meshes(p, solutions.dc_meshes[0])

            plot_points(p, interpolation_input.surface_points.sp_coords, True)
            plot_vector(p, interpolation_input.orientations.dip_positions,
                        interpolation_input.orientations.dip_gradients)
            p.show()



