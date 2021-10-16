import tensorflow

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.exported_structs import InterpOutput
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.integrations.interp_manager.interp_manager_api import interpolate_model
from gempy_engine.integrations.interp_single._interp_single_internals import _input_preprocess, _solve_interpolation
from gempy_engine.integrations.interp_single.interp_single_interface import interpolate_single_field
from gempy_engine.modules import kernel_constructor
from gempy_engine.modules.kernel_constructor import kernel_constructor_interface
from gempy_engine.modules.solver import solver_interface
from test.helper_functions import plot_octree_pyvista, plot_dc_meshes, plot_points, plot_vector, \
    plot_2d_scalar_y_direction

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

        if plot_pyvista or False:
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

        if plot_pyvista or False:
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


class TestInterpolateModelTFEager:
    def test_interpolate_model(self, simple_model_interpolation_input, n_oct_levels=3):
        BackendTensor.change_backend(AvailableBackends.tensorflow, use_gpu=False)

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

        if plot_pyvista or False:
           # pv.global_theme.show_edges = True
            p = pv.Plotter()
            plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
            plot_dc_meshes(p, solutions.dc_meshes[0])
            p.show()

    def test_interpolate_model_no_octtree_gpu(self, simple_model_3_layers_high_res, n_oct_levels = 1):
        BackendTensor.change_backend(AvailableBackends.tensorflow, use_gpu=True)

        interpolation_input, options, structure = simple_model_3_layers_high_res
        print(interpolation_input)

        options.number_octree_levels = n_oct_levels
        solutions = interpolate_model(interpolation_input, options ,structure)

        if plot_pyvista or False:
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


class TestInterpolateModelTFXLA:
    def test_interpolate_model_no_octtree_func(self, simple_model_3_layers_high_res, n_oct_levels = 1):

        tensorflow.config.run_functions_eagerly(False)

        BackendTensor.change_backend(AvailableBackends.tensorflow, use_gpu=False)

        interpolation_input, options, structure = simple_model_3_layers_high_res
        print(interpolation_input)

        options.number_octree_levels = n_oct_levels

        @tensorflow.function
        def tf_function(interpolation_input, options ,structure):
            solutions = interpolate_model(interpolation_input, options ,structure)
            return solutions

        solutions = tf_function(interpolation_input, options, structure)

        if plot_pyvista or False:
           # pv.global_theme.show_edges = True
            p = pv.Plotter()
            plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
            plot_dc_meshes(p, solutions.dc_meshes[0])
            p.show()

    def test_interpolate_model_no_octtree_func_step_by_step(self, simple_model_3_layers_high_res, n_oct_levels = 1):
        BackendTensor.change_backend(AvailableBackends.tensorflow, use_gpu=False)

        interpolation_input, options, structure = simple_model_3_layers_high_res
        print(interpolation_input)

        options.number_octree_levels = n_oct_levels

        @tensorflow.function(experimental_compile=True)
        def tf_function(interpolation_input, options ,structure):
            # TODO: Open this function

            # region interpolate_scalar_field
            grid = interpolation_input.grid
            surface_points = interpolation_input.surface_points
            orientations = interpolation_input.orientations

            # Within series
            xyz_lvl0, ori_internal, sp_internal = _input_preprocess(structure, grid, orientations,
                                                                    surface_points)
            solver_input = SolverInput(sp_internal, ori_internal, options)

            #region solve interpolation
            interp_input = solver_input

            tensorflow.print(solver_input.ori_internal.n_orientations_tiled)
            tensorflow.print(solver_input.sp_internal.n_points)
            tensorflow.print(solver_input.sp_internal.get_n_points())
            tensorflow.print(solver_input.sp_internal.ref_surface_points)
            tensorflow.print(solver_input.sp_internal.ref_surface_points.shape)
            tensorflow.print(solver_input.sp_internal.rest_surface_points.shape)
            tensorflow.print(ori_internal.orientations.dip_positions.shape)

            tensorflow.print(options.n_uni_eq)

         #   A_matrix = kernel_constructor_interface.yield_covariance(interp_input)
            # b_vector = kernel_constructor.yield_b_vector(interp_input.ori_internal, A_matrix.shape[0])
            # # TODO: Smooth should be taken from options
            # weights = solver_interface.kernel_reduction(A_matrix, b_vector, smooth=0.01)
            #endregion

            # endregion

            # weights = output.weights
            # print(weights)
            #
            # Z_x = output.exported_fields.scalar_field


            return interp_input.sp_internal.n_points

        Z_x = tf_function(interpolation_input, options, structure)

        if False:
            plot_2d_scalar_y_direction(interpolation_input, Z_x)

