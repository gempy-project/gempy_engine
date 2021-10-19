import pykeops
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.integrations.interp_manager.interp_manager_api import interpolate_model
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance, yield_b_vector

from test.helper_functions import plot_octree_pyvista, plot_dc_meshes, plot_points, plot_vector
from ...conftest import plot_pyvista

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
except ImportError:
    plot_pyvista = False

class TestInterpolateModel:
    BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=False, pykeops_enabled=False)

    def test_interpolate_model(self, simple_model_interpolation_input, n_oct_levels = 3):
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


class TestInterpolateModelHyperParameters:
    """
    This a test to see the different geometries that cubic vs exp function produces and the impact it has
    using square distances vs proper distances.

    This is important because pykeops "only" works for exp function and squared distance
    """

    def test_interpolate_model(self, simple_model_interpolation_input, n_oct_levels = 3):
        """
        Cubic, euclidean distances, exact range
        """
        BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=False, pykeops_enabled=False)
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

    def test_interpolate_model2(self, simple_model_interpolation_input, n_oct_levels = 3):
        """
        Cubic, SQUARED euclidean distances, exact range
        """
        BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=False, pykeops_enabled=False)
        BackendTensor.euclidean_distances_in_interpolation = False
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

        BackendTensor.euclidean_distances_in_interpolation = True

    def test_interpolate_model3(self, simple_model_interpolation_input, n_oct_levels = 3):
        """
        exp, SQRT euclidean distances, exact range
        """
        BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=False, pykeops_enabled=False)
        interpolation_input, options, structure = simple_model_interpolation_input
        options.kernel_function = AvailableKernelFunctions.exponential
        range_ = .5
        co = range_ ** 2 / 14 / 3

        options.range = range_
        options.co = co

        print(interpolation_input)

        options.number_octree_levels = n_oct_levels
        solutions = interpolate_model(interpolation_input, options ,structure)

        if plot_pyvista or True:
           # pv.global_theme.show_edges = True
            p = pv.Plotter()
            plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
            plot_dc_meshes(p, solutions.dc_meshes[0])
            plot_points(p, interpolation_input.surface_points.sp_coords)
            p.show()

    def test_interpolate_model4(self, simple_model_interpolation_input, n_oct_levels = 3):
        """
        exp, euclidean distances, exact range
        """
        BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=False, pykeops_enabled=True)
        BackendTensor.euclidean_distances_in_interpolation = False

        interpolation_input, options, structure = simple_model_interpolation_input
        options.kernel_function = AvailableKernelFunctions.exponential
        range_ = 4
        co = range_ ** 2 / 14 / 3

        options.range = range_
        options.co = co
        print(interpolation_input)

        options.number_octree_levels = n_oct_levels
        solutions = interpolate_model(interpolation_input, options ,structure)

        if plot_pyvista or True:
           # pv.global_theme.show_edges = True
            p = pv.Plotter()
            plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
            plot_dc_meshes(p, solutions.dc_meshes[0])
            plot_points(p, interpolation_input.surface_points.sp_coords)

            p.show()

        BackendTensor.euclidean_distances_in_interpolation = True

    def test_interpolate_recumbent(self, recumbent_fold_scaled, n_oct_levels=1):
        BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=False, pykeops_enabled=True)

        interpolation_input, options, structure = recumbent_fold_scaled
        options.kernel_function = AvailableKernelFunctions.exponential
        options.uni_degree=0
        print(interpolation_input)

        options.number_octree_levels = n_oct_levels
        solutions = interpolate_model(interpolation_input, options ,structure)

        if plot_pyvista or True:
           # pv.global_theme.show_edges = True
            p = pv.Plotter()
            plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
            plot_dc_meshes(p, solutions.dc_meshes[0])
            plot_points(p, interpolation_input.surface_points.sp_coords)

            p.show()



@pytest.mark.skipif(BackendTensor.engine_backend is AvailableBackends.tensorflow, reason="Only test against numpy")
class TestInterpolateModelOptimized:
    def test_interpolate_model_weights(self, simple_model_interpolation_input_optimized, n_oct_levels=3):
        BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=False, pykeops_enabled=True)
        #pykeops.clean_pykeops()
        import numpy as np
        from pykeops.numpy import Vi, Vj, Pm

        interpolation_input, options, structure = simple_model_interpolation_input_optimized
        print(interpolation_input)

        surface_points = interpolation_input.surface_points
        orientations = interpolation_input.orientations
        tensors_structure = structure
        options.range = 4.464646446464646464#Pm(4.44)#Pm(np.array([4.4453525], dtype="float32"))
        options.i_res = 4#4#Pm(np.array([4], dtype="float32"))#4
        options.gi_res = 2# Pm(np.array([2], dtype="float32"))#2

        from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess
        sp_internals = surface_points_preprocess(surface_points,
                                                 tensors_structure.number_of_points_per_surface)
        from gempy_engine.modules.data_preprocess._input_preparation import orientations_preprocess
        ori_internals = orientations_preprocess(orientations)

        from gempy_engine.core.data.internal_structs import SolverInput
        si = SolverInput(sp_internals, ori_internals, options)
        cov = yield_covariance(si)
        print("\n")
        print(cov)
        print("cov_sum 0", cov.sum(0, backend="CPU"))

        # Test weights and b vector
        b_vec = yield_b_vector(ori_internals, cov.shape[0])
        from gempy_engine.modules.solver.solver_interface import kernel_reduction
        weights = kernel_reduction(cov, b_vec, 1, False)
        print(weights)


    def test_interpolate_model(self, simple_model_interpolation_input_optimized, n_oct_levels = 3):
        BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=False, pykeops_enabled=True)

        interpolation_input, options, structure = simple_model_interpolation_input_optimized
        options.kernel_function = AvailableKernelFunctions.cubic
        options.range = 4.464646446464646464
        options.i_res = 4
        options.gi_res = 2

        options.number_octree_levels = n_oct_levels
        solutions = interpolate_model(interpolation_input, options ,structure)

        if plot_pyvista or False:
           # pv.global_theme.show_edges = True
            p = pv.Plotter()
            plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
            plot_points(p, interpolation_input.surface_points.sp_coords, True)

            plot_dc_meshes(p, solutions.dc_meshes[0])
            p.show()

    def test_interpolate_model_no_octtree(self, simple_model_3_layers_high_res, n_oct_levels = 1):
        interpolation_input, options, structure = simple_model_3_layers_high_res
        BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=False, pykeops_enabled=True)

        options.kernel_function = AvailableKernelFunctions.exponential
        options.range = 4.464646446464646464
        options.i_res = 4
        options.gi_res = 2

        options.number_octree_levels = n_oct_levels
        solutions = interpolate_model(interpolation_input, options ,structure)

        if plot_pyvista or False:
           # pv.global_theme.show_edges = True
            p = pv.Plotter()
            plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
            plot_points(p, interpolation_input.surface_points.sp_coords, True)
            plot_dc_meshes(p, solutions.dc_meshes[0])
            p.show()


    def test_interpolate_model_several_surfaces(self, simple_model_3_layers, n_oct_levels = 3):
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