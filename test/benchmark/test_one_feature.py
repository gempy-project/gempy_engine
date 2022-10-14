from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.solutions import Solutions
from test.conftest import plot_pyvista


def test_one_feature(moureze_model, benchmark):
    BackendTensor.change_backend(
        engine_backend=AvailableBackends.numpy,
        use_gpu=False,
        pykeops_enabled=False
    )
    
    interpolation_input: InterpolationInput
    options: InterpolationOptions
    structure: InputDataDescriptor
    
    interpolation_input, options, structure = moureze_model
    n_oct_levels = options.number_octree_levels
    solutions: Solutions = benchmark.pedantic(
        target=compute_model,
        args=(interpolation_input, options, structure)
    )

    if plot_pyvista and False:
        import pyvista as pv
        from test.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_points

        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
        plot_dc_meshes(p, solutions.dc_meshes[0])
        plot_points(p, interpolation_input.surface_points.sp_coords)
        p.show()
