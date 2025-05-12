import numpy as np

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import InterpolationOptions, SurfacePoints, Orientations, TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid, RegularGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.solutions import Solutions
from ...verify_helper import gempy_verify_array
from ...conftest import plot_pyvista

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
    from gempy_engine.plugins.plotting.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_points, plot_vector
except ImportError:
    plot_pyvista = False


def test_public_interface_simplest_model():
    # region InterpolationInput
    surface_points: SurfacePoints = SurfacePoints(
        sp_coords=np.array([
            [0.25010, 0.50010, 0.37510],
            [0.50010, 0.50010, 0.37510],
            [0.66677, 0.50010, 0.41677],
            [0.70843, 0.50010, 0.47510],
            [0.75010, 0.50010, 0.54177],
            [0.58343, 0.50010, 0.39177],
            [0.73343, 0.50010, 0.50010],
        ]))

    orientations: Orientations = Orientations(
        dip_positions=np.array([
            [0.25010, 0.50010, 0.54177],
            [0.66677, 0.50010, 0.62510],
        ]),
        dip_gradients=np.array([[0, 0, 1],
                                [-.6, 0, .8]])
    )

    regular_grid = RegularGrid(
        orthogonal_extent=[0.25, .75, 0.25, .75, 0.25, .75],
        regular_grid_shape=[2, 2, 3]
    )
    grid: EngineGrid = EngineGrid.from_regular_grid(regular_grid)

    interpolation_input: InterpolationInput = InterpolationInput(
        surface_points=surface_points,
        orientations=orientations,
        grid=grid
    )

    # endregion

    # region InterpolationOptions

    interpolation_options: InterpolationOptions = InterpolationOptions.from_args(
        range=4.166666666667,  # TODO: have constructor from RegularGrid
        c_o=0.1428571429,  # TODO: This should be a property
        number_octree_levels=3,
        kernel_function=AvailableKernelFunctions.cubic,
        uni_degree=0
    )
    interpolation_options.evaluation_options.mesh_extraction_fancy = True

    # endregion

    # region InputDataDescriptor
    tensor_struct: TensorsStructure = TensorsStructure(
        number_of_points_per_surface=np.array([7])
    )

    stack_structure: StacksStructure = StacksStructure(
        number_of_points_per_stack=np.array([7]),
        number_of_orientations_per_stack=np.array([2]),
        number_of_surfaces_per_stack=np.array([1]),
        masking_descriptor=[StackRelationType.ERODE]
    )

    input_data_descriptor: InputDataDescriptor = InputDataDescriptor(
        tensors_structure=tensor_struct,
        stack_structure=stack_structure
    )
    # endregion

    solutions = _compute_model(
        interpolation_input=interpolation_input,
        options=interpolation_options,
        structure=input_data_descriptor
    )
    
    if plot_pyvista or False:
        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, interpolation_options.number_octree_levels - 1)
        plot_dc_meshes(p, solutions.dc_meshes[0])
        surface_points_to_plot = interpolation_input.surface_points.sp_coords
        # If they are torch tensors convert to numpy
        if BackendTensor.engine_backend == AvailableBackends.PYTORCH and isinstance(surface_points_to_plot, BackendTensor.t.Tensor):
            surface_points_to_plot = BackendTensor.t.to_numpy(surface_points_to_plot)
            
        plot_points(p, surface_points_to_plot)
        p.show()


# noinspection DuplicatedCode
def _compute_model(interpolation_input: InterpolationInput, options: InterpolationOptions, structure: InputDataDescriptor):
    n_oct_levels = options.number_octree_levels
    solutions: Solutions = compute_model(interpolation_input, options, structure)

    # TODO [x]: Move Comparer to a separate file
    # TODO [ ]: Verify a bunch of the results to find discrepancies

    if options.debug:
        weights = Solutions.debug_input_data["weights"]
        A_matrix = Solutions.debug_input_data["A_matrix"]
        b_vector = Solutions.debug_input_data["b_vector"]
        cov_gradients = Solutions.debug_input_data["cov_grad"]
        cov_sp = Solutions.debug_input_data["cov_sp"]
        cov_grad_sp = Solutions.debug_input_data["cov_grad_sp"]
        uni_drift = Solutions.debug_input_data["uni_drift"]

        # ! This is commented until I fix the nugget
        if False:
            gempy_verify_array(BackendTensor.tfnp.sum(cov_gradients, axis=1, keepdims=True), "cov_gradients", 1e-1)
            gempy_verify_array(BackendTensor.tfnp.sum(cov_sp, axis=1, keepdims=True), "cov_sp", 1e-2)
            gempy_verify_array(BackendTensor.tfnp.sum(cov_grad_sp, axis=1, keepdims=True), "cov_grad_sp", 1e-2)
            gempy_verify_array(BackendTensor.tfnp.sum(uni_drift, axis=1, keepdims=True), "uni_drift", 1e-2)
            gempy_verify_array(b_vector, "b_vector")
            gempy_verify_array(BackendTensor.tfnp.sum(A_matrix, axis=1, keepdims=True), "A_matrix", 1e-2)
            gempy_verify_array(weights.reshape(1, -1), "weights", rtol=.1)

    return solutions