import pytest
import numpy as np
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import InterpolationOptions, SurfacePoints, Orientations, TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid, RegularGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
import os

# Check if pykeops is available
try:
    import pykeops

    PYKEOPS_AVAILABLE = True
except ImportError:
    PYKEOPS_AVAILABLE = False


@pytest.fixture
def override_setup():
    # 2 stacks, 1 surface each
    # Stack 0: 5 points, 1 orientation
    # Stack 1: 5 points, 1 orientation
    surface_points = SurfacePoints(sp_coords=np.random.rand(10, 3))
    orientations = Orientations(dip_positions=np.random.rand(2, 3), dip_gradients=np.random.rand(2, 3))
    grid = EngineGrid.from_regular_grid(RegularGrid(orthogonal_extent=[0, 1, 0, 1, 0, 1], regular_grid_shape=[2, 2, 2]))

    interpolation_input = InterpolationInput(
        surface_points=surface_points,
        orientations=orientations,
        grid=grid
    )

    # Global options: range=1.0, cubic kernel
    global_options = InterpolationOptions.from_args(
        range=1.0,
        c_o=1.0,
        number_octree_levels=1,
        kernel_function=AvailableKernelFunctions.cubic,
        uni_degree=0
    )
    global_options.evaluation_options.mesh_extraction_fancy = False
    global_options.mesh_extraction = False

    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([5, 5]))
    
    return interpolation_input, global_options, tensor_struct

def test_stack_options_override_serial(override_setup):
    ii, global_options, ts = override_setup
    
    # Custom options for stack 1: range=5.0
    custom_options = InterpolationOptions.from_args(
        range=5.0,
        c_o=1.0,
        number_octree_levels=1,
        kernel_function=AvailableKernelFunctions.cubic,
        uni_degree=0
    )

    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([5, 5]),
        number_of_orientations_per_stack=np.array([1, 1]),
        number_of_surfaces_per_stack=np.array([1, 1]),
        masking_descriptor=[StackRelationType.ERODE, StackRelationType.ERODE],
        interpolation_options_per_stack=[None, custom_options]
    )

    dd = InputDataDescriptor(tensors_structure=ts, stack_structure=stack_structure)

    # Force serial execution
    os.environ["GEMPY_FLAT_STACKS"] = "False"
    
    solutions = compute_model(ii, global_options, dd)
    
    # Verify that results are different because of different ranges
    # We check the scalar field values instead of weights
    field_0 = solutions.octrees_output[0].outputs[0].scalar_fields.exported_fields.scalar_field
    field_1 = solutions.octrees_output[0].outputs[1].scalar_fields.exported_fields.scalar_field
    
    # Different range -> different scalar field
    assert not np.array_equal(field_0, field_1)


@pytest.mark.skipif(not PYKEOPS_AVAILABLE, reason="pykeops not installed")
def test_stack_options_override_flat(override_setup):
    ii, global_options, ts = override_setup
    
    custom_options = InterpolationOptions.from_args(
        range=5.0,
        c_o=1.0,
        number_octree_levels=1,
        kernel_function=AvailableKernelFunctions.cubic,
        uni_degree=0
    )

    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([5, 5]),
        number_of_orientations_per_stack=np.array([1, 1]),
        number_of_surfaces_per_stack=np.array([1, 1]),
        masking_descriptor=[StackRelationType.ERODE, StackRelationType.ERODE],
        interpolation_options_per_stack=[None, custom_options]
    )

    dd = InputDataDescriptor(tensors_structure=ts, stack_structure=stack_structure)

    # Force flat execution
    os.environ["GEMPY_FLAT_STACKS"] = "True"
    
    from gempy_engine.core.backend_tensor import BackendTensor
    original_use_pykeops = BackendTensor.use_pykeops
    # We must mock use_pykeops to True to enter the flat path in _multi_scalar_field_manager.py
    BackendTensor.use_pykeops = True
    
    try:
        solutions = compute_model(ii, global_options, dd)
        
        # Verify that results are different because of different ranges
        field_0 = solutions.octrees_output[0].outputs[0].scalar_fields.exported_fields.scalar_field
        field_1 = solutions.octrees_output[0].outputs[1].scalar_fields.exported_fields.scalar_field
        
        assert not np.array_equal(field_0, field_1)
    finally:
        BackendTensor.use_pykeops = original_use_pykeops
        os.environ["GEMPY_FLAT_STACKS"] = "False"

def test_backward_compatibility(override_setup):
    ii, global_options, ts = override_setup
    
    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([5, 5]),
        number_of_orientations_per_stack=np.array([1, 1]),
        number_of_surfaces_per_stack=np.array([1, 1]),
        masking_descriptor=[StackRelationType.ERODE, StackRelationType.ERODE]
        # interpolation_options_per_stack is None by default
    )

    dd = InputDataDescriptor(tensors_structure=ts, stack_structure=stack_structure)
    
    solutions = compute_model(ii, global_options, dd)
    assert len(solutions.octrees_output[0].outputs) == 2
