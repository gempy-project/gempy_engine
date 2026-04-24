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
from gempy_engine.core.exceptions import GemPyEngineInputError


@pytest.fixture
def basic_input():
    surface_points = SurfacePoints(sp_coords=np.random.rand(10, 3))
    orientations = Orientations(dip_positions=np.random.rand(2, 3), dip_gradients=np.random.rand(2, 3))
    grid = EngineGrid.from_regular_grid(RegularGrid(orthogonal_extent=[0, 1, 0, 1, 0, 1], regular_grid_shape=[2, 2, 2]))

    interpolation_input = InterpolationInput(
        surface_points=surface_points,
        orientations=orientations,
        grid=grid
    )

    options = InterpolationOptions.from_args(
        range=1.0,
        c_o=1.0,
        number_octree_levels=1,
        kernel_function=AvailableKernelFunctions.cubic,
        uni_degree=0
    )

    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([10]))
    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([10]),
        number_of_orientations_per_stack=np.array([2]),
        number_of_surfaces_per_stack=np.array([1]),
        masking_descriptor=[StackRelationType.ERODE]
    )

    data_descriptor = InputDataDescriptor(tensors_structure=tensor_struct, stack_structure=stack_structure)

    return interpolation_input, options, data_descriptor


def test_valid_input(basic_input):
    ii, options, dd = basic_input
    # This should pass validation (though compute_model might fail later if it's not fully mock-able)
    # But since we're testing validation, we care if it raises before calling interpolation
    # For now, we assume it's valid.
    from gempy_engine.API.model.model_api import _check_input_validity
    _check_input_validity(ii, options, dd)


def test_mismatched_surface_points(basic_input):
    ii, options, dd = basic_input
    # Change dd to expect 11 points, but ii only has 10
    new_tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([11]))
    # Need to update stack structure too otherwise it fails there first
    new_stack_struct = StacksStructure(
        number_of_points_per_stack=np.array([11]),
        number_of_orientations_per_stack=np.array([2]),
        number_of_surfaces_per_stack=np.array([1]),
        masking_descriptor=[StackRelationType.ERODE]
    )
    dd = InputDataDescriptor(tensors_structure=new_tensor_struct, stack_structure=new_stack_struct)

    with pytest.raises(GemPyEngineInputError, match="Total surface points in InterpolationInput"):
        compute_model(ii, options, dd)


def test_mismatched_orientations(basic_input):
    ii, options, dd = basic_input
    # Change dd to expect 3 orientations, but ii only has 2
    new_stack_struct = StacksStructure(
        number_of_points_per_stack=np.array([10]),
        number_of_orientations_per_stack=np.array([3]),
        number_of_surfaces_per_stack=np.array([1]),
        masking_descriptor=[StackRelationType.ERODE]
    )
    dd = InputDataDescriptor(tensors_structure=dd.tensors_structure, stack_structure=new_stack_struct)

    with pytest.raises(GemPyEngineInputError, match="Total orientations in InterpolationInput"):
        compute_model(ii, options, dd)


def test_inconsistent_stack_structure(basic_input):
    ii, options, dd = basic_input
    # dd.tensors_structure says 10 points. 
    # We make stack_structure say 9 points.
    new_stack_struct = StacksStructure(
        number_of_points_per_stack=np.array([9]),
        number_of_orientations_per_stack=np.array([2]),
        number_of_surfaces_per_stack=np.array([1]),
        masking_descriptor=[StackRelationType.ERODE]
    )
    dd = InputDataDescriptor(tensors_structure=dd.tensors_structure, stack_structure=new_stack_struct)

    with pytest.raises(GemPyEngineInputError, match="Total points in StacksStructure"):
        compute_model(ii, options, dd)


def test_mismatched_internal_orientations(basic_input):
    ii, options, dd = basic_input
    # Change dip_gradients to have different size than dip_positions
    ii.orientations.dip_gradients = np.random.rand(3, 3)

    with pytest.raises(GemPyEngineInputError, match="Orientations dip_positions .* and dip_gradients .* must have the same number of items"):
        compute_model(ii, options, dd)


def test_mismatched_surfaces_count(basic_input):
    ii, options, dd = basic_input
    # dd.tensors_structure says 1 surface (len([10]))
    # We make stack_structure say 2 surfaces.
    new_stack_struct = StacksStructure(
        number_of_points_per_stack=np.array([10]),
        number_of_orientations_per_stack=np.array([2]),
        number_of_surfaces_per_stack=np.array([2]),  # 2 surfaces
        masking_descriptor=[StackRelationType.ERODE]
    )
    # Note: StacksStructure might throw ValueError if lengths don't match its other arrays
    # But here they are all length 1.
    dd = InputDataDescriptor(tensors_structure=dd.tensors_structure, stack_structure=new_stack_struct)

    with pytest.raises(GemPyEngineInputError, match="Total surfaces in StacksStructure"):
        compute_model(ii, options, dd)

