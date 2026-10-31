"""External stacks that describe a body: no input data of their own, and a field of distances."""
from types import SimpleNamespace

import numpy as np

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure

# Inside a cell of the fully refined level, with its nearest corner further away than the radius.
SPHERE_CENTER, SPHERE_RADIUS = np.array([0.5625, 0.5625, 0.8125]), 0.05


def inside_positive(xyz):
    points = xyz.detach().cpu().numpy() if hasattr(xyz, "detach") else np.asarray(xyz)
    return SPHERE_RADIUS - np.linalg.norm(points - SPHERE_CENTER, axis=1)


def compute(body: CustomInterpolationFunctions, levels: int, units_per_stack=(1, 1)):
    grid = EngineGrid(octree_grid=RegularGrid([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [2, 2, 2]))
    points = np.array([[0.2, 0.2, 0.25], [0.8, 0.2, 0.25], [0.2, 0.8, 0.25], [0.8, 0.8, 0.25]])
    interpolation_input = InterpolationInput(
        SurfacePoints(points),
        Orientations(np.array([[0.5, 0.5, 0.25]]), np.array([[0.0, 0.0, 1.0]])),
        grid,
        np.array([1, 2, 3]),
    )
    descriptor = InputDataDescriptor(
        TensorsStructure(number_of_points_per_surface=np.array([4])),
        StacksStructure(
            number_of_points_per_stack=np.array([0, 4]),
            number_of_orientations_per_stack=np.array([0, 1]),
            number_of_surfaces_per_stack=np.array([0, 1]),
            number_of_units_per_stack=np.array(units_per_stack),
            masking_descriptor=[StackRelationType.ERODE, StackRelationType.BASEMENT],
            interp_functions_per_stack=[body, None],
        ),
    )
    options = InterpolationOptions.from_args(
        0.8660254 * 100, 35.71428571 * 100, uni_degree=0, i_res=4, gi_res=2,
        number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)
    options.number_octree_levels = levels
    options.evaluation_options.number_octree_levels_surface = levels
    options.evaluation_options.mesh_extraction = False
    return compute_model(interpolation_input, options, descriptor)


def _to_numpy(values):
    return values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)


def unit_at(solutions, level: int, point) -> float:
    octree = solutions.octrees_output[level]
    centers = _to_numpy(octree.grid.octree_grid.values)
    nearest = np.argmin(np.linalg.norm(centers - np.asarray(point), axis=1))
    return float(_to_numpy(octree.last_output_center.final_block)[nearest])


def finest_cells_touching_the_sphere(solutions) -> int:
    centers = _to_numpy(solutions.octrees_output[-1].grid.octree_grid.values)
    return int((np.linalg.norm(centers - SPHERE_CENTER, axis=1) < SPHERE_RADIUS).sum())


def test_body_without_input_data_is_a_unit_of_its_own():
    """Unit values are sliced by units, while the tensors keep slicing by surfaces with points."""
    big_sphere = CustomInterpolationFunctions(
        scalar_field_at_surface_points=np.array([0.0]),
        implicit_function=lambda xyz: inside_positive(xyz) + 0.2,
    )
    solutions = compute(big_sphere, levels=3)
    assert unit_at(solutions, 2, SPHERE_CENTER) == 1
    assert unit_at(solutions, 2, (0.1, 0.1, 0.9)) == 2
    assert unit_at(solutions, 2, (0.1, 0.1, 0.1)) == 3


def frame(relation):
    return SimpleNamespace(
        number_of_points_per_element=np.array([0, 4, 0]),
        number_of_elements_per_group=np.array([1, 1]),
        number_of_points_per_group=np.array([0, 4]),
        number_of_orientations_per_group=np.array([0, 1]),
        structural_groups=[SimpleNamespace(custom_interpolation=object(), ignored_grid_types=()),
                           SimpleNamespace(custom_interpolation=None, ignored_grid_types=())],
    ), [relation, StackRelationType.BASEMENT]


def test_descriptor_counts_a_unit_for_an_empty_erode_stack():
    stacks = InputDataDescriptor.from_structural_frame(*frame(StackRelationType.ERODE)).stack_structure
    assert stacks.number_of_surfaces_per_stack.tolist() == [0, 1]
    assert stacks.units_per_stack.tolist() == [1, 1]


def test_descriptor_counts_no_unit_for_an_empty_null_space_stack():
    stacks = InputDataDescriptor.from_structural_frame(*frame(StackRelationType.NULL_SPACE)).stack_structure
    assert stacks.number_of_surfaces_per_stack.tolist() == [0, 1]
    assert stacks.units_per_stack.tolist() == [0, 1]


def test_units_default_to_surfaces():
    stacks = StacksStructure(
        number_of_points_per_stack=np.array([4, 4]),
        number_of_orientations_per_stack=np.array([1, 1]),
        number_of_surfaces_per_stack=np.array([2, 1]),
        masking_descriptor=[StackRelationType.ERODE, StackRelationType.BASEMENT],
    )
    assert stacks.units_per_stack.tolist() == [2, 1]

    stacks.number_of_surfaces_per_stack = np.array([1, 1])  # Callers replace it after construction.
    assert stacks.units_per_stack.tolist() == [1, 1]


def test_body_between_corners_is_missed_without_the_flag():
    body = CustomInterpolationFunctions(np.array([0.0]), inside_positive)
    assert finest_cells_touching_the_sphere(compute(body, levels=6)) == 0


def test_flag_finds_a_body_between_corners():
    body = CustomInterpolationFunctions(np.array([0.0]), inside_positive, refine_near_surface=True)
    solutions = compute(body, levels=6)
    assert finest_cells_touching_the_sphere(solutions) > 0
    assert unit_at(solutions, -1, SPHERE_CENTER) == 1
