from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from gempy_engine.API.interp_single._interp_single_feature import input_preprocess
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.custom_segmentation_functions import ellipsoid_3d_factory, _implicit_3d_ellipsoid_to_slope
from gempy_engine.core.data.grid import Grid, RegularGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor, StackRelationType, TensorsStructure, StacksStructure
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.options import DualContouringMaskingOptions
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level, ValueType
from test import helper_functions_pyvista
from test.conftest import TEST_SPEED, pykeops_enabled
from test.helper_functions import plot_block, plot_2d_scalar_y_direction

PLOT = False


# noinspection PyUnreachableCode
def test_graben_fault_model(graben_fault_model):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = graben_fault_model

    options.compute_scalar_gradient = False
    options.dual_contouring = False

    options.number_octree_levels = 1
    solutions: Solutions = compute_model(interpolation_input, options, structure)

    outputs: list[OctreeLevel] = solutions.octrees_output
    if True:
        plot_scalar_and_input_2d(0, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(1, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(2, interpolation_input, outputs, structure.stack_structure)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.ids)


# noinspection PyUnreachableCode
def test_graben_fault_model_thickness(graben_fault_model):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = graben_fault_model

    options.compute_scalar_gradient = False
    options.dual_contouring = True

    fault_data: FaultsData = FaultsData.from_user_input(thickness=.2)
    fault_data2: FaultsData = FaultsData.from_user_input(thickness=.2)
    structure.stack_structure.faults_input_data = [fault_data, fault_data2, None]

    options.number_octree_levels = 4
    solutions: Solutions = compute_model(interpolation_input, options, structure)

    outputs: list[OctreeLevel] = solutions.octrees_output

    if True:
        plot_scalar_and_input_2d(0, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(1, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(2, interpolation_input, outputs, structure.stack_structure)

    if False:
        plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)
        plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)

    if False:
        plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, ValueType.mask_component)
        plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, ValueType.mask_component)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, ValueType.mask_component)

    if False:
        plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.values_block)
        plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.values_block)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.values_block)

    if True:
        # plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.ids)
        # plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.ids)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.ids)

    if True:
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes
        )


# noinspection PyUnreachableCode
def test_graben_fault_model_offset(graben_fault_model):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = graben_fault_model

    options.compute_scalar_gradient = False
    options.dual_contouring = False

    fault_data: FaultsData = FaultsData.from_user_input(thickness=None, offset=50)
    structure.stack_structure.faults_input_data = [fault_data, None, None]

    solutions: Solutions = compute_model(interpolation_input, options, structure)

    outputs: list[OctreeLevel] = solutions.octrees_output

    if True:
        plot_scalar_and_input_2d(0, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(1, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(2, interpolation_input, outputs, structure.stack_structure)

    if True:
        plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.values_block)
        plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.values_block)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.values_block)

    if True:
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.ids)


def test_one_fault_model_pykeops(one_fault_model):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model

    i = 1
    structure.stack_structure.stack_number = i
    interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(
        interpolation_input, structure.stack_structure)

    tensor_struct_i: TensorsStructure = TensorsStructure.from_tensor_structure_subset(structure, i)

    solver_input = input_preprocess(tensor_struct_i, interpolation_input_i)
    
    A_matrix = yield_covariance(solver_input, options.kernel_options)
    array_to_cache = A_matrix

    if pykeops_enabled is False:
        cache_array = np.save("cached_array", array_to_cache)
    cached_array = np.load("cached_array.npy")
    foo = A_matrix.sum(0).T - cached_array.sum(0)
    print(cached_array)


# noinspection PyUnreachableCode
def test_one_fault_model(one_fault_model, n_oct_levels=8):
    """
    300 MB 4 octree levels and no gradient
    
    """

    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model

    options.compute_scalar_gradient = False
    options.dual_contouring = False
    options.dual_contouring_masking_options = DualContouringMaskingOptions.DISJOINT

    options.number_octree_levels = n_oct_levels
    solutions: Solutions = compute_model(interpolation_input, options, structure)

    outputs: list[OctreeLevel] = solutions.octrees_output

    array_to_cache = outputs[-1].outputs_centers[1].exported_fields.debug

    if pykeops_enabled is False:
        cache_array = np.save("cached_array", array_to_cache)
    cached_array = np.load("cached_array.npy")

    if False:  # * This is in case we need to compare the covariance matrices

        last_cov = outputs[-1].outputs_centers.exported_fields.debug
        gempy_v2_cov = covariance_for_one_fault_model_from_gempy_v2()
        diff = last_cov - gempy_v2_cov

    if False:
        plot_scalar_and_input_2d(0, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(1, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(2, interpolation_input, outputs, structure.stack_structure)

    if False:
        plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)
        plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)

    if False:
        plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, ValueType.mask_component)
        plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, ValueType.mask_component)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, ValueType.mask_component)

    if False:
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure)

    if False:
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes
        )


def test_one_fault_model_thickness(one_fault_model, n_oct_levels=3):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model

    fault_data: FaultsData = FaultsData.from_user_input(thickness=.5)
    structure.stack_structure.faults_input_data = [fault_data, None, None]
    options.dual_contouring = True
    options.dual_contouring_masking_options = DualContouringMaskingOptions.DISJOINT

    options.number_octree_levels = n_oct_levels
    solutions: Solutions = compute_model(interpolation_input, options, structure)

    # TODO: Grab second scalar and create fault kernel
    outputs: list[OctreeLevel] = solutions.octrees_output

    if False:
        plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)
        plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)

    if True:
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes
        )


def project_xyz_coordinates_to_plane_defined_by_set_of_coords(xyz_to_project: np.ndarray, xyz_defining_plane: np.ndarray):
    """
    Find best fit plane using SVD to xyz_defining_plane and project xyz_to_project onto it.
    """
    xyz_to_project = xyz_to_project.astype(np.float64)
    xyz_defining_plane = xyz_defining_plane.astype(np.float64)

    # Find best fit plane using SVD to xyz_defining_plane and project xyz_to_project onto it.
    u, s, vh = np.linalg.svd(xyz_defining_plane - xyz_to_project.mean(0))
    xyz_projected = np.dot(xyz_to_project - xyz_to_project.mean(0), vh.T)
    return xyz_projected
    

def test_data_rotation_for_final_faults(one_fault_model, n_oct_levels=1):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model
    rescaling_factor = 240
    resolution = [20, 4, 20]
    extent = np.array([-500, 500., -500, 500, -450, 550]) / rescaling_factor
    regular_grid = RegularGrid(extent, resolution)
    grid = Grid(regular_grid.values, regular_grid=regular_grid)
    interpolation_input.grid = grid

    structure.stack_structure.stack_number = 0
    interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(interpolation_input,
                                                                                                   structure.stack_structure)

    fault_sp_coord = interpolation_input_i.surface_points.sp_coords
    grid_coord = grid.values

    rotated_coords = project_xyz_coordinates_to_plane_defined_by_set_of_coords(grid_coord, fault_sp_coord)

    if True:
        import pyvista as pv
        p = pv.Plotter()
        p.add_mesh(pv.PolyData(grid_coord), color="b", point_size=1.0, render_points_as_spheres=False)
        p.add_mesh(pv.PolyData(fault_sp_coord), color="g", point_size=10.0, render_points_as_spheres=True)
        p.add_mesh(pv.PolyData(rotated_coords), color="r", point_size=2.0, render_points_as_spheres=False)
        p.show()


def test_implicit_ellipsoid_projection_on_fault(one_fault_model):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model
    
    options.dual_contouring_masking_options = DualContouringMaskingOptions.RAW

    rescaling_factor = 240
    resolution = np.array([20, 4, 20])
    extent = np.array([-500, 500., -500, 500, -450, 550]) / rescaling_factor
    regular_grid = RegularGrid(extent, resolution)
    grid = Grid(regular_grid.values, regular_grid=regular_grid)
    interpolation_input.grid = grid

    solutions: Solutions = compute_model(interpolation_input, options, structure)
    
    fault_mesh = solutions.dc_meshes[0]
    scalar = _implicit_3d_ellipsoid_to_slope(regular_grid.values, np.array([0, 0, 0]), np.array([1, 1, 2]))
    scalar_fault = _implicit_3d_ellipsoid_to_slope(fault_mesh.vertices, np.array([0, 0, 0]), np.array([1, 1, 2]))
    
    if True:
        import pyvista as pv
        p = pv.Plotter()
        regular_grid_values = regular_grid.values_vtk_format

        grid_3d = regular_grid_values.reshape(*(resolution + 1), 3).T
        regular_grid_mesh = pv.StructuredGrid(*grid_3d)
        regular_grid_mesh["lith"] = scalar
        p.add_mesh(regular_grid_mesh, show_edges=False, opacity=.5)

        dual_mesh = pv.PolyData(fault_mesh.vertices, np.insert(fault_mesh.edges, 0, 3, axis=1).ravel())
        dual_mesh["bar"] = scalar_fault
        p.add_mesh(dual_mesh, opacity=1, silhouette=True, show_edges=True)
        
        p.show()
    

def test_implicit_ellipsoid():
    rescaling_factor = 240
    resolution = np.array([20, 4, 20])
    extent = np.array([-500, 500., -500, 500, -450, 550]) / rescaling_factor
    regular_grid = RegularGrid(extent, resolution)
    
    scalar = _implicit_3d_ellipsoid_to_slope(regular_grid.values, np.array([0, 0, 0]), np.array([1, 1, 2]))
    if True:
        import pyvista as pv
        p = pv.Plotter()
        regular_grid_values = regular_grid.values_vtk_format
        shape = regular_grid_values.shape

        grid_3d = regular_grid_values.reshape(*(resolution + 1), 3).T
        regular_grid_mesh = pv.StructuredGrid(*grid_3d)
        regular_grid_mesh["lith"] = scalar
        p.add_mesh(regular_grid_mesh, show_edges=False, opacity=.5)
        p.show()


def test_transforming_implicit_ellipsoid():
    rescaling_factor = 240
    resolution = np.array([20, 20, 20])
    extent = np.array([-500, 500., -500, 500, -450, 550]) / rescaling_factor
    regular_grid = RegularGrid(extent, resolution)
    
    xyz = regular_grid.values
    center = np.array([0, 0, 0])
    radius = np.array([1, 1, 2])*1
    f = 1
    f2 = 1
    scalar = - np.sum((xyz - center) ** 2.00 / (radius ** 2),  axis=1) - 1.0
    #scalar = (((xyz[:, 0] - center[0])) ** 2) / (radius[0] ** 2)
    scalar = scalar - scalar.min()

    sigmoid_slope = 10
    Z_x = scalar
    drift_0 = 8
    scale_0 = 1000
    scalar = scale_0 / (1 + np.exp(-sigmoid_slope * (Z_x - drift_0)))
    
    if False:
        plt.plot(xyz[:, 0], scalar)
        plt.show()

        plt.plot(xyz[:, 0], foo)
        plt.show()
    
    # max_slope = 1000
    # min_slope = 0
    # scalar_slope = (scalar - scalar.min()) / (scalar.max() - scalar.min()) * (max_slope - min_slope) + min_slope
    #
    # # cap scalar
    # scalar[scalar > 10] = 10
    #
    # # map scalar between 0 and 1 but heavily skewed with high values
    # scalar2 = np.power(scalar, 15)
    # scalar_slope2 = (scalar2 - scalar2.min()) / (scalar2.max() - scalar2.min()) * (max_slope - min_slope) + min_slope
    #
    if False:
        plt.hist(scalar, bins=100)
        plt.show()
        # plt.hist(scalar_slope)
        # plt.show()
        # plt.hist(scalar2, log=True, bins=100)
        # plt.show()
        # plt.hist(scalar_slope2, log=True, bins=100)
        # plt.show()
        # 

    if True:
        import pyvista as pv
        
        p = pv.Plotter()
        regular_grid_values = regular_grid.values_vtk_format
        shape = regular_grid_values.shape
        
        grid_3d = regular_grid_values.reshape(*(resolution + 1), 3).T
        regular_grid_mesh = pv.StructuredGrid(*grid_3d)
        regular_grid_mesh["lith"] = scalar
        regular_grid_mesh = regular_grid_mesh.threshold([10, 10000])

        p.add_mesh(regular_grid_mesh, show_edges=False, opacity=.5)
        p.show()
   
   
def test_one_fault_model_finite_fault(one_fault_model, n_oct_levels=4):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model

    rescaling_factor = 240
    resolution = [4,4,4]
    extent = np.array([-500, 500., -500, 500, -450, 550]) / rescaling_factor
    regular_grid = RegularGrid(extent, resolution)
    grid = Grid(regular_grid.values, regular_grid=regular_grid)
    interpolation_input.grid = grid
    options.number_octree_levels = n_oct_levels

    options.compute_scalar_gradient = False
    options.dual_contouring = True
    options.dual_contouring_masking_options = DualContouringMaskingOptions.RAW
    
    # region no faults
    # faults_relations = np.array(
    #     [[False, False, False],
    #      [False, False, False],
    #      [False, False, False]
    #      ]
    # )
    # structure.stack_structure.faults_relations = faults_relations
    # solutions: Solutions = compute_model(interpolation_input, options, structure)
    # meshes = solutions.dc_meshes
    # endregion
    
    # region finite fault
    faults_relations = np.array(
        [[False, True, True],
         [False, False, False],
         [False, False, False]
         ]
    )
    structure.stack_structure.faults_relations = faults_relations
    f1_finite_fault: Callable = ellipsoid_3d_factory(np.array([0, 0, 0]), np.array([3, 1, 1]), 1000, 0.001)
    structure.stack_structure.segmentation_functions_per_stack = [f1_finite_fault, None, None]
    
    solutions: Solutions = compute_model(interpolation_input, options, structure)
    meshes = solutions.dc_meshes #+ meshes

    outputs: list[OctreeLevel] = solutions.octrees_output
    scalar = f1_finite_fault(solutions.octrees_output[-1].grid_centers.regular_grid.values)
    # endregion
    
    if True:
        plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, ValueType.values_block)
        plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, ValueType.values_block)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, ValueType.values_block)
        
    if True:
        plot_scalar_and_input_2d(0, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(1, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(2, interpolation_input, outputs, structure.stack_structure)
        
    if True:
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=meshes,
            scalar = scalar,
            v_just_points=interpolation_input.surface_points.sp_coords
        )


def plot_scalar_and_input_2d(foo, interpolation_input, outputs: list[OctreeLevel], structure: StacksStructure):
    structure.stack_number = foo

    regular_grid_scalar = get_regular_grid_value_for_level(outputs, value_type=ValueType.scalar, scalar_n=foo)
    grid: Grid = outputs[-1].grid_centers

    interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(interpolation_input, structure)
    plot_2d_scalar_y_direction(interpolation_input_i, regular_grid_scalar, grid.regular_grid)


def plot_block_and_input_2d(stack_number, interpolation_input, outputs: list[OctreeLevel], structure: StacksStructure,
                            value_type=ValueType.ids):
    from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level

    regular_grid_scalar = get_regular_grid_value_for_level(outputs, value_type=value_type, scalar_n=stack_number)
    grid: Grid = outputs[-1].grid_centers

    structure.stack_number = stack_number
    interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(interpolation_input, structure)
    plot_block(regular_grid_scalar, grid.regular_grid, interpolation_input_i)


def covariance_for_one_fault_model_from_gempy_v2():
    one_fault_covariance = np.load("one_fault_test_data.npy")
    return one_fault_covariance
