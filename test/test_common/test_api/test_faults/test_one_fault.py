from typing import Callable

import numpy as np

from gempy_engine import compute_model
from gempy_engine.API.interp_single._interp_single_feature import input_preprocess
from gempy_engine.core.data import InterpolationOptions, TensorsStructure
from gempy_engine.core.data.custom_segmentation_functions import ellipsoid_3d_factory
from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.core.data.grid import RegularGrid, Grid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.options import DualContouringMaskingOptions
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance
from gempy_engine.modules.octrees_topology.octrees_topology_interface import ValueType
from test import helper_functions_pyvista
from test.conftest import pykeops_enabled, plot_pyvista
from test.helper_functions import plot_block_and_input_2d, plot_scalar_and_input_2d


def test_one_fault_model(one_fault_model, n_oct_levels=3):

    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model

    options.compute_scalar_gradient = False
    options.dual_contouring = True
    options.dual_contouring_masking_options = DualContouringMaskingOptions.RAW
    options.dual_contouring_fancy = False

    options.number_octree_levels = n_oct_levels

    solutions: Solutions = compute_model(interpolation_input, options, structure)

    outputs: list[OctreeLevel] = solutions.octrees_output

    if check_cov := False:  # * This is in case we need to compare the covariance matrices
        last_cov = outputs[-1].outputs_centers.exported_fields.debug
        gempy_v2_cov = _covariance_for_one_fault_model_from_gempy_v2()
        diff = last_cov - gempy_v2_cov

    if plot_2d := False:
        _plot_stack_raw(interpolation_input, outputs, structure)
        _plot_stack_squeezed_mask(interpolation_input, outputs, structure)
        _plot_stack_mask_component(interpolation_input, outputs, structure)
        _plot_stack_values_block(interpolation_input, outputs, structure)

    if plot_pyvista:
        meshes_: list[DualContouringMesh] = solutions.dc_meshes
        helper_functions_pyvista.plot_pyvista(
            #solutions.octrees_output,
            dc_meshes=[meshes_[0], meshes_[1], meshes_[-1]]
        )


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

    if True:
        if pykeops_enabled is False:
            cache_array = np.save("cached_array", array_to_cache)
        cached_array = np.load("cached_array.npy")
        foo = A_matrix.sum(0).T - cached_array.sum(0)
        print(cached_array)


def test_one_fault_model_thickness(one_fault_model, n_oct_levels=2):
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

    if plot_pyvista or False:
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes
        )


def test_one_fault_model_finite_fault(one_fault_model, n_oct_levels=4):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model

    rescaling_factor = 240
    resolution = [4, 4, 4]
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
    meshes = solutions.dc_meshes  # + meshes

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

    if plot_pyvista or False:
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=meshes,
            scalar=scalar,
            v_just_points=interpolation_input.surface_points.sp_coords
        )


def test_implicit_ellipsoid_projection_on_fault(one_fault_model):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model
    structure.stack_structure.faults_input_data = None

    options.dual_contouring_masking_options = DualContouringMaskingOptions.RAW

    # rescaling_factor = 240
    # resolution = np.array([20, 4, 20])
    # extent = np.array([-500, 500., -500, 500, -450, 550]) / rescaling_factor
    # regular_grid = RegularGrid(extent, resolution)
    # grid = Grid(regular_grid.values, regular_grid=regular_grid)
    # interpolation_input.grid = grid

    solutions: Solutions = compute_model(interpolation_input, options, structure)

    fault_mesh = solutions.dc_meshes[0]

    regular_grid = solutions.octrees_output[-1].outputs_centers[0].grid.regular_grid
    resolution = regular_grid.resolution

    scalar = _implicit_3d_ellipsoid_to_slope(regular_grid.values, np.array([0, 0, 0]), np.array([1, 1, 2]))
    scalar_fault = _implicit_3d_ellipsoid_to_slope(fault_mesh.vertices, np.array([0, 0, 0]), np.array([1, 1, 2]))

    if plot_pyvista or False:
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


def _covariance_for_one_fault_model_from_gempy_v2():
    one_fault_covariance = np.load("one_fault_test_data.npy")
    return one_fault_covariance


def _plot_stack_values_block(interpolation_input, outputs, structure):
    plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, ValueType.values_block)
    plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, ValueType.values_block)
    plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, ValueType.values_block)


def _plot_stack_mask_component(interpolation_input, outputs, structure):
    plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, ValueType.mask_component)
    plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, ValueType.mask_component)
    plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, ValueType.mask_component)


def _plot_stack_squeezed_mask(interpolation_input, outputs, structure):
    plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)
    plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)
    plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)


def _plot_stack_raw(interpolation_input, outputs, structure):
    plot_scalar_and_input_2d(0, interpolation_input, outputs, structure.stack_structure)
    plot_scalar_and_input_2d(1, interpolation_input, outputs, structure.stack_structure)
    plot_scalar_and_input_2d(2, interpolation_input, outputs, structure.stack_structure)