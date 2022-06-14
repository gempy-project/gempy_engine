from typing import List

import numpy as np
import pytest

from gempy_engine.API.interp_single._multi_scalar_field_manager import _interpolate_stack
from gempy_engine.API.interp_single.interp_features import interpolate_all_fields_no_octree
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data.exported_structs import InterpOutput, ExportedFields, ScalarFieldOutput, Solutions
from gempy_engine.core.data.options import DualContouringMaskingOptions
from test import helper_functions_pyvista
from test.conftest import TEST_SPEED
from test.helper_functions import plot_block

PLOT = False


def test_compute_mask_components_all_erode_implicit_sphere(unconformity_complex_implicit):
    """Plot each individual mask compontent"""
    # TODO:
    interpolation_input, options, structure = unconformity_complex_implicit
    outputs: List[ScalarFieldOutput] = _interpolate_stack(structure, interpolation_input, options)

    if PLOT or False:
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].mask_components.mask_lith, grid)
        plot_block(outputs[1].mask_components.mask_lith, grid)
        plot_block(outputs[2].mask_components.mask_lith, grid)
        plot_block(outputs[3].mask_components.mask_lith, grid)


def test_final_block_implicit(unconformity_complex_implicit):
    interpolation_input, options, structure = unconformity_complex_implicit
    outputs: List[InterpOutput] = interpolate_all_fields_no_octree(interpolation_input, options, structure)

    if PLOT or False:
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].final_block, grid)
        plot_block(outputs[1].final_block, grid)
        plot_block(outputs[2].final_block, grid)
        plot_block(outputs[3].final_block, grid)


def test_implicit_function(unconformity_complex):
    from gempy_engine.API.interp_single._interp_single_feature import _segment_scalar_field

    def implicit_sphere(xyz: np.ndarray, extent: np.ndarray):
        x_dir = np.minimum(xyz[:, 0] - extent[0], extent[1] - xyz[:, 0])
        y_dir = np.minimum(xyz[:, 1] - extent[2], extent[3] - xyz[:, 1])
        z_dir = np.minimum(xyz[:, 2] - extent[4], extent[5] - xyz[:, 2])
        return x_dir ** 2 + y_dir ** 2 + z_dir ** 2

    interpolation_input, options, structure = unconformity_complex
    grid = interpolation_input.grid.regular_grid
    xyz = grid.values
    scalar = implicit_sphere(xyz, grid.extent)

    exported_fields = ExportedFields(scalar, _scalar_field_at_surface_points=np.array([20]))
    values_block = _segment_scalar_field(exported_fields, np.array([0, 1]))

    if PLOT or False:
        plot_block(scalar, grid)
        plot_block(values_block, grid)


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_dual_contouring_multiple_independent_fields(unconformity_complex_implicit, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex_implicit
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.dual_contouring_masking_options = DualContouringMaskingOptions.DISJOINT

    solutions: Solutions = compute_model(interpolation_input, options, structure)

    if PLOT or False:
        dc_data = solutions.dc_meshes[0].dc_data  # * Scalar field where to show gradients
        intersection_xyz = dc_data.xyz_on_edge
        gradients = dc_data.gradients

        center_mass = dc_data.bias_center_mass
        normals = dc_data.bias_normals

        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes,
            xyz_on_edge=intersection_xyz, gradients=gradients,  # * Uncomment for more detailed plots
            a=center_mass, b=normals
        )
