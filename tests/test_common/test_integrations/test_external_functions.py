from typing import List

import numpy as np
import pytest

from gempy_engine.API.interp_single._multi_scalar_field_manager import _interpolate_stack
from gempy_engine.API.interp_single.interp_features import interpolate_all_fields_no_octree
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.core.data.scalar_field_output import ScalarFieldOutput
from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.core.data.options import MeshExtractionMaskingOptions
from gempy_engine.modules.activator.activator_interface import activate_formation_block
from gempy_engine.plugins.plotting import helper_functions_pyvista
from tests.conftest import TEST_SPEED
from gempy_engine.plugins.plotting.helper_functions import plot_block

PLOT = False


def test_compute_mask_components_all_erode_implicit_sphere(unconformity_complex_implicit):
    """Plot each individual mask compontent"""
    # TODO:
    interpolation_input, options, structure = unconformity_complex_implicit
    outputs: List[ScalarFieldOutput] = _interpolate_stack(structure, interpolation_input, options)

    if PLOT or False:
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].mask_components_erode, grid)
        plot_block(outputs[1].mask_components_erode, grid)
        plot_block(outputs[2].mask_components_erode, grid)


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
    def implicit_sphere(xyz: np.ndarray, extent: np.ndarray):
        x_dir = np.minimum(xyz[:, 0] - extent[0], extent[1] - xyz[:, 0])
        y_dir = np.minimum(xyz[:, 1] - extent[2], extent[3] - xyz[:, 1])
        z_dir = np.minimum(xyz[:, 2] - extent[4], extent[5] - xyz[:, 2])
        return x_dir ** 2 + y_dir ** 2 + z_dir ** 2

    interpolation_input, options, structure = unconformity_complex
    grid = interpolation_input.grid.octree_grid 
    xyz = grid.values
    scalar = implicit_sphere(xyz, grid.orthogonal_extent)

    from gempy_engine.core.backend_tensor import BackendTensor
    exported_fields = ExportedFields(
        _scalar_field=BackendTensor.t.array(scalar),
        _scalar_field_at_surface_points=BackendTensor.t.array([20])
    )
    values_block = activate_formation_block(exported_fields, np.array([0, 1]), 100000)

    if PLOT or False:
        plot_block(scalar, grid)
        plot_block(values_block, grid)


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_dual_contouring_multiple_independent_fields(unconformity_complex_implicit, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex_implicit
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT

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
            gradient_pos=intersection_xyz, gradients=gradients,  # * Uncomment for more detailed plots
            a=center_mass, b=normals
        )
