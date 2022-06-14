from typing import List

import numpy as np

from gempy_engine.API.interp_single._multi_scalar_field_manager import _interpolate_stack
from gempy_engine.API.interp_single.interp_features import interpolate_all_fields_no_octree
from gempy_engine.core.data.exported_structs import InterpOutput, ExportedFields, ScalarFieldOutput
from test.helper_functions import plot_block


def test_final_block(unconformity_complex):
    interpolation_input, options, structure = unconformity_complex
    outputs: List[InterpOutput] = interpolate_all_fields_no_octree(interpolation_input, options, structure)

    if True:
        grid = interpolation_input.grid.regular_grid
        # plot_block(outputs[0].final_block, grid)
        # plot_block(outputs[1].final_block, grid)
        plot_block(outputs[2].final_block, grid)


def test_compute_mask_components_all_erode_implicit_sphere(unconformity_complex_implicit):
    """Plot each individual mask compontent"""
    # TODO:
    interpolation_input, options, structure = unconformity_complex_implicit
    outputs: List[ScalarFieldOutput] = _interpolate_stack(structure, interpolation_input, options)

    if False:
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].mask_components.mask_lith, grid)
        plot_block(outputs[1].mask_components.mask_lith, grid)
        plot_block(outputs[2].mask_components.mask_lith, grid)
        plot_block(outputs[3].mask_components.mask_lith, grid)


def test_final_block_implicit(unconformity_complex_implicit):
    interpolation_input, options, structure = unconformity_complex_implicit
    outputs: List[InterpOutput] = interpolate_all_fields_no_octree(interpolation_input, options, structure)

    if True:
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].final_block, grid)
        plot_block(outputs[1].final_block, grid)
        plot_block(outputs[2].final_block, grid)
        plot_block(outputs[3].final_block, grid)




def test_implicit_function(unconformity_complex):
    from gempy_engine.modules.activator.activator_interface import activate_formation_block_from_args
    from gempy_engine.API.interp_single._interp_single_feature import _segment_scalar_field
    
    interpolation_input, options, structure = unconformity_complex
    grid = interpolation_input.grid.regular_grid
    xyz = grid.values
    scalar = implicit_sphere(xyz, grid.extent)
    
    exported_fields = ExportedFields(scalar, _scalar_field_at_surface_points=np.array([20]))
    values_block = _segment_scalar_field(exported_fields, np.array([0, 1]))

    plot_block(scalar, grid)
    plot_block(values_block, grid)

# region implicit functions ======================================================
def goursat_tangle(xyz: np.ndarray):
    a, b, c = 0.0, -5.0, 11.8
    return a * xyz[:, 0] ** 2 + b * xyz[:, 1] ** 2 + c * xyz[:, 2] ** 2


def implicit_sphere(xyz: np.ndarray, extent: np.ndarray):
    x_dir = np.minimum(xyz[:, 0] - extent[0], extent[1] - xyz[:, 0])
    y_dir = np.minimum(xyz[:, 1] - extent[2], extent[3] - xyz[:, 1])
    z_dir = np.minimum(xyz[:, 2] - extent[4], extent[5] - xyz[:, 2])
    return x_dir ** 2 + y_dir ** 2 + z_dir ** 2
# endregion
    