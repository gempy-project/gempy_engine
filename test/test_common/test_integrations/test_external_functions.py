from typing import List

import numpy as np

from gempy_engine.API.interp_single._interp_single_feature import _segment_scalar_field
from gempy_engine.API.interp_single.interp_features import interpolate_all_fields_no_octree
from gempy_engine.core.data.exported_structs import InterpOutput, ExportedFields
from test.helper_functions import plot_block


def test_final_block(unconformity_complex):
    interpolation_input, options, structure = unconformity_complex
    outputs: List[InterpOutput] = interpolate_all_fields_no_octree(interpolation_input, options, structure)

    if True:
        grid = interpolation_input.grid.regular_grid
        # plot_block(outputs[0].final_block, grid)
        # plot_block(outputs[1].final_block, grid)
        plot_block(outputs[2].final_block, grid)


def test_implicit_function(unconformity_complex):
    from gempy_engine.modules.activator.activator_interface import activate_formation_block_from_args
    
    interpolation_input, options, structure = unconformity_complex
    grid = interpolation_input.grid.regular_grid
    xyz = grid.values
    scalar = implicit_sphere(xyz, grid.extent)

    plot_block(scalar, grid)
    
    values_block = activate_formation_block_from_args(scalar, np.array([0, 1]), np.array([20]), 50000)
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
    