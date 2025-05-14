import sys

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/miguel/PycharmProjects/gempy_engine'])  # ! This has to be up here

import numpy as np
import pandas as pd

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import SurfacePoints, Orientations, InterpolationOptions, TensorsStructure
from gempy_engine.core.data.grid import RegularGrid, Grid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.options import MeshExtractionMaskingOptions
from gempy_engine.core.data.solutions import Solutions


def my_func():
    interpolation_input, structure, options = one_fault_model()

    options.compute_scalar_gradient = False
    options.dual_contouring = False
    options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.RAW

    options.number_octree_levels = 8
    solutions: Solutions = compute_model(interpolation_input, options, structure)

    return


def one_fault_model():
    data_path = "../fixtures/graben_data/"

    BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=True, pykeops_enabled=True)

    centers = np.array([500, 500, -550])
    rescaling_factor = 240

    # region InterpolationInput
    orientations = pd.read_csv(data_path + "Tutorial_ch1-9a_Fault_relations_orientations.csv")
    sp = pd.read_csv(data_path + "Tutorial_ch1-9a_Fault_relations_surface_points.csv")

    sp_coords = (sp[["X", "Y", "Z"]].values - centers) / rescaling_factor
    dip_postions = (orientations[["X", "Y", "Z"]].values - centers) / rescaling_factor
    dip_gradients_ = calculate_gradient(orientations["dip"], orientations["azimuth"], orientations["polarity"])
    dip_gradients = np.vstack(dip_gradients_).T

    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_postions, dip_gradients)
    ids = np.array([0, 1, 2, 3, 4, 5])

    # resolution = [40, 2, 40]
    resolution = [4, 4, 4]
    extent = np.array([-500, 500., -500, 500, -450, 550]) / rescaling_factor
    regular_grid = RegularGrid(extent, resolution)
    grid = Grid(regular_grid.values, regular_grid=regular_grid)

    interpolation_input = InterpolationInput(spi, ori, grid, ids)
    # endregion

    # region Structure

    faults_relations = np.array(
        [[False, False, True],
         [False, False, False],
         [False, False, False]
         ]
    )

    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([9, 24, 37]),
        number_of_orientations_per_stack=np.array([1, 4, 6]),
        number_of_surfaces_per_stack=np.array([1, 2, 3]),
        masking_descriptor=[StackRelationType.FAULT, StackRelationType.ERODE, StackRelationType.ERODE, StackRelationType.BASEMENT],
        faults_relations=faults_relations
    )

    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([9, 12, 12, 13, 12, 12]))
    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)
    # endregion

    # region InterpolationOptions

    # range_ = 1732 / rescaling_factor
    # c_o = 71428.57 / rescaling_factor

    range_ = 7 ** 2  # ? Since we are not getting the square root should we also square this? 
    c_o = 1

    options = InterpolationOptions.from_args(
        range_, c_o,
        uni_degree=1,
        number_dimensions=3,
        kernel_function=AvailableKernelFunctions.exponential)

    # endregion

    return interpolation_input, input_data_descriptor, options


def calculate_gradient(dip, az, pol):
    """Calculates the gradient from dip, azimuth and polarity values."""
    g_x = np.sin(np.deg2rad(dip)) * np.sin(np.deg2rad(az)) * pol
    g_y = np.sin(np.deg2rad(dip)) * np.cos(np.deg2rad(az)) * pol
    g_z = np.cos(np.deg2rad(dip)) * pol
    return g_x, g_y, g_z


if __name__ == '__main__':
    my_func()
