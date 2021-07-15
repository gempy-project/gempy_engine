import pandas as pd
import numpy as np
import pytest
import os

from gempy_engine.core.data import TensorsStructure, InterpolationOptions, SurfacePoints, \
    Orientations
from gempy_engine.core.data.grid import RegularGrid, Grid
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from test.helper_functions import calculate_gradient

dir_name = os.path.dirname(__file__)
data_path = dir_name + "/simple_geometries/"



@pytest.fixture(scope="session")
def horizontal_stratigraphic():
    orientations = pd.read_csv(data_path+"model1_orientations.csv")
    sp = pd.read_csv(data_path+"model1_surface_points.csv")

    sp_coords = sp[["X", "Y", "Z"]].values
    dip_postions = orientations[["X", "Y", "Z"]].values
    dip_gradients_ = calculate_gradient(orientations["dip"],
                                       orientations["azimuth"],
                                       orientations[ "polarity"])
    dip_gradients = np.vstack(dip_gradients_).T
    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([6, 6]))

    range_ = 1732
    options = InterpolationOptions(range_, range_ ** 2 / 14 / 3, 0, i_res=4, gi_res=2,
                               number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)

    resolution = [50, 50, 50]
    extent = [0,1000,0,1000,0,1000]

    regular_grid = RegularGrid(extent, resolution)
    grid = Grid.from_regular_grid(regular_grid)

    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_postions, dip_gradients)
    ids = np.array([0,1,2])

    interpolation_input = InterpolationInput(spi, ori, grid, ids)
    return interpolation_input, options, tensor_struct


@pytest.fixture(scope="session")
def horizontal_stratigraphic_scaled():
    orientations = pd.read_csv(data_path+"model1_orientations_scaled.csv")
    sp = pd.read_csv(data_path+"model1_surface_points_scaled.csv")

    sp_coords = sp[["X", "Y", "Z"]].values
    dip_postions = orientations[["X", "Y", "Z"]].values
    dip_gradients_ = calculate_gradient(orientations["dip"],
                                       orientations["azimuth"],
                                       orientations[ "polarity"])
    dip_gradients = np.vstack(dip_gradients_).T
    tensor_struct = TensorsStructure(number_of_points_per_surface=np.array([6, 6]))

    range_ = 1.083
    c_o = 44.643
    i_r = 4
    gi_r = 2

    # options = InterpolationOptions(range_, range_ ** 2 / 14 / 3, 0, i_res=1, gi_res=1,
    #                            number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)

    options = InterpolationOptions(range_, c_o, 0, i_res=i_r, gi_res=gi_r,
                                   number_dimensions=3,
                                   kernel_function=AvailableKernelFunctions.cubic)

    resolution = [50, 50, 50]
    extent = [0,1,0,1,0,1]

    regular_grid = RegularGrid(extent, resolution)


    g = np.load(data_path+"model1_scaled.npy")
    grid = Grid(g, regular_grid=regular_grid)

    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_postions, dip_gradients)
    ids = np.array([0,1,2])

    interpolation_input = InterpolationInput(spi, ori, grid, ids)
    return interpolation_input, options, tensor_struct



"""
 [[ 5.333e+02  4.189e+02  0.000e+00  0.000e+00  0.000e+00  0.000e+00     -1.301e+02 -2.603e+02  0.000e+00 -1.301e+02 -2.603e+02 -1.209e+02 -2.419e+02  0.000e+00 -1.209e+02 -2.419e+02]
 [ 4.189e+02  5.333e+02  0.000e+00  0.000e+00  0.000e+00  0.000e+00     -1.209e+02 -2.419e+02 -1.137e-13 -1.209e+02 -2.419e+02 -1.301e+02 -2.603e+02 -2.842e-14 -1.301e+02 -2.603e+02]
 [ 0.000e+00  0.000e+00  5.333e+02  4.189e+02  0.000e+00  0.000e+00      3.872e+01  0.000e+00 -1.952e+02 -2.339e+02 -1.952e+02  3.346e+01  0.000e+00 -1.814e+02 -2.149e+02 -1.814e+02]
 [ 0.000e+00  0.000e+00  4.189e+02  5.333e+02  0.000e+00  0.000e+00      3.346e+01 -8.527e-14 -1.814e+02 -2.149e+02 -1.814e+02  3.872e+01 -1.421e-13 -1.952e+02 -2.339e+02 -1.952e+02]
 [ 0.000e+00  0.000e+00  0.000e+00  0.000e+00  5.333e+02  3.065e+02      0.000e+00  0.000e+00  0.000e+00  0.000e+00  0.000e+00  2.231e+01  0.000e+00  0.000e+00  2.231e+01 -7.105e-14]
 [ 0.000e+00  0.000e+00  0.000e+00  0.000e+00  3.065e+02  5.333e+02     -2.231e+01  5.684e-14  5.684e-14 -2.231e+01  1.279e-13  0.000e+00  0.000e+00  0.000e+00  0.000e+00  0.000e+00]
 
 [-1.301e+02 -1.209e+02  3.872e+01  3.346e+01  0.000e+00 -2.231e+01      9.566e+01  1.252e+02  2.448e+01  7.118e+01  8.958e+01  8.574e+01  1.150e+02  2.105e+01  6.469e+01  8.187e+01]
 [-2.603e+02 -2.419e+02  0.000e+00 -8.527e-14  0.000e+00  5.684e-14      1.252e+02  2.505e+02  6.014e+01  1.252e+02  1.904e+02  1.150e+02  2.300e+02  5.418e+01  1.150e+02  1.758e+02]
 [ 0.000e+00 -1.137e-13 -1.952e+02 -1.814e+02  0.000e+00  5.684e-14      2.448e+01  6.014e+01  1.762e+02  1.517e+02  1.160e+02  2.105e+01  5.418e+01  1.605e+02  1.395e+02  1.063e+02]
 [-1.301e+02 -1.209e+02 -2.339e+02 -2.149e+02  0.000e+00 -2.231e+01      7.118e+01  1.252e+02  1.517e+02  2.229e+02  2.168e+02  6.469e+01  1.150e+02  1.395e+02  2.042e+02  2.003e+02]
 [-2.603e+02 -2.419e+02 -1.952e+02 -1.814e+02  0.000e+00  1.279e-13      8.958e+01  1.904e+02  1.160e+02  2.168e+02  3.064e+02  8.187e+01  1.758e+02  1.063e+02  2.003e+02  2.821e+02]
 [-1.209e+02 -1.301e+02  3.346e+01  3.872e+01  2.231e+01  0.000e+00      8.574e+01  1.150e+02  2.105e+01  6.469e+01  8.187e+01  9.566e+01  1.252e+02  2.448e+01  7.118e+01  8.958e+01]
 [-2.419e+02 -2.603e+02  0.000e+00 -1.421e-13  0.000e+00  0.000e+00      1.150e+02  2.300e+02  5.418e+01  1.150e+02  1.758e+02  1.252e+02  2.505e+02  6.014e+01  1.252e+02  1.904e+02]
 [ 0.000e+00 -2.842e-14 -1.814e+02 -1.952e+02  0.000e+00  0.000e+00      2.105e+01  5.418e+01  1.605e+02  1.395e+02  1.063e+02  2.448e+01  6.014e+01  1.762e+02  1.517e+02  1.160e+02]
 [-1.209e+02 -1.301e+02 -2.149e+02 -2.339e+02  2.231e+01  0.000e+00      6.469e+01  1.150e+02  1.395e+02  2.042e+02  2.003e+02  7.118e+01  1.252e+02  1.517e+02  2.229e+02  2.168e+02]
 [-2.419e+02 -2.603e+02 -1.814e+02 -1.952e+02 -7.105e-14  0.000e+00      8.187e+01  1.758e+02  1.063e+02  2.003e+02  2.821e+02  8.958e+01  1.904e+02  1.160e+02  2.168e+02  3.064e+02]]

"""