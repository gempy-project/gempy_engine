import pandas as pd
import numpy as np
import pytest

from gempy_engine.core.data.data_shape import TensorsStructure
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.modules.covariance._input_preparation import orientations_preprocess, surface_points_preprocess
from gempy_engine.modules.covariance._structs import SurfacePointsInternals, OrientationsInternals


@pytest.fixture(scope='session')
def simple_model():
    spi = SurfacePointsInternals(
        ref_surface_points=np.array(
            [[4, 0],
             [4, 0],
             [4, 0],
             [3, 3],
             [3, 3]]),
        rest_surface_points=np.array([[0, 0],
                                      [2, 0],
                                      [3, 0],
                                      [0, 2],
                                      [2, 2]]),
        nugget_effect_ref_rest=0
    )

    ori_i = Orientations(
        dip_positions=np.array([[0, 6],
                                [2, 13]]),
        nugget_effect_grad=0.0000001
    )
    ori_int = orientations_preprocess(ori_i)

    kri = InterpolationOptions(5, 5 ** 2 / 14 / 3, 0, i_res=1, gi_res=1,
                               number_dimensions=2)

    return spi, ori_int, kri


@pytest.fixture(scope='session')
def simple_model_2():
    sp_coords = np.array([[4, 0],
                          [0, 0],
                          [2, 0],
                          [3, 0],
                          [3, 3],
                          [0, 2],
                          [2, 2]])

    nugget_effect_scalar = 0
    spi = SurfacePoints(sp_coords, nugget_effect_scalar)

    dip_positions = np.array([[0, 6],
                              [2, 13]])

    nugget_effect_grad = 0.0000001
    ori_i = Orientations(dip_positions, nugget_effect_grad)

    kri = InterpolationOptions(5, 5 ** 2 / 14 / 3, 0, i_res=1, gi_res=1,
                               number_dimensions=2)

    _ = np.ones(3)
    tensor_structure = TensorsStructure(np.array([3, 2]), _, _, _, _)

    return spi, ori_i, kri, tensor_structure


@pytest.fixture(scope='session')
def moureze():
    # %%
    # Loading surface points from repository:
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # With pandas we can do it directly from the web and with the right args
    # we can directly tidy the data in gempy style:
    #

    # %%
    Moureze_points = pd.read_csv(
        'https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Moureze_Points.csv', sep=';',
        names=['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', '_'], header=0, )
    Sections_EW = pd.read_csv(
        'https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Sections_EW.csv',
        sep=';',
        names=['X', 'Y', 'Z', 'ID', '_'], header=1).dropna()
    Sections_NS = pd.read_csv(
        'https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Sections_NS.csv',
        sep=';',
        names=['X', 'Y', 'Z', 'ID', '_'], header=1).dropna()

    # %%
    # Extracting the orientatins:
    #

    # %%
    mask_surfpoints = Moureze_points['G_x'] < -9999
    sp = Moureze_points[mask_surfpoints]
    orientations = Moureze_points[~mask_surfpoints]

    sp['smooth'] = 1e4

    return sp, orientations


@pytest.fixture(scope='session')
def moureze_sp(moureze):
    sp, ori = moureze
    sp_t = SurfacePoints(sp[['X', 'Y', 'Z']].values, sp['smooth'].values)
    return sp_t


@pytest.fixture(scope='session')
def moureze_kriging():
    return InterpolationOptions(10000, 1000)


@pytest.fixture(scope='session')
def moureze_orientations_heavy(moureze):
    _, ori = moureze
    n = 2
    ori_poss = ori[['X', 'Y', 'Z']].values,
    ori_pos = ori_poss[0]
    ori_grad = ori[['G_x', 'G_y', 'G_z']].values

    for i in range(n):
        ori_pos = np.vstack([ori_pos, ori_pos + np.array([i * 100, i * 100, i * 100])])
        ori_grad = np.vstack([ori_grad, ori_grad])

    ori_t = Orientations(ori_pos, dip_gradients=ori_grad)

    return ori_t


@pytest.fixture(scope='session')
def moureze_internals(moureze_sp, moureze_orientations_heavy, moureze_kriging):
    moureze_sp.sp_positions /= 1000

    n_points_per_surf = np.array([10, 50, moureze_sp.sp_positions.shape[0] - 60 - 3], dtype='int32')
    sp_int = surface_points_preprocess(moureze_sp, n_points_per_surf)

    moureze_orientations_heavy.dip_positions /= 1000
    ori_int = orientations_preprocess(moureze_orientations_heavy)
    opts = moureze_kriging
    opts.range = 10
    opts.c_o = 1
    opts.number_dimensions = 3
    return sp_int, ori_int, opts
