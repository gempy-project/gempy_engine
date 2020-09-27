import pandas as pd
import numpy as np
import pytest

from gempy_engine.data_structures.private_structures import SurfacePointsInternals, OrientationsInternals
from gempy_engine.data_structures.public_structures import SurfacePointsInput, InterpolationOptions, OrientationsInput
from gempy_engine.systems.generators import get_ref_rest, tile_dip_positions


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
    sp_t = SurfacePointsInput(sp[['X', 'Y', 'Z']].values,
                              sp['smooth'].values)
    return sp_t

@pytest.fixture(scope='session')
def moureze_kriging():
    return InterpolationOptions(10000, 1000)


@pytest.fixture(scope='session')
def moureze_orientations_heavy(moureze):
    _, ori = moureze
    n = 5
    ori_poss = ori[['X', 'Y', 'Z']].values,
    ori_pos = ori_poss[0]
    ori_grad = ori[['G_x', 'G_y', 'G_z']].values

    for i in range(n):
        ori_pos = np.vstack([ori_pos, ori_pos + np.array([i * 100, i * 100, i * 100])])
        ori_grad = np.vstack([ori_grad, ori_grad])

    ori_t = OrientationsInput(ori_pos,
                              dip_gradients=ori_grad)

    return ori_t


@pytest.fixture(scope='session')
def moureze_internals(moureze_sp, moureze_orientations_heavy, moureze_kriging):
    moureze_sp.sp_positions /= 1000

    args = get_ref_rest(
        moureze_sp,
        np.array([10, 50, moureze_sp.sp_positions.shape[0] - 60 - 3],
                 dtype='int32'))
    sp_int = SurfacePointsInternals(*args)

    args = tile_dip_positions(moureze_orientations_heavy.dip_positions/1000, 3)
    ori_int = OrientationsInternals(args)
    opts = moureze_kriging
    opts.range = 10
    opts.c_o = 1
    opts.number_dimensions = 3
    return sp_int, ori_int, opts