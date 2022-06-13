import enum

import pandas as pd
import numpy as np
import pytest

from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.modules.data_preprocess._input_preparation import orientations_preprocess, surface_points_preprocess

# ! Do not delete the fixtures imports
# Import fixtures
from test.fixtures.simple_models import\
    simple_model_2,\
    simple_model_2_internals,\
    simple_model,\
    simple_model_3_layers,\
    simple_model_3_layers_high_res, \
    simple_model_values_block_output, \
    simple_model_interpolation_input, \
    unconformity_complex, \
    unconformity_complex_one_layer

from test.fixtures.simple_geometries import\
    horizontal_stratigraphic,\
    horizontal_stratigraphic_scaled, \
    recumbent_fold_scaled, \
    unconformity

from test.fixtures.grids import \
    simple_grid_2d,\
    simple_grid_3d_more_points_grid, \
    simple_grid_3d_octree

backend = np.random.choice([AvailableBackends.numpy, AvailableBackends.tensorflow])
using_gpu = bool(np.random.choice([True, False]))
using_pykeops = bool(np.random.choice([True, False]))

# TODO: For now pykeops is always disabled
BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=True,
                             pykeops_enabled=False)

plot_pyvista = False


class TestSpeed(enum.Enum):
    SECONDS = 0
    MINUTES = 1
    HOURS = 2


TEST_SPEED = TestSpeed.SECONDS  # * Use seconds for compile errors, minutes before pushing and hours before release


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
