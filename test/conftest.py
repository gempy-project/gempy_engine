import enum


from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends

# ! Do not delete the fixtures imports
# Import fixtures
from test.fixtures.simple_models import *

from test.fixtures.simple_geometries import\
    horizontal_stratigraphic,\
    horizontal_stratigraphic_scaled, \
    recumbent_fold_scaled, \
    unconformity

from test.fixtures.grids import \
    simple_grid_2d,\
    simple_grid_3d_more_points_grid, \
    simple_grid_3d_octree

from test.fixtures.complex_geometries import *

# * Eventually I want this to be random... Hopefully
# backend = np.random.choice([AvailableBackends.numpy, AvailableBackends.tensorflow])
# using_gpu = bool(np.random.choice([True, False]))
# using_pykeops = bool(np.random.choice([True, False]))


pykeops_enabled = True
backend = AvailableBackends.numpy
use_gpu = False

BackendTensor.change_backend(backend, use_gpu=use_gpu, pykeops_enabled=pykeops_enabled)

plot_pyvista = False


class TestSpeed(enum.Enum):
    MILLISECONDS = 0
    SECONDS = 1
    MINUTES = 2
    HOURS = 3


TEST_SPEED = TestSpeed.MINUTES  # * Use seconds for compile errors, minutes before pushing and hours before release


@pytest.fixture(scope='session', autouse=True)
def set_up_approval_tests():
    try:
        from approvaltests.reporters import GenericDiffReporter, DiffReporter, set_default_reporter
    except ImportError:
        return

    path_to_pycharm_executable = "/home/miguel/pycharm-2022.1.3/bin/pycharm.sh"
    if (path_to_pycharm_executable is not None) and os.path.exists(path_to_pycharm_executable):
        reporter = GenericDiffReporter.create(path_to_pycharm_executable)
        reporter.extra_args = ["diff"]
    else:
        reporter = DiffReporter()
    
    set_default_reporter(reporter) 


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
