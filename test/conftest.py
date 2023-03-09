import enum

from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends

# ! Do not delete the fixtures imports
# Import fixtures
from test.fixtures.simple_geometries import \
    horizontal_stratigraphic, \
    horizontal_stratigraphic_scaled, \
    recumbent_fold_scaled, \
    unconformity

from test.fixtures.grids import \
    simple_grid_2d, \
    simple_grid_3d_more_points_grid, \
    simple_grid_3d_octree

from test.fixtures.complex_geometries import *
from test.fixtures.simple_models import *
from test.fixtures.heavy_models import *

pykeops_enabled = False
backend = AvailableBackends.numpy
use_gpu = False
plot_pyvista = True  # ! Set here if you want to plot the results

BackendTensor.change_backend(backend, use_gpu=use_gpu, pykeops_enabled=pykeops_enabled)

try:
    import pyvista as pv
except ImportError:
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
