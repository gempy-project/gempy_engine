import enum

from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends

# ! Do not delete the fixtures imports
# Import fixtures
from tests.fixtures.simple_geometries import unconformity

from tests.fixtures.grids import \
    simple_grid_2d, \
    simple_grid_3d_more_points_grid, \
    simple_grid_3d_octree

from tests.fixtures.complex_geometries import *
from tests.fixtures.simple_models import *
from tests.fixtures.heavy_models import *

pykeops_enabled = False
backend = AvailableBackends.numpy
use_gpu = False
plot_pyvista = False  # ! Set here if you want to plot the results


BackendTensor._change_backend(
    engine_backend=backend,
    use_gpu=use_gpu,
    use_pykeops=pykeops_enabled
)

try:
    import pyvista as pv
except ImportError:
    plot_pyvista = False


class TestSpeed(enum.Enum):
    MILLISECONDS = 0
    SECONDS = 1
    MINUTES = 2
    HOURS = 3


class Requirements(enum.Enum):
    CORE = 0
    OPTIONAL = 1
    DEV = 2


TEST_SPEED = TestSpeed.MINUTES  # * Use seconds for compile errors, minutes before pushing and hours before release
REQUIREMENT_LEVEL = Requirements.CORE  # * Use CORE for mandatory tests, OPTIONAL for optional tests and DEV for development tests


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
    
    
@pytest.fixture(scope="session")
def tests_root():
    # Return the root 'tests/' directory
    return os.path.dirname(os.path.abspath(__file__))
