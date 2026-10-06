import enum
import os

# Tests are headless by default. Set GEMPY_TEST_PLOTS=True for interactive plotting.
if os.getenv("GEMPY_TEST_PLOTS", "False") != "True":
    os.environ.setdefault("MPLBACKEND", "Agg")

# Allow overriding backend via DEFAULT_BACKEND env var (for CI matrix builds)
_backend_name = os.getenv('DEFAULT_BACKEND', 'numpy')

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

backend = AvailableBackends[_backend_name]
use_gpu = os.getenv('USE_GPU', 'False') == 'True'
plot_pyvista = False  # ! Set here if you want to plot the results


BackendTensor._change_backend(
    engine_backend=backend,
    use_gpu=use_gpu
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


# * Allow CI to override via env vars; fall back to defaults if value is invalid
_test_speed_name = os.getenv('TEST_SPEED', 'MINUTES')
_requirement_name = os.getenv('REQUIREMENT_LEVEL', 'CORE')
try:
    TEST_SPEED = TestSpeed[_test_speed_name]
except KeyError:
    TEST_SPEED = TestSpeed.MINUTES
try:
    REQUIREMENT_LEVEL = Requirements[_requirement_name]
except KeyError:
    REQUIREMENT_LEVEL = Requirements.CORE


@pytest.fixture(scope='session', autouse=True)
def set_up_approval_tests():
    try:
        from approvaltests.reporters import set_default_reporter
        from tests.verify_helper import get_approval_reporter
    except ImportError:
        return

    set_default_reporter(get_approval_reporter())
    
    
@pytest.fixture(scope="session")
def tests_root():
    # Return the root 'tests/' directory
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True)
def restore_backend():
    from gempy_engine.config import is_pytorch_installed

    def settings():
        return dict(engine_backend=BackendTensor.engine_backend, use_gpu=BackendTensor.use_gpu,
                    use_pykeops=BackendTensor.use_pykeops, dtype=BackendTensor.dtype,
                    grads=BackendTensor.COMPUTE_GRADS)

    original = settings()
    pykeops_enabled = BackendTensor.pykeops_enabled
    if is_pytorch_installed:
        import torch
        grad_enabled = torch.is_grad_enabled()
    try:
        yield
    finally:
        if settings() != original:
            BackendTensor._change_backend(**original)
        BackendTensor.pykeops_enabled = pykeops_enabled
        if is_pytorch_installed:
            torch.set_grad_enabled(grad_enabled)
