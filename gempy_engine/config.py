from importlib.util import find_spec
from enum import Enum, auto, Flag
import os
from dotenv import load_dotenv


class AvailableBackends(Flag):
    numpy = auto()
    PYTORCH = auto()


# Define the paths for the .env files

script_dir = os.path.dirname(os.path.abspath(__file__))

dotenv_path = os.path.join(script_dir, '../.env')
dotenv_gempy_engine_path = os.path.expanduser('~/.env_gempy_engine')

# Check if the .env files exist and prioritize the local .env file
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
elif os.path.exists(dotenv_gempy_engine_path):
    load_dotenv(dotenv_gempy_engine_path)
else:
    load_dotenv()

DEBUG_MODE = os.getenv('DEBUG_MODE', 'True') == 'True'  # Note the handling of Boolean values
OPTIMIZE_MEMORY = os.getenv('OPTIMIZE_MEMORY', 'True') == 'True'
DEFAULT_BACKEND = AvailableBackends[os.getenv('DEFAULT_BACKEND', 'numpy')]
DEFAULT_PYKEOPS = os.getenv('DEFAULT_PYKEOPS', 'False') == 'True'
DEFAULT_TENSOR_DTYPE = os.getenv('DEFAULT_TENSOR_DTYPE', 'float64')
LINE_PROFILER_ENABLED = os.getenv('LINE_PROFILER_ENABLED', 'False') == 'True'
SET_RAW_ARRAYS_IN_SOLUTION = os.getenv('SET_RAW_ARRAYS_IN_SOLUTION', 'True') == 'True'
NOT_MAKE_INPUT_DEEP_COPY = os.getenv('NOT_MAKE_INPUT_DEEP_COPY', 'False') == 'True'
DUAL_CONTOURING_VERTEX_OVERLAP = os.getenv('NOT_MAKE_INPUT_DEEP_COPY', 'False') == 'True'

is_numpy_installed = find_spec("numpy") is not None
is_tensorflow_installed = find_spec("tensorflow") is not None
is_pytorch_installed = find_spec("torch")
is_pykeops_installed = find_spec("pykeops") is not None
