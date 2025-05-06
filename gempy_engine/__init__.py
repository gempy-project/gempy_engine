from gempy_engine.API.model.model_api import compute_model
from datetime import datetime

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # For Python <3.8, fallback
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("gempy_engine")  # Use package name
except ImportError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. subsurface should be
    # installed properly!
    __version__ = 'unknown-'+datetime.today().strftime('%Y%m%d')
