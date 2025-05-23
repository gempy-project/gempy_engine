from dataclasses import dataclass
from typing import Annotated

import numpy as np

from .encoders.converters import numpy_array_short_validator


@dataclass
class GeophysicsInput:
    tz: Annotated[np.ndarray,  numpy_array_short_validator]
    densities: Annotated[np.ndarray,  numpy_array_short_validator]