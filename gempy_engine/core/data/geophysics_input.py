import warnings
from dataclasses import dataclass
from typing import Annotated, Optional

import numpy as np

from .encoders.converters import numpy_array_short_validator


@dataclass
class GravityInput:
    tz: Annotated[np.ndarray, numpy_array_short_validator]
    densities: Annotated[np.ndarray, numpy_array_short_validator]


@dataclass
class MagneticsInput:
    mag_kernel: np.ndarray
    susceptibilities: np.ndarray
    igrf_params: dict


@dataclass
class GeophysicsInput:
    gravity_input: Optional[GravityInput] = None
    magnetics_input: Optional[MagneticsInput] = None
    
    def __init__(self, gravity_input: Optional[GravityInput] = None,
                 magnetics_input: Optional[MagneticsInput] = None,
                 tz: Optional[Annotated[np.ndarray, numpy_array_short_validator]] = None,
                 densities: Optional[Annotated[np.ndarray, numpy_array_short_validator]] = None):
        if gravity_input is not None:
            self.gravity_input = gravity_input
        else:
            warnings.warn("Using deprecated GeophysicsInput constructor. Use GravityInput instead.", DeprecationWarning)
            self.gravity_input = GravityInput(tz=tz, densities=densities)
        if magnetics_input is not None:
            self.magnetics_input = magnetics_input

    @property
    def tz(self) -> Optional[Annotated[np.ndarray, numpy_array_short_validator]]:
        if self.gravity_input is not None:
            return self.gravity_input.tz
        return None

    @tz.setter
    def tz(self, value: Optional[Annotated[np.ndarray, numpy_array_short_validator]]):
        if value is not None:
            if self.gravity_input is None:
                self.gravity_input = GravityInput(tz=value, densities=None)
            else:
                self.gravity_input.tz = value

    @property
    def densities(self) -> Optional[Annotated[np.ndarray, numpy_array_short_validator]]:
        if self.gravity_input is not None:
            return self.gravity_input.densities
        return None

    @densities.setter
    def densities(self, value: Optional[Annotated[np.ndarray, numpy_array_short_validator]]):
        if value is not None:
            if self.gravity_input is None:
                self.gravity_input = GravityInput(tz=None, densities=value)
            else:
                self.gravity_input.densities = value
