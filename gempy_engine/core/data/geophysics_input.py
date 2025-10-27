from dataclasses import dataclass
from typing import Annotated, Optional

import numpy as np

from .encoders.converters import numpy_array_short_validator


@dataclass
class GeophysicsInput:
    # Gravity inputs (optional to allow magnetics-only workflows)
    tz: Optional[Annotated[np.ndarray, numpy_array_short_validator]] = None
    densities: Optional[Annotated[np.ndarray, numpy_array_short_validator]] = None

    # Magnetics inputs (optional for Phase 1)
    # Pre-projected TMI kernel per voxel (per device geometry), shape: (n_voxels_per_device,)
    mag_kernel: Optional[np.ndarray] = None
    # Susceptibilities per geologic unit (dimensionless SI), shape: (n_units,)
    susceptibilities: Optional[np.ndarray] = None
    # IGRF parameters metadata used to build kernel (inclination, declination, intensity (nT))
    igrf_params: Optional[dict] = None