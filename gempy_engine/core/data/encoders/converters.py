from typing import Annotated

import numpy as np
from pydantic import BeforeValidator

numpy_array_short_validator = BeforeValidator(lambda v: np.array(v) if v is not None else None)
short_array_type = Annotated[np.ndarray, numpy_array_short_validator]
