import numpy as np
from pydantic import BeforeValidator

numpy_array_short_validator = BeforeValidator(lambda v: np.array(v) if v is not None else None)
