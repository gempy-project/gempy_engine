from dataclasses import dataclass, field

import numpy as np

from gempy_engine.core.utils import cast_type_inplace


@dataclass
class GenericGrid:
    values: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    name: str = "Generic Grid"

    def __post_init__(self):
        pass
        
    def __len__(self):
        return self.values.shape[0]
