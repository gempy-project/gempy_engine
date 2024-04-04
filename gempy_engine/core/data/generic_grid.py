from dataclasses import dataclass, field

import numpy as np


@dataclass
class GenericGrid:
    values: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    name: str = "Generic Grid"
    
    def __len__(self):
        return self.values.shape[0]
