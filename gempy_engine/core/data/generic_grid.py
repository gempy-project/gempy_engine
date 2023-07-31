from dataclasses import dataclass

import numpy as np


@dataclass
class GenericGrid:
    values: np.ndarray = np.zeros((0, 3))
    name: str = "Generic Grid"
    
    def __len__(self):
        return self.values.shape[0]
