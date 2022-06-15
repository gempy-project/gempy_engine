import dataclasses

import numpy as np


@dataclasses.dataclass
class FaultsInternals:
    fault_values_ref: np.ndarray
    fault_values_rest: np.ndarray
    
    @property
    def n_faults(self):
        return self.fault_values_ref.shape[1]
    