import dataclasses

import numpy as np


@dataclasses.dataclass
class FaultsData:
    fault_values_on_grid: np.ndarray
    fault_values_on_sp: np.ndarray
    
    fault_values_ref: np.ndarray = None
    fault_values_rest: np.ndarray = None
    
    @property
    def n_faults(self):
        return self.fault_values_on_sp.shape[0]
    