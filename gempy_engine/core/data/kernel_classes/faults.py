import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class FaultsData:
    fault_values_everywhere: np.ndarray
    fault_values_on_sp: np.ndarray
    
    fault_values_ref: np.ndarray = None
    fault_values_rest: np.ndarray = None
    
    # User given data:
    thickness: Optional[float] = None
    offset: Optional[float] = 1
    # TODO: Add finite fault scalar field

    def __hash__(self):
        i = hash(self.__repr__())
        return i
        "FaultsData(fault_values_everywhere=array([], shape=(0, 353), dtype=float64), fault_values_on_sp=array([], shape=(0, 345), dtype=float64), fault_values_ref=array([], shape=(0, 344), dtype=float64), fault_values_rest=array([], shape=(0, 344), dtype=float64), thickness=None, offset=1.0)"

    @classmethod
    def from_user_input(cls, thickness: Optional[float], offset: Optional[float]=1) -> "FaultsData":
        return cls(
            fault_values_everywhere=np.zeros(0),
            fault_values_on_sp=np.zeros(0),
            thickness=thickness,
            offset=offset,
            fault_values_ref=np.zeros(0),
            fault_values_rest=np.zeros(0)
        )
    
    @property
    def n_faults(self):
        return self.fault_values_on_sp.shape[0]
    