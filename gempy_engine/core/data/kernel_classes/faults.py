import dataclasses
from typing import Optional

import numpy as np

from ..encoders.converters import short_array_type
from ..finite_fault import FiniteFault


@dataclasses.dataclass
class FaultsData:
    fault_values_everywhere: short_array_type | None = None
    fault_values_on_sp: short_array_type | None = None
    
    fault_values_ref: short_array_type | None = None
    fault_values_rest: short_array_type | None = None
    
    # User given data:
    thickness: Optional[float] = None
    finite_fault: Optional[FiniteFault] = None
    
    def __hash__(self):
        i = hash(self.__repr__())
        return i

    @classmethod
    def from_user_input(cls, thickness: Optional[float], finite_fault: Optional[FiniteFault] = None) -> "FaultsData":
        return cls(
            fault_values_everywhere=np.zeros(0),
            fault_values_on_sp=np.zeros(0),
            thickness=thickness,
            finite_fault=finite_fault,
            fault_values_ref=np.zeros(0),
            fault_values_rest=np.zeros(0)
        )

    @property
    def finite_fault_defined(self) -> bool:
        return self.finite_fault is not None

    @property
    def n_faults(self):
        return self.fault_values_on_sp.shape[0]
