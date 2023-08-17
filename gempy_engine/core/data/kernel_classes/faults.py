import dataclasses
from typing import Optional

import numpy as np

from gempy_engine.core.data.transforms import Transform


@dataclasses.dataclass
class FiniteFaultData:
    implicit_function: callable
    implicit_function_transform: Transform
    pivot: np.ndarray
    
    def apply(self, points: np.ndarray) -> np.ndarray:
        transformed_points = self.implicit_function_transform.apply_inverse_with_pivot(
            points=points,
            pivot=self.pivot
        )
        scalar_block = self.implicit_function(transformed_points)
        return scalar_block 
        


@dataclasses.dataclass
class FaultsData:
    fault_values_everywhere: np.ndarray = None
    fault_values_on_sp: np.ndarray = None
    
    fault_values_ref: np.ndarray = None
    fault_values_rest: np.ndarray = None
    
    # User given data:
    thickness: Optional[float] = None
    finite_fault_data: Optional[FiniteFaultData] = None  
    
    def __hash__(self):
        i = hash(self.__repr__())
        return i

    @classmethod
    def from_user_input(cls, thickness: Optional[float]) -> "FaultsData":
        return cls(
            fault_values_everywhere=np.zeros(0),
            fault_values_on_sp=np.zeros(0),
            thickness=thickness,
            fault_values_ref=np.zeros(0),
            fault_values_rest=np.zeros(0)
        )
    
    @property
    def finite_faults_defined(self) -> bool:
        return self.finite_fault_data is not None
    
    @property
    def n_faults(self):
        return self.fault_values_on_sp.shape[0]
    