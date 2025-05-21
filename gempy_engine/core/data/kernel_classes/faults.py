import dataclasses
from typing import Optional, Callable

import numpy as np
from pydantic import Field

from ..encoders.converters import short_array_type
from ..transforms import Transform


@dataclasses.dataclass
class FiniteFaultData:
    implicit_function: Callable | None =  Field(exclude=True, default=None)#, default=None)
    implicit_function_transform: Transform = Field()
    pivot: short_array_type = Field()

    def apply(self, points: np.ndarray) -> np.ndarray:
        transformed_points = self.implicit_function_transform.apply_inverse_with_pivot(
            points=points,
            pivot=self.pivot
        )
        if self.implicit_function is None:
            raise ValueError("No implicit function defined. This can happen after deserializing (loading).")
        
        scalar_block = self.implicit_function(transformed_points)
        return scalar_block 
        


@dataclasses.dataclass
class FaultsData:
    fault_values_everywhere: short_array_type | None = None
    fault_values_on_sp: short_array_type | None = None
    
    fault_values_ref: short_array_type | None = None
    fault_values_rest: short_array_type | None = None
    
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
    