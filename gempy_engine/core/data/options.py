import enum
from dataclasses import dataclass

from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions, KernelFunction


class DualContouringMaskingOptions(enum.Enum):
    NOTHING = enum.auto()  # * This is only for testing
    DISJOINT = enum.auto()
    INTERSECT = enum.auto()
    RAW = enum.auto()


@dataclass
class InterpolationOptions:

    range: int
    c_o: float
    uni_degree: int = 1
    i_res: float = 4.
    gi_res: float = 2.
    number_dimensions: int = 3
    number_octree_levels:int = 1
    kernel_function: AvailableKernelFunctions = AvailableKernelFunctions.exponential
    
    dual_contouring_masking_options: DualContouringMaskingOptions = DualContouringMaskingOptions.DISJOINT
    
    debug: bool = False
    debug_water_tight: bool = False

    @property
    def n_uni_eq(self):
        if self.uni_degree == 1:
            n = self.number_dimensions
        elif self.uni_degree == 2:
            n = self.number_dimensions * 3
        elif self.uni_degree == 0:
            n = 0
        else:
            raise AttributeError('uni_degree must be 0,1 or 2')

        return n
