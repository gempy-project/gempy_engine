import enum
import warnings
from dataclasses import dataclass

from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions, KernelFunction


class DualContouringMaskingOptions(enum.Enum):
    NOTHING = enum.auto()  # * This is only for testing
    DISJOINT = enum.auto()
    INTERSECT = enum.auto()
    RAW = enum.auto()


@dataclass
class KernelOptions:
    range: int
    c_o: float
    uni_degree: int = 1
    i_res: float = 4.
    gi_res: float = 2.
    number_dimensions: int = 3

    kernel_function: AvailableKernelFunctions = AvailableKernelFunctions.exponential

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


class InterpolationOptions:
    kernel_options: KernelOptions = None  # * This is the compression of the fields above and the way to go in the future

    number_octree_levels: int = 1
    dual_contouring: bool = True
    dual_contouring_masking_options: DualContouringMaskingOptions = DualContouringMaskingOptions.DISJOINT
    
    debug: bool = False
    debug_water_tight: bool = False
    
    def __init__(self, range: int | float, c_o: float, uni_degree: int = 1, i_res: float = 4, gi_res: float = 2,
                 number_dimensions: int = 3, number_octree_levels: int = 1,
                 kernel_function: AvailableKernelFunctions = AvailableKernelFunctions.exponential, dual_contouring: bool = True):

        self.number_octree_levels = number_octree_levels
        self.kernel_options = KernelOptions(range, c_o, uni_degree, i_res, gi_res, number_dimensions, kernel_function)

    @property
    def range(self):
        return self.kernel_options.range

    @property
    def c_o(self):
        return self.kernel_options.c_o

    @property
    def uni_degree(self):
        return self.kernel_options.uni_degree
    
    @uni_degree.setter
    def uni_degree(self, value):
        warnings.warn("The uni_degree attribute is deprecated and will be removed in the future. ", DeprecationWarning)
        self.kernel_options.uni_degree = value

    @property
    def i_res(self):
        return self.kernel_options.i_res
    
    @i_res.setter
    def i_res(self, value):
        warnings.warn("The i_res attribute is deprecated and will be removed in the future. ", DeprecationWarning)
        self.kernel_options.i_res = value

    @property
    def gi_res(self):
        return self.kernel_options.gi_res
    
    @gi_res.setter
    def gi_res(self, value):
        warnings.warn("The gi_res attribute is deprecated and will be removed in the future. ", DeprecationWarning)
        self.kernel_options.gi_res = value

    @property
    def number_dimensions(self):
        return self.kernel_options.number_dimensions

    @property
    def kernel_function(self):
        return self.kernel_options.kernel_function
    
    @kernel_function.setter
    def kernel_function(self, value):
        warnings.warn("The kernel_function attribute is deprecated and will be removed in the future. ", DeprecationWarning)
        self.kernel_options.kernel_function = value
        
    @property
    def n_uni_eq(self):
        return self.kernel_options.n_uni_eq