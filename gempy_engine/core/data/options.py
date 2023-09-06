import enum
import pprint
import warnings
from dataclasses import dataclass, asdict

import gempy_engine.config
from gempy_engine.core.data.kernel_classes.solvers import Solvers
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions


class DualContouringMaskingOptions(enum.Enum):
    NOTHING = enum.auto()  # * This is only for testing
    DISJOINT = enum.auto()
    INTERSECT = enum.auto()
    RAW = enum.auto()


@dataclass(frozen=False)
class KernelOptions:
    range: int  # TODO: have constructor from RegularGrid
    c_o: float  # TODO: This should be a property
    uni_degree: int = 1
    i_res: float = 4.
    gi_res: float = 2.
    number_dimensions: int = 3

    kernel_function: AvailableKernelFunctions = AvailableKernelFunctions.exponential
    compute_condition_number: bool = False
    kernel_solver: Solvers = Solvers.DEFAULT

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

    def update_options(self, **kwargs):
        """
        Updates the options of the KernelOptions class based on the provided keyword arguments.

        Kwargs:
            range (int): Defines the range for the kernel. Must be provided. 
            c_o (float): A floating point value. Must be provided.
            uni_degree (int, optional): Degree for unification. Defaults to 1.
            i_res (float, optional): Resolution for `i`. Defaults to 4.0.
            gi_res (float, optional): Resolution for `gi`. Defaults to 2.0.
            number_dimensions (int, optional): Number of dimensions. Defaults to 3.
            kernel_function (AvailableKernelFunctions, optional): The function used for the kernel. Defaults to AvailableKernelFunctions.exponential.
            compute_condition_number (bool, optional): Whether to compute the condition number. Defaults to False.
            kernel_solver (Solvers, optional): Solver for the kernel. Defaults to Solvers.DEFAULT.

        Returns:
            None

        Raises:
            Warning: If a provided keyword is not a recognized attribute.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):  # checks if the attribute exists
                setattr(self, key, value)  # sets the attribute to the provided value
            else:
                warnings.warn(f"{key} is not a recognized attribute and will be ignored.")

    def __hash__(self):
        # Using a tuple to hash all the values together
        return hash((
            self.range,
            self.c_o,
            self.uni_degree,
            self.i_res,
            self.gi_res,
            self.number_dimensions,
            self.kernel_function,
            self.compute_condition_number,
        ))
    
    def __repr__(self):
        return f"KernelOptions({', '.join(f'{k}={v}' for k, v in asdict(self).items())})"

    def _repr_html_(self):
        html = f"""
            <table>
                <tr><td colspan='2' style='text-align:center'><b>KernelOptions</b></td></tr>
                {''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in asdict(self).items())}
            </table>
            """
        return html


@dataclass
class InterpolationOptions:
    # @off
    kernel_options                 : KernelOptions                = None  # * This is the compression of the fields above and the way to go in the future

    number_octree_levels           : int                          = 1
    current_octree_level           : int                          = 0  # * Make this a read only property 

    compute_scalar_gradient        : bool                         = False

    dual_contouring                : bool                         = True
    dual_contouring_masking_options: DualContouringMaskingOptions = DualContouringMaskingOptions.RAW
    dual_contouring_fancy          : bool                         = True

    debug                          : bool                         = gempy_engine.config.DEBUG_MODE
    debug_water_tight              : bool                         = False

    tensor_dtype                   : str                          = gempy_engine.config.TENSOR_DTYPE
    _number_octree_levels_surface  : int                          = 4
    
    def __init__(
            self,
            range                     : int | float,
            c_o                       : float,
            uni_degree                : int                              = 1,
            i_res                     : float                            = 4,
            gi_res                    : float                            = 2                                   , # ! This should be DEP
            number_dimensions         : int                              = 3                                   , # ? This probably too
            number_octree_levels      : int                              = 1,
            kernel_function           : AvailableKernelFunctions         = AvailableKernelFunctions.cubic,
            dual_contouring           : bool                             = True,
            compute_scalar_gradient   : bool                             = False,
            compute_condition_number  : bool                             = False,
            tensor_dtype              : gempy_engine.config.TENSOR_DTYPE = gempy_engine.config.TENSOR_DTYPE,  # TODO: This is unused
            
    ):
        self.number_octree_levels = number_octree_levels
        
        self.kernel_options = KernelOptions(
            range                      = range,
            c_o                        = c_o,
            uni_degree                 = uni_degree,
            i_res                      = i_res,
            gi_res                     = gi_res,
            number_dimensions          = number_dimensions,
            kernel_function            = kernel_function,
            compute_condition_number = compute_condition_number
        )

        self.dual_contouring         = dual_contouring
        self.compute_scalar_gradient = compute_scalar_gradient

        self.tensor_dtype = tensor_dtype
    # @on

    def __repr__(self):
        return f"InterpolationOptions({', '.join(f'{k}={v}' for k, v in asdict(self).items())})"

    def _repr_html_(self):
        html = f"""
                <table>
                    <tr><td colspan='2' style='text-align:center'><b>InterpolationOptions</b></td></tr>
                    {''.join(f'<tr><td>{k}</td><td>{v._repr_html_() if isinstance(v, KernelOptions) else v}</td></tr>' for k, v in asdict(self).items())}
                </table>
                """
        return html

    def update_options(self, **kwargs):
        """
        Updates the options of the class based on the provided keyword arguments.

        Kwargs:
            kernel_options (KernelOptions, optional): Options for the kernel. Default is None.
            number_octree_levels (int, optional): Number of octree levels. Default is 1.
            current_octree_level (int, optional): Current octree level. Default is 0.
            compute_scalar_gradient (bool, optional): Whether to compute the scalar gradient. Default is False.
            dual_contouring (bool, optional): Whether to use dual contouring. Default is True.
            dual_contouring_masking_options (DualContouringMaskingOptions, optional): Options for dual contouring masking.
            dual_contouring_fancy (bool, optional): Fancy version of dual contouring. Default is True.
            debug (bool, optional): Debug mode status. Default is derived from config.
            debug_water_tight (bool, optional): Debug mode for water-tight conditions. Default is False.
            tensor_dtype (str, optional): Data type for tensors. Default is derived from config.

        Returns:
            None

        Raises:
            Warning: If a provided keyword is not a recognized attribute.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):  # checks if the attribute exists
                setattr(self, key, value)  # sets the attribute to the provided value
            else:
                warnings.warn(f"{key} is not a recognized attribute and will be ignored.")

    @property
    def compute_corners(self):
        is_not_last_octree = (self.is_last_octree_level is False)
        is_dual_contouring = self.dual_contouring
        is_octree_for_surfaces = self.current_octree_level == self.number_octree_levels_surface - 1
        is_dual_contouring_and_octree_is_for_surfaces = is_dual_contouring and is_octree_for_surfaces
        
        corners_for_dual_cont = is_dual_contouring_and_octree_is_for_surfaces or is_not_last_octree
        return corners_for_dual_cont

    @property
    def is_last_octree_level(self) -> bool:
        return self.current_octree_level == self.number_octree_levels - 1

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
    
    @property
    def number_octree_levels_surface(self):
        if self._number_octree_levels_surface - 1 >= self.number_octree_levels:
            return self.number_octree_levels
        else:
            return self._number_octree_levels_surface
        
    @number_octree_levels_surface.setter
    def number_octree_levels_surface(self, value):
        # Check value is between 1 and number_octree_levels
        if not 1 <= value <= self.number_octree_levels:
            raise ValueError("number_octree_levels_surface must be between 1 and number_octree_levels")
        self._number_octree_levels_surface = value
    
