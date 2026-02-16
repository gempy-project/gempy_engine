import enum
import warnings

from pydantic import BaseModel, ConfigDict, Field, model_validator, PrivateAttr

import gempy_engine.config
from .evaluation_options import MeshExtractionMaskingOptions, EvaluationOptions
from .temp_interpolation_values import TempInterpolationValues
from ..kernel_classes.kernel_functions import AvailableKernelFunctions
from .kernel_options import KernelOptions
from ..raw_arrays_solution import RawArraysSolution


class InterpolationOptions(BaseModel):
    class CacheMode(enum.Enum):
        """ Cache mode for the interpolation"""
        NO_CACHE: int = enum.auto()  #: No cache at all even during the interpolation computation. This is quite expensive for no good reason.
        CACHE = enum.auto()
        IN_MEMORY_CACHE = enum.auto()
        CLEAR_CACHE = enum.auto()

    model_config = ConfigDict(
        arbitrary_types_allowed=False,
        use_enum_values=False,
        json_encoders={
                CacheMode: lambda e: e.value,
                AvailableKernelFunctions: lambda e: e.name
        }
    )

    # @off
    kernel_options: KernelOptions = Field(init=True, exclude=False)  # * This is the compression of the fields above and the way to go in the future
    evaluation_options: EvaluationOptions = Field(init=True, exclude= False)

    debug: bool
    cache_mode: CacheMode
    cache_model_name: str  # : Model name for the cache
    block_solutions_type: RawArraysSolution.BlockSolutionType
    sigmoid_slope: int
    debug_water_tight: bool = False

    # region Volatile
    temp_interpolation_values: TempInterpolationValues = Field(
        default_factory=TempInterpolationValues,
        exclude=True,
        repr=False
    )
   
    # endregion

    @classmethod
    def from_args(
            cls,
            range: int | float,
            c_o: float,
            uni_degree: int = 1,
            i_res: float = 4.,
            gi_res: float = 2.,  # ! This should be DEP
            number_dimensions: int = 3,  # ? This probably too
            number_octree_levels: int = 1,
            kernel_function: AvailableKernelFunctions = AvailableKernelFunctions.cubic,
            mesh_extraction: bool = True,
            compute_scalar_gradient: bool = False,
            compute_condition_number: bool = False,
    ):

        kernel_options = KernelOptions(
            range=range,
            c_o=c_o,
            uni_degree=uni_degree,
            i_res=i_res,
            gi_res=gi_res,
            number_dimensions=number_dimensions,
            kernel_function=kernel_function,
            compute_condition_number=compute_condition_number
        )

        evaluation_options = EvaluationOptions(
            _number_octree_levels=number_octree_levels,
            _number_octree_levels_surface=4,
            mesh_extraction=mesh_extraction,
            mesh_extraction_masking_options=MeshExtractionMaskingOptions.INTERSECT,
            mesh_extraction_fancy=True,
            compute_scalar_gradient=compute_scalar_gradient

        )

        temp_interpolation_values = TempInterpolationValues()
        debug = gempy_engine.config.DEBUG_MODE
        cache_mode = InterpolationOptions.CacheMode.IN_MEMORY_CACHE
        cache_model_name = ""
        block_solutions_type = RawArraysSolution.BlockSolutionType.OCTREE
        sigmoid_slope = 5_000_000

        return InterpolationOptions(
            kernel_options=kernel_options,
            evaluation_options=evaluation_options,
            # temp_interpolation_values=temp_interpolation_values,
            debug=debug,
            cache_mode=cache_mode,
            cache_model_name=cache_model_name,
            block_solutions_type=block_solutions_type,
            sigmoid_slope=sigmoid_slope,
            debug_water_tight=False,
        )

    # @on

    @classmethod
    def init_octree_options(cls, range=1.7, c_o=10., refinement: int = 1):
        return InterpolationOptions.from_args(
            range=range,
            c_o=c_o,
            mesh_extraction=True,
            number_octree_levels=refinement,
        )

    @classmethod
    def init_dense_grid_options(cls):
        options = InterpolationOptions.from_args(
            range=1.7,
            c_o=10.,
            mesh_extraction=False,
            number_octree_levels=1
        )
        options.block_solutions_type = RawArraysSolution.BlockSolutionType.DENSE_GRID
        return options

    @classmethod
    def probabilistic_options(cls):
        # TODO: This should have the sigmoid slope different
        raise NotImplementedError("Probabilistic interpolation is not yet implemented.")

    # def __repr__(self):
    #     return f"InterpolationOptions.from_args({', '.join(f'{k}={v}' for k, v in asdict(self).items())})"

    # def _repr_html_(self):
    #     html = f"""
    #             <table>
    #                 <tr><td colspan='2' style='text-align:center'><b>InterpolationOptions</b></td></tr>
    #                 {''.join(f'<tr><td>{k}</td><td>{v._repr_html_() if isinstance(v, KernelOptions) else v}</td></tr>' for k, v in asdict(self).items())}
    #             </table>
    #             """
    #     return html

    def update_options(self, **kwargs):
        """
        Updates the options of the class based on the provided keyword arguments.

        Kwargs:
            kernel_options (KernelOptions, optional): Options for the kernel. Default is None.
            number_octree_levels (int, optional): Number of octree levels. Default is 1.
            current_octree_level (int, optional): Current octree level. Default is 0.
            compute_scalar_gradient (bool, optional): Whether to compute the scalar gradient. Default is False.
            dual_contouring (bool, optional): Whether to use dual contouring. Default is True.
            mesh_extraction_masking_options (MeshExtractionMaskingOptions, optional): Options for dual contouring masking.
            evalution_options.mesh_extraction_fancy (bool, optional): Fancy version of dual contouring. Default is True.
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
    def number_octree_levels(self):
        return self.evaluation_options.number_octree_levels

    @number_octree_levels.setter
    def number_octree_levels(self, value):
        warnings.warn("The number_octree_levels attribute is deprecated and will be removed in the future. Use"
                      "evaluation_options.number_octree_levels instead.", DeprecationWarning)
        self.evaluation_options.number_octree_levels = value

    @property
    def mesh_extraction(self):
        return self.evaluation_options.mesh_extraction

    @mesh_extraction.setter
    def mesh_extraction(self, value):
        warnings.warn(
            "The mesh_extraction attribute is deprecated and will be removed in the future. Use"
            "evaluation_options.mesh_extraction instead.", DeprecationWarning)
        self.evaluation_options.mesh_extraction = value

    @property
    def compute_corners(self):
        is_not_last_octree = (self.is_last_octree_level is False)
        is_dual_contouring = self.mesh_extraction
        is_octree_for_surfaces = self.temp_interpolation_values.current_octree_level + 1 == self.number_octree_levels_surface
        is_dual_contouring_and_octree_is_for_surfaces = is_dual_contouring and is_octree_for_surfaces

        compute_corners = is_dual_contouring_and_octree_is_for_surfaces or is_not_last_octree
        return compute_corners

    @property
    def compute_scalar_gradient(self):
        return self.evaluation_options.compute_scalar_gradient

    @compute_scalar_gradient.setter
    def compute_scalar_gradient(self, value):
        warnings.warn("The compute_scalar_gradient attribute is deprecated and will be removed in the future. Use"
                      "evaluation_options.compute_scalar_gradient instead.", DeprecationWarning)
        self.evaluation_options.compute_scalar_gradient = value

    @property
    def is_last_octree_level(self) -> bool:
        return self.temp_interpolation_values.current_octree_level == self.number_octree_levels - 1

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
        return self.evaluation_options.number_octree_levels_surface

    @number_octree_levels_surface.setter
    def number_octree_levels_surface(self, value):
        warnings.warn("The number_octree_levels_surface attribute is deprecated and will be removed in the future. Use"
                      "evaluation_options.number_octree_levels_surface instead."
                      , DeprecationWarning)
        self.evaluation_options.number_octree_levels_surface = value

    @property
    def evaluation_chunk_size(self):
        return self.evaluation_options.evaluation_chunk_size
