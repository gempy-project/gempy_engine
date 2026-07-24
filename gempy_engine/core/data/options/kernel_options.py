import warnings
import math

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional

from pydantic import field_validator

from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.kernel_classes.solvers import Solvers


class NuggetImplementation(str, Enum):
    LEGACY = "legacy"
    DIAGONAL_REF_REST = "diagonal_ref_rest"
    FULL_POINT_COVARIANCE = "full_point_covariance"


@dataclass(frozen=False)
class KernelOptions:
    range: int | float  # TODO: have constructor from RegularGrid
    c_o: float  # TODO: This should be a property
    uni_degree: int = 1
    i_res: float = 4.
    gi_res: float = 2.
    number_dimensions: int = 3

    kernel_function: AvailableKernelFunctions = AvailableKernelFunctions.exponential
    kernel_solver: Solvers = Solvers.DEFAULT

    compute_condition_number: bool = False
    optimizing_condition_number: bool = False
    condition_number: Optional[float] = None
    condition_number_before: Optional[float] = None
    condition_number_after: Optional[float] = None

    fault_drift_equilibration: bool = True
    fault_drift_regularization: float = 1e-3
    symmetric_equilibration_method: str = "none"
    symmetric_equilibration_max_iterations: int = 10
    symmetric_equilibration_tolerance: float = 1e-2
    nugget_implementation: NuggetImplementation = NuggetImplementation.LEGACY

    def __post_init__(self):
        self.range = float(self.range)
        self.c_o = float(self.c_o)
        if isinstance(self.nugget_implementation, str):
            self.nugget_implementation = NuggetImplementation(self.nugget_implementation)
        if not math.isfinite(self.fault_drift_regularization) or self.fault_drift_regularization < 0:
            raise ValueError("fault_drift_regularization must be finite and non-negative")
        if self.symmetric_equilibration_method not in ("none", "ruiz"):
            raise ValueError("symmetric_equilibration_method must be 'none' or 'ruiz'")
        if self.symmetric_equilibration_max_iterations < 1:
            raise ValueError("symmetric_equilibration_max_iterations must be at least 1")
        if not math.isfinite(self.symmetric_equilibration_tolerance) or self.symmetric_equilibration_tolerance < 0:
            raise ValueError("symmetric_equilibration_tolerance must be finite and non-negative")

    @field_validator('kernel_function', mode='before', json_schema_input_type=str)
    @classmethod
    def _deserialize_kernel_function_from_name(cls, value):
        """
        Ensures that a string input (e.g., "cubic" from JSON)
        is correctly converted to an AvailableKernelFunctions enum member.
        """
        if isinstance(value, str):
            try:
                return AvailableKernelFunctions[value]  # Lookup enum member by name
            except KeyError:
                # This provides a more specific error if the name doesn't exist
                valid_names = [member.name for member in AvailableKernelFunctions]
                raise ValueError(f"Invalid kernel function name '{value}'. Must be one of: {valid_names}")
        # If it's already an AvailableKernelFunctions member (e.g., during direct model instantiation),
        # or if it's another type that Pydantic's later validation will catch as an error.
        return value

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
            fault_drift_equilibration (bool, optional): Scale fault rows and columns before solving. Defaults to True.
            fault_drift_regularization (float, optional): Relative diagonal loading for fault coefficients. Defaults to 1e-3.
            nugget_implementation (NuggetImplementation, optional): Nugget covariance implementation.

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
                self.fault_drift_equilibration,
                self.fault_drift_regularization,
                self.symmetric_equilibration_method,
                self.symmetric_equilibration_max_iterations,
                self.symmetric_equilibration_tolerance,
                self.nugget_implementation,
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
