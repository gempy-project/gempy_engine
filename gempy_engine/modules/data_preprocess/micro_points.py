import warnings

import numpy as np

from ...core.data.interpolation_input import InterpolationInput
from ...core.data.options import InterpolationOptions
from ...core.data.stack_relation_type import StackRelationType
from ...core.data.stacks_structure import StacksStructure


def prepare_micro_points(interpolation_input: InterpolationInput, options: InterpolationOptions,
                         stack_structure: StacksStructure) -> None:
    """Validate authored contacts and select the shared micro evaluation suffix."""
    micro = interpolation_input.micro_points
    interpolation_input._all_micro_points = None
    if micro is None or not len(micro.points):
        return

    surface_boundaries = np.cumsum(stack_structure.number_of_surfaces_per_stack)
    if (micro.surface_indices >= surface_boundaries[-1]).any():
        raise ValueError("micro_points.surface_indices contains an unknown global surface index")
    enabled = False
    overrides = stack_structure.interpolation_options_per_stack
    for stack_index in np.unique(np.searchsorted(surface_boundaries, micro.surface_indices, side="right")):
        stack_options = overrides[stack_index] if overrides is not None and overrides[stack_index] is not None else options
        if stack_options.micro_options.enabled:
            if stack_structure.masking_descriptor[stack_index] is StackRelationType.FAULT:
                raise ValueError(
                    f"Stack {stack_index} is a fault stack: enabled authored micro points on fault surfaces "
                    "are not supported. Disable micro_options for this stack or remove its micro points."
                )
            functions = stack_structure.interp_functions_per_stack
            if functions is not None and functions[stack_index] is not None:
                raise NotImplementedError("Authored micro points on external-function stacks are not supported")
            enabled = True
            continue
        warnings.warn(
            f"Stack {stack_index} contains micro points, but micro_options.enabled is False; "
            "these points will be ignored. Set it to True to apply the correction.",
            UserWarning,
            stacklevel=3,
        )
    if enabled:
        interpolation_input._all_micro_points = micro
