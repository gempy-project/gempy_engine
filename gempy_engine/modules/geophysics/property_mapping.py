"""Map per-unit properties (density, susceptibility) to points without rounding unit ids.

The default mappers index properties with rounded unit ids (``properties[rint(ids) - 1]``), which has no gradient
with respect to the geometry. Here each stack's continuous unit field (its ``values_block``, a sigmoid blend of
consecutive unit ids) is mapped to the property by linear interpolation between those consecutive units, and the
stacks are then combined with their blend masks. Interpolating inside each stack, before the stacks are combined,
matters: ids of different stacks are not consecutive, so mapping an already combined block would pass through
units that are not at the contact.
"""
from typing import List

from ...core.backend_tensor import BackendTensor
from ...core.data.interp_output import InterpOutput
from ...core.data.stack_relation_type import StackRelationType


def blended_unit_property(stack_outputs: List[InterpOutput], unit_properties, grid_slice: slice):
    """Per-point property from every stack's continuous unit field, combined with the stacks' blend masks.

    Args:
        stack_outputs: One ``InterpOutput`` per stack, in stack order (e.g. ``OctreeLevel.outputs_centers``).
        unit_properties: Property per unit, indexed by 1-based unit id (``unit_properties[id - 1]``).
        grid_slice: The part of the grid to map, e.g. ``grid.geophysics_grid_slice``.

    Returns:
        Backend tensor with one property value per point of ``grid_slice``.
    """
    blended = None
    for output in stack_outputs:
        stack = output.scalar_fields
        if stack.is_null_space:
            raise NotImplementedError("Blended property mapping does not support null-space stacks yet.")
        if stack.stack_relation is StackRelationType.FAULT:
            continue  # Fault stacks own no lithology: their blend row is zero.
        weight = output.combined_scalar_field.blend_mask_array
        if weight is None:
            raise ValueError("Stack outputs carry no blend masks; they must come from combine_scalar_fields.")
        unit_field = stack.values_block.reshape(-1)[grid_slice] if stack.values_block.ndim == 1 else \
            stack.values_block[0, grid_slice]
        term = weight[grid_slice] * interpolate_unit_property(unit_field, unit_properties)
        blended = term if blended is None else blended + term
    return blended


def interpolate_unit_property(unit_field, unit_properties):
    """Linear interpolation of ``unit_properties`` (1-based) at a continuous unit id field.

    Exact for a sigmoid between two consecutive units, which is what one stack's segmentation produces at each of
    its contacts. Outside [1, n] the end values are kept.
    """
    n_units = unit_properties.shape[0]
    if n_units == 1:
        return unit_field * 0 + unit_properties[0]
    clipped = unit_field.clip(1, n_units)
    lower = BackendTensor.t.floor(clipped).clip(1, n_units - 1)
    fraction = clipped - lower
    lower_index = lower.long() if hasattr(lower, "long") else lower.astype(int)
    below = unit_properties[lower_index - 1]
    above = unit_properties[lower_index]
    return below + fraction * (above - below)
