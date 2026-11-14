from typing import List, Optional

import numpy as np
from numpy import ndarray

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.core.data.exported_structs import CombinedScalarFieldsOutput
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.scalar_field_output import ScalarFieldOutput
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.modules.activator._soft_segment import _sigmoid


def blends_erosion_contact(options: InterpolationOptions, interpolation_input: InterpolationInput) -> bool:
    """Whether this stack's erosion contact is blended by the soft mask (see ``erosion_sigmoid_slope``).

    If so, the stack's own segmentation leaves out its lowest surface: the soft mask carries that contact.
    """
    return (options.erosion_sigmoid_slope is not None
            and interpolation_input.stack_relation is StackRelationType.ERODE
            and not interpolation_input.is_null_space)


def combine_scalar_fields(all_scalar_fields_outputs: List[ScalarFieldOutput],
                          data_descriptor: InputDataDescriptor,
                          compute_scalar_grad: bool = False,
                          erosion_sigmoid_slope: Optional[float] = None) -> List[CombinedScalarFieldsOutput]:
    stack_structure = data_descriptor.stack_structure
    lithology_mask = _lithology_mask(all_scalar_fields_outputs, stack_structure)
    blend_mask = (lithology_mask if erosion_sigmoid_slope is None else
                  _lithology_mask(all_scalar_fields_outputs, stack_structure, erosion_sigmoid_slope))
    return _combine_scalar_fields(
        all_scalar_fields_outputs=all_scalar_fields_outputs,
        lithology_mask=lithology_mask,
        faults_mask=_faults_mask(all_scalar_fields_outputs, data_descriptor.stack_relation),
        stack_relation=data_descriptor.stack_relation,
        stack_structure=stack_structure,
        compute_scalar_grad=compute_scalar_grad,
        blend_mask=blend_mask
    )


def _soft_mask_components_erode(output: ScalarFieldOutput, erosion_sigmoid_slope: float):
    """Sigmoid version of ``mask_components_erode``: ~1 above the lowest surface of the stack, ~0 below."""
    erode_limit_value = output.exported_fields.scalar_field_at_surface_points.min()
    return _sigmoid(output.exported_fields.scalar_field, erode_limit_value, 1 / erosion_sigmoid_slope)


def _lithology_mask(all_scalar_fields_outputs: List[ScalarFieldOutput], stack_structure: StacksStructure,
                    erosion_sigmoid_slope: Optional[float] = None) -> np.ndarray:
    """Which stack owns each point.

    With ``erosion_sigmoid_slope`` None (default) the mask is boolean. Otherwise ERODE rows are sigmoids, the mask
    is float, and the stacks are combined as ``m_i * prod_{j<i}(1 - m_j)`` so the result stays differentiable with
    respect to the position of erosion contacts. Onlap, fault, basement and null-space rows stay hard.
    """
    soft = erosion_sigmoid_slope is not None
    n_scalar_fields = len(all_scalar_fields_outputs)
    grid_size = all_scalar_fields_outputs[0].grid_size
    mask_dtype = BackendTensor.dtype_obj if soft else bool
    mask_matrix = BackendTensor.t.zeros((n_scalar_fields, grid_size), dtype=mask_dtype)
    stack_relation = stack_structure.masking_descriptor

    onlap_chain_counter = 0
    for i in range(n_scalar_fields):
        onlap_chain_cont: bool = stack_relation[i - 1] in [StackRelationType.ONLAP, StackRelationType.FAULT]
        onlap_chain_began: bool = stack_relation[i - 1 - onlap_chain_counter] is StackRelationType.ONLAP
        onlap_chain_counter: int = (onlap_chain_counter + 1) * onlap_chain_cont * onlap_chain_began

        if onlap_chain_counter:
            mask_matrix[i - 1] = all_scalar_fields_outputs[i].mask_components_erode_components_onlap

            x_mask = mask_matrix[(i - onlap_chain_counter):i, :]
            reversed_x_mask = BackendTensor.t.flip(x_mask, axis=0)
            cumprod_mask = BackendTensor.t.cumprod(reversed_x_mask, axis=0)
            reversed_cumprod_mask = BackendTensor.t.flip(cumprod_mask, axis=0)
            mask_matrix[i - onlap_chain_counter: i] = reversed_cumprod_mask

        match stack_relation[i]:
            case StackRelationType.ONLAP:
                pass
            case StackRelationType.ERODE:
                if soft and not stack_structure.is_null_space(i):
                    mask_lith = _soft_mask_components_erode(all_scalar_fields_outputs[i], erosion_sigmoid_slope)
                else:
                    mask_lith = all_scalar_fields_outputs[i].mask_components_erode
                mask_matrix[i, :] = mask_lith
            case StackRelationType.FAULT:
                mask_matrix[i, :] = all_scalar_fields_outputs[i].mask_components_fault
            case False | StackRelationType.BASEMENT:
                mask_matrix[i, :] = all_scalar_fields_outputs[i].mask_components_basement
            case _:
                raise ValueError(f"Stack relation {stack_relation[i]} not recognized")

    # After the loop: an onlap chain rewrites an ONLAP stack's row at the next stack.
    _exempt_ignored_grids_from_null_space(mask_matrix, all_scalar_fields_outputs, stack_structure)

    final_mask_array = BackendTensor.t.zeros((n_scalar_fields, grid_size), dtype=mask_dtype)
    final_mask_array[0] = mask_matrix[-1]
    not_above = (1 - mask_matrix[:-1]) if soft else BackendTensor.t.invert(mask_matrix[:-1])
    final_mask_array[1:] = BackendTensor.t.cumprod(not_above, axis=0)
    final_mask_array *= mask_matrix

    if stack_structure.is_bottom_null_space(n_scalar_fields - 1):
        # The bottom null stack keeps only its basement: what lies above its lowest surface is left void.
        not_void = BackendTensor.t.invert(_bottom_null_space_void(all_scalar_fields_outputs[-1], stack_structure))
        final_mask_array[-1] = final_mask_array[-1] * not_void if soft else final_mask_array[-1] & not_void

    return final_mask_array


def _bottom_null_space_void(output: ScalarFieldOutput, stack_structure: StacksStructure) -> np.ndarray:
    """Where the bottom null stack is void: above its lowest surface, except on the grids it ignores."""
    void = output.mask_components_erode  # A new array on every access
    ignored = stack_structure.ignored_grid_types_per_stack
    for entry in (ignored[-1] or ()) if ignored is not None else ():
        void[_grid_slice_for(output.grid, entry)] = False
    return void


def _exempt_ignored_grids_from_null_space(mask_matrix, all_scalar_fields_outputs: List[ScalarFieldOutput],
                                          stack_structure: StacksStructure) -> None:
    """A null-space stack does not claim the grids (or grid parts) it ignores: they keep the geology below."""
    if stack_structure.ignored_grid_types_per_stack is None:
        return
    for i, ignored in enumerate(stack_structure.ignored_grid_types_per_stack):
        if not ignored or not stack_structure.is_null_space(i) or stack_structure.is_bottom_null_space(i):
            continue  # The bottom null stack's ignores only shrink its void (``_bottom_null_space_void``)
        for entry in ignored:
            mask_matrix[i, _grid_slice_for(all_scalar_fields_outputs[i].grid, entry)] = False


def ignored_grid_name(entry) -> str:
    """The grid an ``ignored_grid_types`` entry refers to: a grid name, or ``(name, start, stop)``."""
    return entry if isinstance(entry, str) else entry[0]


def _grid_slice_for(grid: "EngineGrid", entry) -> slice:
    """A whole grid's slice, or for ``(name, start, stop)`` its points ``start`` to ``stop``.

    Temporary query grids (dual contouring edges, field samples) put other points in the custom grid,
    so a range past their end is clipped rather than rejected.
    """
    if isinstance(entry, str):
        return _grid_slice_for_name(grid, entry)
    name, start, stop = entry
    whole = _grid_slice_for_name(grid, name)
    length = whole.stop - whole.start
    return slice(whole.start + min(int(start), length), whole.start + min(int(stop), length))


def _grid_slice_for_name(grid: "EngineGrid", name: str) -> slice:
    match name:
        case "custom_grid":
            return grid.custom_grid_slice
        case "dense_grid":
            return grid.dense_grid_slice
        case "topography":
            return grid.topography_slice
        case "sections":
            return grid.sections_slice
        case "geophysics_grid":
            return grid.geophysics_grid_slice
        case "octree_grid":
            return grid.octree_grid_slice
        case "corners_grid":
            return grid.corners_grid_slice
        case _:
            raise ValueError(f"Unknown grid type: {name}")


def _faults_mask(all_scalar_fields_outputs: List[ScalarFieldOutput], stack_relation: List[StackRelationType]) -> np.ndarray:
    n_scalar_fields = len(all_scalar_fields_outputs)
    grid_size = all_scalar_fields_outputs[0].grid_size
    mask_matrix = BackendTensor.t.zeros((n_scalar_fields, grid_size), dtype=bool)

    for i in range(len(all_scalar_fields_outputs)):
        match stack_relation[i]:
            case StackRelationType.FAULT:
                mask_matrix[i, :] = all_scalar_fields_outputs[i].mask_components_erode  # * Faults behave as erosion contacts for the fault block
            case _:
                mask_matrix[i, :] = all_scalar_fields_outputs[i].mask_components_fault

    return mask_matrix


def _combine_scalar_fields(all_scalar_fields_outputs: List[ScalarFieldOutput],
                           lithology_mask: np.ndarray,
                           faults_mask: np.ndarray,
                           stack_relation: List[StackRelationType],
                           stack_structure: StacksStructure,
                           compute_scalar_grad: bool = False,
                           blend_mask: Optional[np.ndarray] = None) -> List[CombinedScalarFieldsOutput]:
    """``blend_mask`` (float, from ``erosion_sigmoid_slope``) replaces ``lithology_mask`` only when combining the
    values block. Scalar fields, their gradients and ``squeezed_mask_array`` (meshing, null space) stay hard."""
    if blend_mask is None:
        blend_mask = lithology_mask
    n_scalar_fields: int = len(all_scalar_fields_outputs)
    has_null_space = stack_structure.has_null_space
    init_value = 0
    squeezed_value_block: ndarray = BackendTensor.t.full((1, lithology_mask.shape[1]), init_value)
    squeezed_fault_block: ndarray = BackendTensor.t.zeros((1, lithology_mask.shape[1]))
    squeezed_scalar_field_block: ndarray = BackendTensor.t.zeros((1, lithology_mask.shape[1]))
    assigned_mask: ndarray = BackendTensor.t.zeros((1, lithology_mask.shape[1]), dtype=bool)

    def _apply_mask(block_to_squeeze: np.ndarray, squeezed_mask_array: np.ndarray, previous_block: np.ndarray) -> np.ndarray:
        return (previous_block + block_to_squeeze * squeezed_mask_array).reshape(-1)

    all_combined_scalar_fields = []
    for i in range(n_scalar_fields):
        interp_output: ScalarFieldOutput = all_scalar_fields_outputs[i]

        if stack_structure.is_null_space(i) and not stack_structure.is_bottom_null_space(i):
            squeezed_value_block = squeezed_value_block + BackendTensor.t.zeros_like(squeezed_value_block)
        else:
            squeezed_value_block = _apply_mask(
                block_to_squeeze=interp_output.values_block,
                squeezed_mask_array=(blend_mask[i]),
                previous_block=squeezed_value_block
            )
            assigned_mask = assigned_mask | lithology_mask[i]

        squeezed_scalar_field_block = _apply_mask(
            block_to_squeeze=interp_output.exported_fields.scalar_field,
            squeezed_mask_array=(lithology_mask[i]),
            previous_block=squeezed_scalar_field_block
        )

        squeezed_fault_block = _apply_mask(
            block_to_squeeze=interp_output.values_block,
            squeezed_mask_array=faults_mask[i],
            previous_block=squeezed_fault_block
        )

        if compute_scalar_grad is True and interp_output.exported_fields.gx_field is not None:
            squeezed_gx_block: Optional[ndarray] = BackendTensor.t.zeros((1, lithology_mask.shape[1]))
            squeezed_gy_block: Optional[ndarray] = BackendTensor.t.zeros((1, lithology_mask.shape[1]))
            squeezed_gz_block: Optional[ndarray] = BackendTensor.t.zeros((1, lithology_mask.shape[1]))

            squeezed_gx_block = _apply_mask(interp_output.exported_fields.gx_field, lithology_mask[i], squeezed_gx_block)
            squeezed_gy_block = _apply_mask(interp_output.exported_fields.gy_field, lithology_mask[i], squeezed_gy_block)
            squeezed_gz_block = _apply_mask(interp_output.exported_fields.gz_field, lithology_mask[i], squeezed_gz_block)
        else:
            squeezed_gx_block = None
            squeezed_gy_block = None
            squeezed_gz_block = None

        final_exported_fields = ExportedFields(
            _scalar_field=squeezed_scalar_field_block,
            _gx_field=squeezed_gx_block,
            _gy_field=squeezed_gy_block,
            _gz_field=squeezed_gz_block,
            _n_points_per_surface=interp_output.exported_fields._n_points_per_surface,
            _slice_feature=slice(None)
        )

        combined_scalar_fields = CombinedScalarFieldsOutput(
            squeezed_mask_array=(lithology_mask[i]),
            final_block=squeezed_value_block,
            faults_block=squeezed_fault_block,
            final_exported_fields=final_exported_fields
        )

        all_combined_scalar_fields.append(combined_scalar_fields)

    if has_null_space:
        null_mask = ~assigned_mask.reshape(-1)
        squeezed_value_block[null_mask] = stack_structure.null_space_id
        squeezed_fault_block[null_mask] = 0

    return all_combined_scalar_fields
