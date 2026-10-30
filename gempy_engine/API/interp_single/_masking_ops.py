from typing import List, Optional

import numpy as np
from numpy import ndarray

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.compute_scope import checking
from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.core.data.exported_structs import CombinedScalarFieldsOutput
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.scalar_field_output import ScalarFieldOutput
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure


def combine_scalar_fields(all_scalar_fields_outputs: List[ScalarFieldOutput],
                          data_descriptor: InputDataDescriptor,
                          compute_scalar_grad: bool = False) -> List[CombinedScalarFieldsOutput]:
    stack_structure = data_descriptor.stack_structure
    if _stackable(all_scalar_fields_outputs, stack_structure.masking_descriptor, data_descriptor.stack_relation,
                  compute_scalar_grad):
        combined = _combine_stacked(all_scalar_fields_outputs, stack_structure.masking_descriptor,
                                    data_descriptor.stack_relation, compute_scalar_grad)
        if checking():
            _assert_same_combination(combined, _combine_per_stack(all_scalar_fields_outputs, data_descriptor,
                                                                  compute_scalar_grad))
        return combined
    return _combine_per_stack(all_scalar_fields_outputs, data_descriptor, compute_scalar_grad)


def _combine_per_stack(all_scalar_fields_outputs: List[ScalarFieldOutput], data_descriptor: InputDataDescriptor,
                       compute_scalar_grad: bool) -> List[CombinedScalarFieldsOutput]:
    stack_structure = data_descriptor.stack_structure
    return _combine_scalar_fields(
        all_scalar_fields_outputs=all_scalar_fields_outputs,
        lithology_mask=_lithology_mask(all_scalar_fields_outputs, stack_structure),
        faults_mask=_faults_mask(all_scalar_fields_outputs, data_descriptor.stack_relation),
        stack_relation=data_descriptor.stack_relation,
        stack_structure=stack_structure,
        compute_scalar_grad=compute_scalar_grad
    )


def _lithology_mask(all_scalar_fields_outputs: List[ScalarFieldOutput], stack_structure: StacksStructure) -> np.ndarray:
    n_scalar_fields = len(all_scalar_fields_outputs)
    grid_size = all_scalar_fields_outputs[0].grid_size
    mask_matrix = BackendTensor.t.zeros((n_scalar_fields, grid_size), dtype=bool)
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
                mask_lith = all_scalar_fields_outputs[i].mask_components_erode
                mask_matrix[i, :] = mask_lith
            case StackRelationType.NULL_SPACE:
                mask_lith = all_scalar_fields_outputs[i].mask_components_erode
                # Zero out ignored grid slices for this null-space stack
                stack_structure.stack_number = i
                ignored_types = stack_structure.active_ignored_grid_types
                if ignored_types:
                    output_grid = all_scalar_fields_outputs[i].grid
                    for grid_name in ignored_types:
                        slc = _grid_slice_for_name(output_grid, grid_name)
                        mask_lith[slc] = False
                mask_matrix[i, :] = mask_lith
            case StackRelationType.FAULT:
                mask_matrix[i, :] = all_scalar_fields_outputs[i].mask_components_fault
            case False | StackRelationType.BASEMENT:
                mask_matrix[i, :] = all_scalar_fields_outputs[i].mask_components_basement
            case _:
                raise ValueError(f"Stack relation {stack_relation[i]} not recognized")

    return _final_lithology_mask(mask_matrix)


def _final_lithology_mask(mask_matrix):
    final_mask_array = BackendTensor.t.zeros(tuple(mask_matrix.shape), dtype=bool)
    final_mask_array[0] = mask_matrix[-1]
    final_mask_array[1:] = BackendTensor.t.cumprod(BackendTensor.t.invert(mask_matrix[:-1]), axis=0)
    final_mask_array *= mask_matrix

    return final_mask_array


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
                           compute_scalar_grad: bool = False) -> List[CombinedScalarFieldsOutput]:
    n_scalar_fields: int = len(all_scalar_fields_outputs)
    has_null_space = any(r == StackRelationType.NULL_SPACE for r in stack_relation)
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

        if stack_relation[i] is StackRelationType.NULL_SPACE:
            squeezed_value_block = squeezed_value_block + BackendTensor.t.zeros_like(squeezed_value_block)
        else:
            squeezed_value_block = _apply_mask(
                block_to_squeeze=interp_output.values_block,
                squeezed_mask_array=(lithology_mask[i]),
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


# region Stacked combination (PyTorch)
# The same masks and blocks as the per-stack loops above, with the stacks' rows in (n_stacks, grid) tensors:
# a few large ops per octree level instead of a handful per stack.

def _row_kind(relation) -> str | None:
    """Which row ``_lithology_mask`` gives a stack: its erosion mask, all False or all True; None for onlap and
    null-space stacks, which ``_combine_stacked`` leaves to the per-stack loops."""
    match relation:
        case StackRelationType.ONLAP | StackRelationType.NULL_SPACE:
            return None
        case StackRelationType.ERODE:
            return "erode"
        case StackRelationType.FAULT:
            return "false"  # mask_components_fault: zeros without a fault thickness
        case False | StackRelationType.BASEMENT:
            return "true"
        case _:
            return None


def _stackable(outputs: List[ScalarFieldOutput], masking_descriptor, stack_relation, compute_scalar_grad) -> bool:
    from gempy_engine.config import AvailableBackends
    if BackendTensor.engine_backend is not AvailableBackends.PYTORCH or len(outputs) < 2:
        return False
    if any(_row_kind(relation) is None for relation in list(masking_descriptor) + list(stack_relation)):
        return False
    grid_size = outputs[0].grid_size
    fields = [output.exported_fields for output in outputs]
    if any(output.grid_size != grid_size for output in outputs):
        return False
    if len({output.values_block.dtype for output in outputs}) != 1 or len({f.scalar_field.dtype for f in fields}) != 1:
        return False
    if compute_scalar_grad is True:
        with_gradients = [f.gx_field is not None for f in fields]
        if any(with_gradients) and not (all(with_gradients) and len({f.gx_field.dtype for f in fields}) == 1):
            return False
    return True


def _combine_stacked(outputs: List[ScalarFieldOutput], masking_descriptor, stack_relation,
                     compute_scalar_grad: bool) -> List[CombinedScalarFieldsOutput]:
    """``_combine_scalar_fields`` with ``_lithology_mask`` and ``_faults_mask``, for stacks that ``_stackable`` accepts.

    Bit-identical to the loops: the products are the same elementwise ops on stacked rows; the fault block is
    summed stack by stack as before; and the lithology masks are disjoint (each grid point is in at most one
    stack's mask), so each value and scalar-field block is the single unmasked term plus exact zeros, which a
    cumulative sum (sequential from +0, as the loop starts) gives exactly, signed zeros included.
    """
    import torch
    n_scalar_fields = len(outputs)
    grid_size = outputs[0].grid_size
    fields = [output.exported_fields for output in outputs]
    scalar_rows = [f.scalar_field for f in fields]
    lithology_kinds = [_row_kind(relation) for relation in masking_descriptor]
    fault_kinds = [_fault_row_kind(relation) for relation in stack_relation]

    rows = {}
    eroding = [i for i in range(n_scalar_fields) if "erode" in (lithology_kinds[i], fault_kinds[i])]
    if eroding:
        limits = torch.stack([fields[i].scalar_field_at_surface_points.min() for i in eroding])
        limits = limits.to(torch.result_type(scalar_rows[0], limits[0]))  # a 0-dim limit takes the field's dtype
        eroded = torch.stack([scalar_rows[i] for i in eroding]) > limits.reshape(-1, 1)
        rows.update(zip(eroding, eroded.unbind(0)))
    kinds = set(lithology_kinds + fault_kinds)
    if "false" in kinds:
        rows["false"] = BackendTensor.t.zeros(grid_size, dtype=bool)
    if "true" in kinds:
        rows["true"] = BackendTensor.t.ones(grid_size, dtype=bool)

    def mask(row_kinds):
        return torch.stack([rows[i if kind == "erode" else kind] for i, kind in enumerate(row_kinds)])

    lithology_mask = _final_lithology_mask(mask(lithology_kinds))
    faults_mask = mask(fault_kinds)

    values = torch.cat([output.values_block for output in outputs])
    lithology_values = values * lithology_mask
    lithology_scalars = torch.stack(scalar_rows) * lithology_mask
    # The loop's blocks start as full((1, N), 0), int64 so the values' dtype wins, and zeros((1, N)) of the
    # default dtype.
    final_blocks = torch.cumsum(lithology_values, dim=0).unbind(0)
    scalar_dtype = torch.promote_types(torch.get_default_dtype(), lithology_scalars.dtype)
    scalar_blocks = torch.cumsum(lithology_scalars.to(scalar_dtype), dim=0).unbind(0)

    squeezed_fault_block = BackendTensor.t.zeros((1, grid_size))
    fault_blocks = []
    for row in (values * faults_mask).unbind(0):
        squeezed_fault_block = (squeezed_fault_block + row).reshape(-1)
        fault_blocks.append(squeezed_fault_block)

    gradients = [[None] * n_scalar_fields] * 3
    if compute_scalar_grad is True and fields[0].gx_field is not None:
        zeros = BackendTensor.t.zeros((1, grid_size))
        gradients = [(zeros + torch.stack([getattr(f, name) for f in fields]) * lithology_mask).unbind(0)
                     for name in ("gx_field", "gy_field", "gz_field")]

    all_combined_scalar_fields = []
    for i, output in enumerate(outputs):
        final_exported_fields = ExportedFields(
            _scalar_field=scalar_blocks[i],
            _gx_field=gradients[0][i],
            _gy_field=gradients[1][i],
            _gz_field=gradients[2][i],
            _n_points_per_surface=output.exported_fields._n_points_per_surface,
            _slice_feature=slice(None)
        )
        all_combined_scalar_fields.append(CombinedScalarFieldsOutput(
            squeezed_mask_array=lithology_mask[i],
            final_block=final_blocks[i],
            faults_block=fault_blocks[i],
            final_exported_fields=final_exported_fields
        ))
    return all_combined_scalar_fields


def _fault_row_kind(relation) -> str:
    """Which row ``_faults_mask`` gives a stack: a fault's erosion mask, otherwise all False."""
    match relation:
        case StackRelationType.FAULT:
            return "erode"
        case _:
            return "false"


def bitwise_equal(a, b) -> bool:
    """Same dtype, shape and bits (signed zeros and NaN payloads included); None only equals None."""
    if a is None or b is None:
        return a is b
    import torch
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.is_floating_point:
        as_int = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.contiguous().view(as_int), b.contiguous().view(as_int))
    return torch.equal(a, b)


def _assert_same_combination(stacked: List[CombinedScalarFieldsOutput], per_stack: List[CombinedScalarFieldsOutput]):
    """Raise unless the stacked combination is bit for bit the per-stack one (GEMPY_CHECK_COMPUTE_CACHE)."""
    for s, p in zip(stacked, per_stack, strict=True):
        pairs = [(s.squeezed_mask_array, p.squeezed_mask_array), (s.final_block, p.final_block),
                 (s.faults_block, p.faults_block)]
        pairs += [(getattr(s.final_exported_fields, name), getattr(p.final_exported_fields, name))
                  for name in ("_scalar_field", "_gx_field", "_gy_field", "_gz_field")]
        if not all(bitwise_equal(a, b) for a, b in pairs):
            raise AssertionError("compute cache: stacked scalar-field combination differs from the per-stack one")

# endregion
