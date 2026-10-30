from typing import List, Optional

import numpy as np
from numpy import ndarray

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.compute_scope import assert_close, checking, scope_cache
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
    masking_descriptor, stack_relation = stack_structure.masking_descriptor, data_descriptor.stack_relation
    if _stackable(all_scalar_fields_outputs, masking_descriptor, stack_relation, compute_scalar_grad):
        limits = _compute_erosion_limits(all_scalar_fields_outputs, masking_descriptor, stack_relation, data_descriptor)
        combined = _combine_stacked(all_scalar_fields_outputs, masking_descriptor, stack_relation, compute_scalar_grad,
                                    limits)
        if checking():
            _assert_close_combination(combined, _combine_per_stack(all_scalar_fields_outputs, data_descriptor,
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
                     compute_scalar_grad: bool, limits=None) -> List[CombinedScalarFieldsOutput]:
    """``_combine_scalar_fields`` with ``_lithology_mask`` and ``_faults_mask``, for stacks that ``_stackable`` accepts.

    Masks and blocks come from one fused kernel on CUDA (``fused.combine``), otherwise from ``_stacked_blocks``.
    ``limits``: the stacks' erosion limits (``erosion_limits``), computed here when not given.
    """
    import torch
    from gempy_engine.modules import fused
    n_scalar_fields = len(outputs)
    grid_size = outputs[0].grid_size
    fields = [output.exported_fields for output in outputs]
    lithology_kinds = [_row_kind(relation) for relation in masking_descriptor]
    fault_kinds = [_fault_row_kind(relation) for relation in stack_relation]
    if limits is None:
        limits = erosion_limits(fields, lithology_kinds, fault_kinds)

    scalars = torch.stack([f.scalar_field for f in fields])
    values = torch.cat([output.values_block for output in outputs])
    # The loop's blocks start as full((1, N), 0), int64 so the values' dtype wins, and zeros((1, N)) of the
    # default dtype.
    scalar_dtype = torch.promote_types(torch.get_default_dtype(), scalars.dtype)
    fault_dtype = torch.promote_types(torch.get_default_dtype(), values.dtype)
    blocks = fused.combine(scalars, values, limits, lithology_kinds, fault_kinds, scalar_dtype, fault_dtype)
    if blocks is None:
        blocks = _stacked_blocks(scalars, values, limits, lithology_kinds, fault_kinds, scalar_dtype)
    lithology_mask, final_blocks, scalar_blocks, fault_blocks = blocks

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


def _stacked_blocks(scalars, values, limits, lithology_kinds, fault_kinds, scalar_dtype):
    """Lithology mask (n_stacks, N) and the value, scalar-field and fault blocks with torch ops.

    Bit-identical to the loops: the products are the same elementwise ops on stacked rows; the fault block is
    summed stack by stack as before; and the lithology masks are disjoint (each grid point is in at most one
    stack's mask), so each value and scalar-field block is the single unmasked term plus exact zeros, which a
    cumulative sum (sequential from +0, as the loop starts) gives exactly, signed zeros included.
    """
    import torch
    grid_size = scalars.shape[1]
    eroded = (scalars > limits.reshape(-1, 1)).unbind(0)
    kinds = set(lithology_kinds + fault_kinds)
    constant = {}
    if "false" in kinds:
        constant["false"] = BackendTensor.t.zeros(grid_size, dtype=bool)
    if "true" in kinds:
        constant["true"] = BackendTensor.t.ones(grid_size, dtype=bool)

    def mask(row_kinds):
        return torch.stack([eroded[i] if kind == "erode" else constant[kind] for i, kind in enumerate(row_kinds)])

    lithology_mask = _final_lithology_mask(mask(lithology_kinds))
    final_blocks = torch.cumsum(values * lithology_mask, dim=0)
    scalar_blocks = torch.cumsum((scalars * lithology_mask).to(scalar_dtype), dim=0)

    squeezed_fault_block = BackendTensor.t.zeros((1, grid_size))
    fault_blocks = []
    for row in (values * mask(fault_kinds)).unbind(0):
        squeezed_fault_block = (squeezed_fault_block + row).reshape(-1)
        fault_blocks.append(squeezed_fault_block)
    return lithology_mask, final_blocks, scalar_blocks, fault_blocks


def erosion_limits(fields: List[ExportedFields], lithology_kinds, fault_kinds):
    """The stacks' erosion limits, (n_stacks,) in the fields' dtype: the minimum scalar field at the surface points
    of a stack whose lithology or fault row is an erosion mask (as ``_compute_mask_components``), 0 elsewhere."""
    import torch
    limits = []
    for f, lithology_kind, fault_kind in zip(fields, lithology_kinds, fault_kinds):
        if "erode" in (lithology_kind, fault_kind):
            limit = f.scalar_field_at_surface_points.min()
            limits.append(limit.to(torch.result_type(f.scalar_field, limit)))  # a 0-dim limit takes the field's dtype
        else:
            limits.append(BackendTensor.t.zeros((), dtype=fields[0].scalar_field.dtype))
    return torch.stack(limits)


def _compute_erosion_limits(outputs: List[ScalarFieldOutput], masking_descriptor, stack_relation,
                            data_descriptor: InputDataDescriptor):
    """``erosion_limits``, within a compute taken at the first octree level: they are field values at the surface
    points, which do not depend on the grid."""
    fields = [output.exported_fields for output in outputs]
    kinds = ([_row_kind(relation) for relation in masking_descriptor],
             [_fault_row_kind(relation) for relation in stack_relation])
    shared = scope_cache()
    if shared is None:
        return erosion_limits(fields, *kinds)
    key = ("erosion_limits", id(data_descriptor))
    if key not in shared:  # holding the descriptor keeps its id from being reused during the scope
        shared[key] = (data_descriptor, erosion_limits(fields, *kinds))
    elif checking():
        assert_close(shared[key][1], erosion_limits(fields, *kinds), "erosion limits")
    return shared[key][1]


def _fault_row_kind(relation) -> str:
    """Which row ``_faults_mask`` gives a stack: a fault's erosion mask, otherwise all False."""
    match relation:
        case StackRelationType.FAULT:
            return "erode"
        case _:
            return "false"


def _assert_close_combination(stacked: List[CombinedScalarFieldsOutput], per_stack: List[CombinedScalarFieldsOutput]):
    """Raise unless the stacked combination matches the per-stack one (GEMPY_CHECK_COMPUTE_CACHE).

    A few points may differ: erosion limits taken at the first level can put a point that lies on a limit on
    the other side of it.
    """
    for s, p in zip(stacked, per_stack, strict=True):
        pairs = [(s.squeezed_mask_array, p.squeezed_mask_array, "lithology mask"),
                 (s.final_block, p.final_block, "block"), (s.faults_block, p.faults_block, "fault block")]
        pairs += [(getattr(s.final_exported_fields, name), getattr(p.final_exported_fields, name), name)
                  for name in ("_scalar_field", "_gx_field", "_gy_field", "_gz_field")]
        for a, b, what in pairs:
            assert_close(a, b, f"stacked combination, {what}", max_mismatch=1e-4)

# endregion
