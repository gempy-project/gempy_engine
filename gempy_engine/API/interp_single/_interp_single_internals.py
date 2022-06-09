from typing import Tuple, List

import numpy as np
from numpy import ndarray

from ...core import data
from ...core.data import InterpolationOptions
from ...core.data.input_data_descriptor import StackRelationType, InputDataDescriptor, TensorsStructure
from ...core.data.exported_structs import InterpOutput, ExportedFields, MaskMatrices, Solutions
from ...core.data.internal_structs import SolverInput
from ...core.data.interpolation_input import InterpolationInput
from ...modules.activator import activator_interface
from ...modules.data_preprocess import data_preprocess_interface
from ...modules.kernel_constructor import kernel_constructor_interface as kernel_constructor
from ...modules.solver import solver_interface


class Buffer:
    weights = None

    @classmethod
    def clean(cls):
        cls.weights = None


def interpolate_all_fields(interpolation_input: InterpolationInput, options: InterpolationOptions,
                           data_descriptor: InputDataDescriptor) -> List[InterpOutput]:
    """Interpolate all scalar fields given a xyz array of points"""

    all_scalar_fields_outputs = _interpolate_stack(data_descriptor, interpolation_input, options)
    # TODO [x]: squeeze mask
    final_mask_matrix = _squeeze_mask(all_scalar_fields_outputs, data_descriptor.stack_relation)

    # TODO [x]: Now we need to multiply each row of the final_mask_matrix with val
    all_scalar_fields_outputs = _compute_final_block(all_scalar_fields_outputs, final_mask_matrix)

    return all_scalar_fields_outputs


def _compute_final_block(all_scalar_fields_outputs: List[InterpOutput], squeezed_mask_arrays: np.ndarray) -> List[InterpOutput]:
    n_scalar_fields = len(all_scalar_fields_outputs)
    squeezed_value_block: ndarray = np.zeros((1, squeezed_mask_arrays.shape[1]))
    squeezed_scalar_field_block: ndarray = np.zeros((1, squeezed_mask_arrays.shape[1]))
    squeezed_gx_block: ndarray = np.zeros((1, squeezed_mask_arrays.shape[1]))
    squeezed_gy_block: ndarray = np.zeros((1, squeezed_mask_arrays.shape[1]))
    squeezed_gz_block: ndarray = np.zeros((1, squeezed_mask_arrays.shape[1]))

    def _mask_and_squeeze(block_to_squeeze: np.ndarray, squeezed_mask_array: np.ndarray, previous_block: np.ndarray) -> np.ndarray:
        return (previous_block + block_to_squeeze * squeezed_mask_array).reshape(-1)

    # ? For the octrees I guess we need to apply the mask also to the ExportedFields
    for i in range(n_scalar_fields):
        interp_output: InterpOutput = all_scalar_fields_outputs[i]

        squeezed_value_block = _mask_and_squeeze(interp_output.values_block, squeezed_mask_arrays[i], squeezed_value_block)

        # shifted_scalar_field = interp_output.exported_fields.scalar_field - interp_output.exported_fields.scalar_field.min()
        # scalar_field_shift = (squeezed_scalar_field_block.max() - interp_output.exported_fields.scalar_field.min()) * 10 * i  # * Make sure each scalar field is in a different range
        # shifted_scalar_field = (interp_output.exported_fields.scalar_field + scalar_field_shift)

        shifted_scalar_field = interp_output.exported_fields.scalar_field  # ! This name here does no make any sense becose I am not shifting 

        squeezed_scalar_field_block = _mask_and_squeeze(shifted_scalar_field, squeezed_mask_arrays[i], squeezed_scalar_field_block)
        # interp_output.exported_fields.scalar_field_shift = scalar_field_shift

        squeezed_gx_block = _mask_and_squeeze(interp_output.exported_fields.gx_field, squeezed_mask_arrays[i], squeezed_gx_block)
        squeezed_gy_block = _mask_and_squeeze(interp_output.exported_fields.gy_field, squeezed_mask_arrays[i], squeezed_gy_block)
        squeezed_gz_block = _mask_and_squeeze(interp_output.exported_fields.gz_field, squeezed_mask_arrays[i], squeezed_gz_block)

        interp_output.final_block = squeezed_value_block
        interp_output.final_exported_fields = ExportedFields(
            _scalar_field=squeezed_scalar_field_block,
            _gx_field=squeezed_gx_block,
            _gy_field=squeezed_gy_block,
            _gz_field=squeezed_gz_block,
            n_points_per_surface=interp_output.exported_fields.n_points_per_surface,
            n_surface_points=None
        )

    return all_scalar_fields_outputs


def _interpolate_a_scalar_field(interpolation_input: InterpolationInput, options: InterpolationOptions,
                                data_shape: TensorsStructure, clean_buffer: bool = True) -> InterpOutput:
    output = InterpOutput()
    output.grid = interpolation_input.grid
    output.weights, output.exported_fields = interpolate_scalar_field(interpolation_input, options, data_shape)
    output.values_block = _segment_scalar_field(output, interpolation_input.unit_values)

    output.mask_components = _compute_mask_components(output.exported_fields, interpolation_input.stack_relation)
    if clean_buffer: Buffer.clean()
    return output


def _interpolate_stack(root_data_descriptor: InputDataDescriptor, interpolation_input: InterpolationInput,
                       options: InterpolationOptions) -> InterpOutput | list[InterpOutput]:
    all_scalar_fields_outputs: List[InterpOutput] = []

    stack_structure = root_data_descriptor.stack_structure

    if stack_structure is None:  # ! This branch is just for backward compatibility but we should try to get rid of it as soon as possible
        solutions = _interpolate_a_scalar_field(interpolation_input, options, root_data_descriptor.tensors_structure)
        all_scalar_fields_outputs.append(solutions)
        return all_scalar_fields_outputs
    else:
        for i in range(stack_structure.n_stacks):
            stack_structure.stack_number = i

            tensor_struct_i: TensorsStructure = TensorsStructure.from_tensor_structure_subset(root_data_descriptor, i)
            interpolation_input_i = InterpolationInput.from_interpolation_input_subset(interpolation_input, stack_structure)

            solutions = _interpolate_a_scalar_field(interpolation_input_i, options, tensor_struct_i)
            all_scalar_fields_outputs.append(solutions)

    return all_scalar_fields_outputs


def _squeeze_mask(all_scalar_fields_outputs: List[InterpOutput], stack_relation: List[StackRelationType]) -> np.ndarray:
    n_scalar_fields = len(all_scalar_fields_outputs)
    grid_size = all_scalar_fields_outputs[0].grid_size
    mask_matrix = np.zeros((n_scalar_fields, grid_size), dtype=np.bool)

    # Setting the mask matrix
    for i in range(n_scalar_fields):
        mask_lith = all_scalar_fields_outputs[i].mask_components.mask_lith
        match stack_relation[i]:
            case StackRelationType.ERODE:
                mask_matrix[i, :] = mask_lith
            case StackRelationType.ONLAP:
                pass
            case StackRelationType.FAULT:
                pass
            case False:
                mask_matrix[i, :] = mask_lith
            case _:
                raise ValueError(f"Unknown stack relation type: {stack_relation[i]}")

    # Doing the black magic
    final_mask_array = np.zeros((n_scalar_fields, grid_size), dtype=bool)
    final_mask_array[0] = mask_matrix[-1]
    final_mask_array[1:] = np.cumprod(np.invert(mask_matrix[:-1]), axis=0)
    final_mask_array *= mask_matrix

    for i in range(n_scalar_fields):
        all_scalar_fields_outputs[i].squeezed_mask_array = final_mask_array[i]

    return final_mask_array


def _compute_mask_components(exported_fields: ExportedFields, stack_relation: StackRelationType):
    # ! This is how I am setting the stackRelation in gempy
    # is_erosion = self.series.df['BottomRelation'].values[self.non_zero] == 'Erosion'
    # is_onlap = np.roll(self.series.df['BottomRelation'].values[self.non_zero] == 'Onlap', 1)
    # ! if len(is_erosion) != 0:
    # !     is_erosion[-1] = False

    # * This are the default values
    mask_erode = np.ones_like(exported_fields.scalar_field)
    mask_onlap = None  # ! it is the mask of the previous stack (from gempy: mask_matrix[n_series - 1, shift:x_to_interpolate_shape + shift])

    match stack_relation:
        case StackRelationType.ERODE:
            erode_limit_value = exported_fields.scalar_field_at_surface_points.min()
            mask_lith = exported_fields.scalar_field > erode_limit_value
        case StackRelationType.ONLAP:
            onlap_limit_value = exported_fields.scalar_field_at_surface_points.max()
            mask_lith = exported_fields.scalar_field > onlap_limit_value
        case StackRelationType.FAULT:
            mask_lith = np.zeros_like(exported_fields.scalar_field)
        case False:
            mask_lith = np.ones_like(exported_fields.scalar_field)
        case _:
            raise ValueError("Stack relation type is not supported")

    return MaskMatrices(mask_lith, None)


def interpolate_scalar_field(interpolation_input: InterpolationInput, options: data.InterpolationOptions,
                             data_shape: data.TensorsStructure) -> Tuple[np.ndarray, ExportedFields]:
    grid = interpolation_input.grid
    surface_points = interpolation_input.surface_points
    orientations = interpolation_input.orientations

    # Within series
    xyz_lvl0, ori_internal, sp_internal = _input_preprocess(data_shape, grid, orientations, surface_points)
    solver_input = SolverInput(sp_internal, ori_internal, options)

    if Buffer.weights is None:
        weights = _solve_interpolation(solver_input)
        Buffer.weights = weights
    else:
        weights = Buffer.weights

    # Within octree level
    # +++++++++++++++++++
    exported_fields = _evaluate_sys_eq(xyz_lvl0, solver_input, weights)
    exported_fields.n_points_per_surface = data_shape.reference_sp_position
    exported_fields.n_surface_points = surface_points.n_points

    Buffer.clean()
    return weights, exported_fields


def _segment_scalar_field(output: InterpOutput, unit_values: np.ndarray) -> np.ndarray:
    return activator_interface.activate_formation_block(output.exported_fields, unit_values, sigmoid_slope=50000)


def _solve_interpolation(interp_input: SolverInput):
    A_matrix = kernel_constructor.yield_covariance(interp_input)
    b_vector = kernel_constructor.yield_b_vector(interp_input.ori_internal, A_matrix.shape[0])
    # TODO: Smooth should be taken from options
    weights = solver_interface.kernel_reduction(A_matrix, b_vector, smooth=0.01)
    return weights


def _input_preprocess(data_shape: TensorsStructure, grid, orientations, surface_points) -> \
        Tuple[np.ndarray, data.OrientationsInternals, data.SurfacePointsInternals]:
    sp_internal = data_preprocess_interface.prepare_surface_points(surface_points, data_shape)
    ori_internal = data_preprocess_interface.prepare_orientations(orientations)
    grid_internal = data_preprocess_interface.prepare_grid(grid.values, surface_points)
    return grid_internal, ori_internal, sp_internal


def _evaluate_sys_eq(xyz: np.ndarray, interp_input: SolverInput, weights: np.ndarray) -> ExportedFields:
    options = interp_input.options

    eval_kernel = kernel_constructor.yield_evaluation_kernel(xyz, interp_input)
    eval_gx_kernel = kernel_constructor.yield_evaluation_grad_kernel(xyz, interp_input, axis=0)
    eval_gy_kernel = kernel_constructor.yield_evaluation_grad_kernel(xyz, interp_input, axis=1)

    scalar_field = weights @ eval_kernel
    gx_field = weights @ eval_gx_kernel
    gy_field = weights @ eval_gy_kernel

    if options.number_dimensions == 3:
        eval_gz_kernel = kernel_constructor.yield_evaluation_grad_kernel(xyz, interp_input, axis=2)
        gz_field = weights @ eval_gz_kernel
    elif options.number_dimensions == 2:
        gz_field = None
    else:
        raise ValueError("Number of dimensions have to be 2 or 3")

    return ExportedFields(scalar_field, gx_field, gy_field, gz_field)


def _get_scalar_field_at_surface_points(Z_x: np.ndarray, number_of_points_per_surface: np.ndarray,
                                        n_surface_points: int):
    npf = number_of_points_per_surface
    scalar_field_at_surface_points_values = Z_x[-n_surface_points:][npf]

    return scalar_field_at_surface_points_values


def _set_scalar_field_at_surface_points(exported_fields: ExportedFields, number_of_points_per_surface: np.ndarray,
                                        n_surface_points: int):
    exported_fields.n_points_per_surface = number_of_points_per_surface
    exported_fields.n_surface_points = n_surface_points
