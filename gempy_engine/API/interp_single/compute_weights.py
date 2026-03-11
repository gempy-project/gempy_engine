import numpy as np

from ._aux_faults_ops import _modify_faults_values_output, _grab_stack_fault_data
from ._interp_scalar_field import compute_weights, _evaluate_sys_eq
from ._interp_single_feature import input_preprocess_v2, scalar_field_segmentation_v2
from ._stack_ops import _construct_experted_fields
from ...core.backend_tensor import BackendTensor
from ...core.data import TensorsStructure
from ...core.data.exported_fields import ExportedFields
from ...core.data.internal_structs import SolverInput_v2, EvaluatorInput, SegmentationInput
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.kernel_classes.faults import FaultsData
from ...core.data.options import InterpolationOptions
from ...core.data.stack_relation_type import StackRelationType
from ...core.data.stacks_structure import StacksStructure


def compute_weights_for_stacks(interpolation_inputs: list[InterpolationInput], options: InterpolationOptions, stack_structure: StacksStructure,
                               tensor_structs: list[TensorsStructure], all_surface_points_size: int) -> list[SolverInput_v2]:
    solver_inputs: list[SolverInput_v2] = []

    all_stack_values_block: np.ndarray = BackendTensor.t.zeros(
        (stack_structure.n_stacks, all_surface_points_size),
        dtype=BackendTensor.dtype_obj
    )  # * Used for faults

    for i in range(stack_structure.n_stacks):
        stack_structure.stack_number = i
        # TODO: revive faults
        fv_on_all_sp = all_stack_values_block[stack_structure.active_faults_relations]
        fault_values_on_sp_on_stack = fv_on_all_sp[:, interpolation_inputs[i].slice_feature]

        # fault_input: FaultsData = _grab_stack_fault_data(  # * FAULTS
        #     _all_stack_values_block=all_stack_values_block,
        #     _interpolation_input_i=interpolation_inputs[i],
        #     _stack_structure=stack_structure
        # )

        solver_input: SolverInput_v2 = input_preprocess_v2(
            data_shape=tensor_structs[i],
            interpolation_input=interpolation_inputs[i],
            faults_on_sp=fault_values_on_sp_on_stack
        )
        solver_inputs.append(solver_input)

        # region compute weights
        # TODO: Adding external function
        weights = compute_weights(
            solver_input=solver_input,
            stack_number=i,
            options=options
        )
        solver_input.weights_x0 = weights

        eval_input: EvaluatorInput = EvaluatorInput(
            solver_input=solver_inputs[i],
            interpolation_input=interpolation_inputs[i],
            tensor_struct=tensor_structs[i],
            only_surface_points=True
        )

        # region evaluate
        values_block = scalar_field_segmentation_v2(
            exported_fields=(_construct_experted_fields(eval_input, options)),
            segmentation_input=SegmentationInput(
                unit_values=interpolation_inputs[i].unit_values,
                sigmoid_slope=(stack_structure.segmentation_function(eval_input.xyz_to_interpolate) if stack_structure.segmentation_function is not None
                               else options.sigmoid_slope
                               )
            )
        )

        # endregion

        # region Something with faults
        for i in range(stack_structure.n_stacks):  # TODO: This is the loop we need to split

            if interpolation_inputs[i].stack_relation is not StackRelationType.FAULT:  # * This is also for faults!
                continue
            # # * Modify the values for Fault stacks
            values_output = _modify_faults_values_output(  # ! This is all_STACK_values_block (not all_scalar_fields_outputs)
                fault_input=fault_input,
                values_on_all_xyz=values_block,
                xyz_to_interpolate=solver_input.xyz_to_interpolate
            )
            all_stack_values_block[i, :] = values_output
        # endregion

    return solver_inputs
