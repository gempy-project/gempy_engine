from typing import List

from ._interp_scalar_field import _evaluate_sys_eq
from ._interp_single_feature import scalar_field_segmentation_v2
from ...core.data import TensorsStructure, InterpolationOptions
from ...core.data.exported_fields import ExportedFields
from ...core.data.internal_structs import EvaluatorInput, SegmentationInput
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.options import InterpolationOptions
from ...core.data.scalar_field_output import ScalarFieldOutput
from ...core.data.stacks_structure import StacksStructure


def segment(eval_inputs: list[EvaluatorInput], exported_fields_per_stack: list[ExportedFields], interpolation_inputs: list[InterpolationInput], options: InterpolationOptions, solver_inputs, stack_structure: StacksStructure) -> list[ScalarFieldOutput | None]:
    all_scalar_fields_outputs: List[ScalarFieldOutput | None] = [None] * stack_structure.n_stacks
    for i in range(stack_structure.n_stacks):  # TODO: This is the loop we need to split
        # region segmentation
        values_block = scalar_field_segmentation_v2(
            exported_fields=exported_fields_per_stack[i],
            segmentation_input=SegmentationInput(
                unit_values=interpolation_inputs[i].unit_values,
                sigmoid_slope=(stack_structure.segmentation_function(eval_inputs[i].xyz_to_interpolate) if stack_structure.segmentation_function is not None
                               else options.sigmoid_slope
                               )
            )
        )
        # endregion
        output = ScalarFieldOutput(
            weights=solver_inputs[i].weights_x0,
            grid=interpolation_inputs[i].grid,
            exported_fields=exported_fields_per_stack[i],
            values_block=values_block,
            stack_relation=interpolation_inputs[i].stack_relation
        )

        # @on
        all_scalar_fields_outputs[i] = output
    return all_scalar_fields_outputs


def evaluate(interpolation_inputs: list[InterpolationInput], options: InterpolationOptions, solver_inputs, stack_structure: StacksStructure, tensor_structs: list[TensorsStructure]) -> tuple[list[EvaluatorInput], list[ExportedFields]]:
    eval_inputs: list[EvaluatorInput] = []
    exported_fields_per_stack: list[ExportedFields] = []
    for i in range(stack_structure.n_stacks):  # TODO: This is the loop we need to split
        eval_input: EvaluatorInput = EvaluatorInput(
            solver_input=solver_inputs[i],
            interpolation_input=interpolation_inputs[i],
            tensor_struct=tensor_structs[i],
        )

        eval_inputs.append(eval_input)

        # region evaluate
        exported_fields = _construct_experted_fields(eval_input, options)

        exported_fields_per_stack.append(exported_fields)
        # endregion
    return eval_inputs, exported_fields_per_stack


def _construct_experted_fields(eval_input: EvaluatorInput, options: InterpolationOptions) -> ExportedFields:
    exported_fields: ExportedFields = _evaluate_sys_eq(
        eval_input=eval_input,
        weights=eval_input.solver_input.weights_x0,
        options=options
    )

    exported_fields.set_structure_values_from_eval_input(eval_input)
    exported_fields.debug = eval_input.solver_input.debug
    return exported_fields
