from ._interp_scalar_field import compute_weights
from ._interp_single_feature import input_preprocess_v2
from ...core.data import TensorsStructure
from ...core.data.internal_structs import SolverInput_v2
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.options import InterpolationOptions
from ...core.data.stacks_structure import StacksStructure


def compute_weights_for_stacks(interpolation_inputs: list[InterpolationInput], options: InterpolationOptions, stack_structure: StacksStructure,
                               tensor_structs: list[TensorsStructure], stack_indices: list[int] | None = None) -> list[SolverInput_v2]:
    solver_inputs: list[SolverInput_v2] = []

    if stack_indices is None:
        stack_indices = list(range(stack_structure.n_stacks))

    for idx, global_i in enumerate(stack_indices):
        stack_structure.stack_number = global_i
        solver_input: SolverInput_v2 = input_preprocess_v2(
            data_shape=tensor_structs[idx],
            interpolation_input=interpolation_inputs[idx]
        )
        solver_inputs.append(solver_input)

        # region compute weights
        # TODO: Adding external function
        weights = compute_weights(
            solver_input=solver_input,
            stack_number=global_i,
            options=options
        )
        solver_input.weights_x0 = weights
    
        # endregion

    return solver_inputs
