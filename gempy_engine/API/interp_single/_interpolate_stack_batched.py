# ... existing code ...

from typing import List, Tuple

import numpy as np
import torch

from ._interp_single_feature import input_preprocess, _interpolate_external_function
from ...core.backend_tensor import BackendTensor
from ...core.data import TensorsStructure
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.options import InterpolationOptions
from ...core.data.scalar_field_output import ScalarFieldOutput
from ...core.data.stack_relation_type import StackRelationType
from ...modules.evaluator.generic_evaluator import generic_evaluator
from ...modules.evaluator.symbolic_evaluator import symbolic_evaluator
from ...modules.kernel_constructor import kernel_constructor_interface as kernel_constructor


# TODO: [ ] Batch only pykeops evaluations.
# TODO: [ ] To speed up the interpolation, we should try pykeops solver with fall back

def _interpolate_stack_batched(root_data_descriptor: InputDataDescriptor, root_interpolation_input: InterpolationInput,
                               options: InterpolationOptions) -> List[ScalarFieldOutput]:
    """
    Optimized batched interpolation using Split-Loop Pipelining and CUDA Streams.

    Strategy:
    1. CPU Phase: Pre-process all stacks (Python overhead, CPU prep, Data transfer initiation).
    2. GPU Phase: Launch Kernel Assembly, Solve, and Evaluation into parallel streams.

    This avoids the O(N^3) cost of padding matrices and prevents CPU prep from stalling the GPU.
    """
    BackendTensor.pykeops_enabled = False
    stack_structure = root_data_descriptor.stack_structure
    n_stacks = stack_structure.n_stacks

    # Result container
    all_scalar_fields_outputs: List[ScalarFieldOutput | None] = [None] * n_stacks

    # Shared memory for results (Fault interactions need this)
    xyz_to_interpolate_size: int = root_interpolation_input.grid.len_all_grids + root_interpolation_input.surface_points.n_points

    # Allocate on GPU once
    all_stack_values_block: torch.Tensor = BackendTensor.t.zeros(
        (n_stacks, xyz_to_interpolate_size),
        dtype=BackendTensor.dtype_obj,
        device=BackendTensor.device
    )

    # === Phase 1: CPU Preparation Loop ===
    # We prepare all python objects and initiate data transfers here.
    # This ensures that when we start launching GPU kernels, we don't stop for CPU work.
    prepared_stacks: List[Tuple[int, InterpolationInput, object]] = []
    for i in range(n_stacks):
        stack_structure.stack_number = i
        tensor_struct_i = TensorsStructure.from_tensor_structure_subset(root_data_descriptor, i)
        interpolation_input_i = InterpolationInput.from_interpolation_input_subset(
            all_interpolation_input=root_interpolation_input,
            stack_structure=stack_structure
        )

        # Handle Faults Dependencies
        # Note: In a fully pipelined approach, we can't read the *results* of previous stacks yet.
        # However, the 'fault_values_everywhere' usually comes from the Shared Tensor 'all_stack_values_block'.
        # As long as we enforce stream dependencies later, we can set up the views/pointers here.

        if interpolation_input_i.fault_values:
            fault_data = interpolation_input_i.fault_values
            # Create views into the shared block
            fault_data.fault_values_everywhere = all_stack_values_block[stack_structure.active_faults_relations]

            # We need to be careful with slicing here. 
            # The slicing operation itself is fast/metadata only on Tensors.
            fv_on_all_sp = fault_data.fault_values_everywhere[:, interpolation_input_i.grid.len_all_grids:]
            fault_data.fault_values_on_sp = fv_on_all_sp[:, interpolation_input_i.slice_feature]
            interpolation_input_i.fault_values = fault_data

        # Heavy CPU work: Prepare SolverInput (converts numpy to tensors, etc)
        solver_input = input_preprocess(tensor_struct_i, interpolation_input_i)

        prepared_stacks.append((i, interpolation_input_i, solver_input))

    # === Phase 2: GPU Execution Loop ===
    # Create streams and events
    streams = [torch.cuda.Stream() for _ in range(n_stacks)]
    events = [torch.cuda.Event() for _ in range(n_stacks)]

    for i, interpolation_input_i, solver_input in prepared_stacks:
        stream = streams[i]

        with torch.cuda.stream(stream):
            # 1. Synchronization: Wait for dependencies (Faults)
            active_faults = root_data_descriptor.stack_structure.active_faults_relations
            if active_faults is not None:
                # Find which stacks we depend on
                # active_faults is likely a boolean mask for previous stacks
                if hasattr(active_faults, 'dtype') and active_faults.dtype == bool:
                    dep_indices = np.where(active_faults)[0]
                elif isinstance(active_faults, (list, tuple, np.ndarray)):
                    dep_indices = active_faults
                else:
                    dep_indices = []

                for dep_idx in dep_indices:
                    if dep_idx < i:  # Can only wait on previous stacks
                        stream.wait_event(events[dep_idx])

            # 2. Compute or Evaluate
            if root_data_descriptor.stack_structure.interp_function is None:
                # --- A. Kriging Solve (GPU) ---

                # Construct Covariance Matrix (O(N^2))
                # This is now inside the stream, so it runs in parallel with other stacks
                A_mat = kernel_constructor.yield_covariance(solver_input, options.kernel_options)
                b_vec = kernel_constructor.yield_b_vector(solver_input.ori_internal, A_mat.shape[0])

                # Solve System (O(N^3))
                weights = torch.linalg.solve(A_mat, b_vec)

                # Evaluate Field (O(M*N))
                if BackendTensor.pykeops_eval_enabled:
                    exported_fields = symbolic_evaluator(solver_input, weights, options)
                else:
                    exported_fields = generic_evaluator(solver_input, weights, options)

                # Post-process results
                exported_fields.set_structure_values(
                    reference_sp_position=TensorsStructure.from_tensor_structure_subset(root_data_descriptor, i).reference_sp_position,
                    slice_feature=interpolation_input_i.slice_feature,
                    grid_size=interpolation_input_i.grid.len_all_grids
                )
                exported_fields.debug = solver_input.debug

            else:
                # --- B. External Function ---
                weights = None
                xyz = interpolation_input_i.grid.values
                exported_fields = _interpolate_external_function(
                    root_data_descriptor.stack_structure.interp_function, xyz
                )
                exported_fields.set_structure_values(
                    reference_sp_position=None,
                    slice_feature=None,
                    grid_size=xyz.shape[0]
                )

            # 3. Segmentation & Activation
            if root_data_descriptor.stack_structure.segmentation_function is not None:
                sigmoid_slope = root_data_descriptor.stack_structure.segmentation_function(solver_input.xyz_to_interpolate)
            else:
                sigmoid_slope = options.sigmoid_slope

            from ...modules.activator import activator_interface
            values_block = activator_interface.activate_formation_block(
                exported_fields, interpolation_input_i.unit_values, sigmoid_slope=sigmoid_slope
            )

            output = ScalarFieldOutput(
                weights=weights,
                grid=interpolation_input_i.grid,
                exported_fields=exported_fields,
                values_block=values_block,
                stack_relation=interpolation_input_i.stack_relation
            )
            all_scalar_fields_outputs[i] = output

            # 4. Update Shared Block (In-place GPU write)
            if interpolation_input_i.stack_relation is StackRelationType.FAULT:
                fault_data = interpolation_input_i.fault_values
                val_min = BackendTensor.t.min(output.values_on_all_xyz, axis=1).reshape(-1, 1)
                shifted_vals = (output.values_on_all_xyz - val_min)

                if fault_data.finite_faults_defined:
                    finite_fault_scalar = fault_data.finite_fault_data.apply(points=solver_input.xyz_to_interpolate)
                    fault_scalar_field = shifted_vals * finite_fault_scalar
                else:
                    fault_scalar_field = shifted_vals

                all_stack_values_block[i, :] = fault_scalar_field
            else:
                all_stack_values_block[i, :] = output.values_on_all_xyz

            # 5. Record Event (Stack Finished)
            events[i].record(stream)

    # Wait for everything to finish
    torch.cuda.synchronize()

    return all_scalar_fields_outputs
