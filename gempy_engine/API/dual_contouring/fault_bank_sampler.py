"""Detached fixed-weight queries for one verified infinite planar fault, per fault bank.

Values are raw stack potentials, not surface-expanded fields or lithology IDs.
Bank 0 is the positive fault-potential side; bank 1 is the negative side.
"""

import copy
from itertools import product

import numpy as np

from ..interp_single._aux_faults_ops import _modify_faults_values_output
from ..interp_single._interp_scalar_field import _evaluate_sys_eq
from .fixed_weight_snapshots import (check_fixed_query_contract, check_fixed_weight_backend, fault_matrix,
                                     fixed_weight_snapshots, freeze_arrays, host)
from ...config import AvailableBackends
from ...core.backend_tensor import BackendTensor as BT
from ...core.data.kernel_classes.faults import FaultsData
from ...core.data.stack_relation_type import StackRelationType


def prepare_fault_bank_sampler(descriptor, interpolation_input, options, octree_list):
    """Return raw ``query(points, bank, stacks) -> (S x M, S x M x 3)`` fields.

    Queries touching an affected stack require bank 0 or 1; fault and
    unaffected stacks are bank-independent. Only query
    coordinates and query fault drift change; solved SP/ref/rest drift does not.
    NumPy and Torch (CPU or CUDA, optionally KeOps) float64 are supported: queries
    take and return host NumPy arrays. Unsupported contracts fail before emission.
    ``snapshots`` are detached inspection copies, never the query's private state.
    """
    check_fixed_weight_backend()
    stacks = descriptor.stack_structure
    n_stacks = stacks.n_stacks
    faults = [i for i, relation in enumerate(stacks.masking_descriptor)
              if relation is StackRelationType.FAULT]
    if len(faults) != 1:
        raise ValueError('unsupported_fault_count: exactly one independent fault required')
    fault_stack = faults[0]
    matrix = fault_matrix(stacks)
    other_rows = matrix.copy()
    other_rows[fault_stack] = False
    if other_rows.any() or matrix[:, fault_stack].any():
        raise ValueError('unsupported_fault_relations: only independent fault -> affected edges allowed')
    affected = tuple(int(i) for i in np.flatnonzero(matrix[fault_stack]))
    unaffected = tuple(i for i in range(n_stacks) if i != fault_stack and i not in affected)
    fault_inputs = check_fixed_query_contract(stacks, interpolation_input, octree_list)
    last = octree_list[-1]
    if host(last.outputs[fault_stack].scalar_field_at_sp).size != 1:
        raise ValueError('unsupported_fault_surfaces: one separator required')

    snapshots, fingerprints, private_input, grid_size = fixed_weight_snapshots(
        descriptor, interpolation_input, options, octree_list, (fault_stack,), fault_inputs)
    backend = (BT.engine_backend, BT.dtype, BT.use_gpu, BT.use_pykeops, BT.COMPUTE_GRADS)
    on_device = BT.engine_backend is not AvailableBackends.numpy

    def device(values):
        return BT.t.array(np.ascontiguousarray(values), dtype=BT.dtype_obj) if on_device else values

    # Frozen host weights stay the inspected snapshot; evaluation uses a device copy.
    device_weights = [device(weights) for _, weights, _ in snapshots]

    def evaluate_stack(stack_index, points, bank=None):
        original, _, stack_options = snapshots[stack_index]
        view = copy.copy(original)
        view.xyz_to_interpolate = device(points)
        if stack_index in affected:
            view.fault_internal = copy.copy(original.fault_internal)
            view.fault_internal.fault_values_everywhere = device(np.full((1, len(points)), bank, dtype=np.float64))
        fields = _evaluate_sys_eq(view, device_weights[stack_index], stack_options)
        values = np.array(host(fields._scalar_field), dtype=np.float64, copy=True)
        gradients = np.stack([host(g) for g in (fields._gx_field, fields._gy_field, fields._gz_field)], axis=-1)
        if values.shape != (len(points),) or gradients.shape != (len(points), 3) or not (
                np.isfinite(values).all() and np.isfinite(gradients).all()):
            raise ValueError('nonfinite_or_invalid_fault_query')
        return values, gradients

    def query(points, bank=None, stacks=None):
        """Rows follow ``stacks`` (default: every stack in order)."""
        if backend != (BT.engine_backend, BT.dtype, BT.use_gpu, BT.use_pykeops, BT.COMPUTE_GRADS):
            raise RuntimeError('fault sampler backend changed after preparation')
        stacks = tuple(range(n_stacks)) if stacks is None else tuple(stacks)
        if not stacks or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool)
                             or not 0 <= i < n_stacks for i in stacks):
            raise ValueError('invalid_query_stacks: nonempty in-bounds stack indices required')
        if (bank is not None and (not np.isscalar(bank) or bank not in (0, 1))
                or bank is None and any(i in affected for i in stacks)):
            raise ValueError('invalid_bank: affected stack queries require explicit 0 or 1')
        points = np.array(points, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('invalid_query_points: finite (M, 3) coordinates required')
        values, gradients = zip(*(evaluate_stack(i, points, bank) for i in stacks))
        return np.stack(values), np.stack(gradients)

    # Verify the separator using actual fixed-weight scalar AND gradient kernels,
    # including off-observation domain probes, not just an observation-plane fit.
    root_points = host(octree_list[0].grid.values)
    extent = host(octree_list[0].grid.octree_grid.orthogonal_extent).reshape(3, 2)
    probes = np.array(list(product(*[np.linspace(a, b, 3) for a, b in extent])))
    check_points = np.concatenate((root_points, probes))
    scalar, gradients = evaluate_stack(fault_stack, check_points)
    design = np.column_stack((check_points, np.ones(len(check_points))))
    plane = np.linalg.lstsq(design, scalar, rcond=None)[0]
    scalar_error = float(np.max(np.abs(design @ plane - scalar)))
    gradient_error = float(np.max(np.abs(gradients - gradients[0])))
    # Engine kernel gradients retain their production scaling convention.
    # Verify direction, but never replace them with a fitted plane normal.
    normal_length = np.linalg.norm(plane[:3])
    gradient_length = np.linalg.norm(gradients[0])
    alignment_error = (float(np.linalg.norm(gradients[0] / gradient_length - plane[:3] / normal_length))
                       if min(normal_length, gradient_length) > 1e-12 else float('inf'))
    if (np.linalg.matrix_rank(design) != 4 or np.linalg.norm(plane[:3]) < 1e-12 or
            scalar_error > 1e-10 or gradient_error > 1e-10 or alignment_error > 1e-10):
        raise ValueError('unsupported_nonplanar_fault: actual scalar and gradients must define a plane')
    drift_ranges = []
    for level in (octree_list[0], last):
        output = level.outputs[fault_stack].scalar_fields
        level_xyz = np.concatenate((host(output.grid.values), host(private_input.all_surface_points.sp_coords)))
        drift = host(_modify_faults_values_output(FaultsData(), output, level_xyz))[0]
        scalar_field = host(output.exported_fields._scalar_field)
        isovalue = float(host(output.scalar_field_at_sp).reshape(-1)[0])
        separator = scalar_field[:output.grid.len_all_grids] - isovalue
        grid_drift = drift[:output.grid.len_all_grids]
        away = np.abs(separator) > 1e-10
        if (not np.isfinite(drift).all() or not np.isclose(drift.min(), 0, rtol=0, atol=1e-12) or
                not np.isclose(drift.max(), 1, rtol=0, atol=1e-12) or not away.any() or
                not np.array_equal(grid_drift[away] > .5, separator[away] < 0)):
            raise ValueError('unverified_fault_banks: production drift must span 0/1 with bank 0 positive')
        drift_ranges.append((float(drift.min()), float(drift.max())))
    # Inspection snapshots do not expose mutable attributes of the closure.
    inspection = copy.deepcopy(snapshots)
    freeze_arrays(inspection)
    return dict(fault_stack=fault_stack, affected_stacks=affected, unaffected_stacks=unaffected,
                query=query, snapshots=inspection,
                diagnostics=dict(backend=f"{BT.engine_backend.name}_{'cuda' if BT.use_gpu else 'cpu'}_float64"
                                         f"{'_keops' if BT.use_pykeops else ''}", fixed_weights=True,
                                 weight_provenance='independent_production_output_snapshot',
                                 cache_fingerprints=tuple(fingerprints), plane=tuple(plane),
                                 plane_scalar_max_error=scalar_error, plane_gradient_max_error=gradient_error,
                                 plane_gradient_alignment_error=alignment_error,
                                 production_drift_ranges=tuple(drift_ranges), bank_zero_side='positive',
                                 reference_grid_size=grid_size, surface_level=len(octree_list) - 1,
                                 field_contract='raw_stack_values_and_actual_kernel_gradients'))
