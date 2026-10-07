"""Bounded, detached Model7 bank queries and pre-emission bank-side skin experiment.

Capture is process-local and not thread-safe. No production solve or fault
segmentation is replaced. Fault skins are tagged per bank, not a single
conforming fault mesh. Unsupported competing contacts fail before emission.
"""

import copy
from unittest.mock import patch

import numpy as np

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.internal_structs import EvaluatorInput
from gempy_engine.core.data.stack_relation_type import StackRelationType as R


def capture_model7(interpolation_input, options, descriptor):
    """Run the unmodified solve, retaining root-level custom-grid metadata."""
    from gempy_engine.API.model.model_api import compute_model
    from gempy_engine.API.interp_single import _interp_single_feature as feature

    if BackendTensor.engine_backend is not AvailableBackends.numpy or BackendTensor.use_gpu or \
            BackendTensor.use_pykeops or np.dtype(BackendTensor.dtype) != np.dtype('float64'):
        raise ValueError('unsupported_backend: NumPy CPU float64 without KeOps required')
    relations = descriptor.stack_structure
    if tuple(relations.masking_descriptor) != (R.FAULT, R.ERODE, R.BASEMENT) or not np.array_equal(
            relations.faults_relations, [[False, True, True], [False, False, False], [False, False, False]]):
        raise ValueError('unsupported_model: one infinite fault affecting both Model7 stacks required')
    if interpolation_input.fault_values is not None and interpolation_input.fault_values.finite_fault_defined:
        raise ValueError('unsupported_finite_fault')
    if relations.faults_input_data is not None and any(
            f is not None and f.finite_fault_defined for f in relations.faults_input_data):
        raise ValueError('unsupported_finite_fault')
    captures = []
    evaluate = feature._evaluate_sys_eq

    def record(eval_input, weights, evaluation_options, grid=None):
        captures.append((copy.deepcopy(eval_input), np.array(weights, copy=True),
                         evaluation_options.model_copy(deep=True)))
        return evaluate(eval_input, weights, evaluation_options, grid=grid)

    with patch.object(feature, '_evaluate_sys_eq', record):
        solution = compute_model(interpolation_input, options, descriptor)
    if len(captures) < 3:
        raise ValueError('unsupported_evaluator: no three-stack cokriging capture')
    captures = captures[:3]
    if [c[0].fault_internal.n_faults for c in captures] != [0, 1, 1]:
        raise ValueError('unsupported_fault_count: expected one independent and two dependent fields')
    for eval_input, _, _ in captures[1:]:
        faults = eval_input.fault_internal
        values = faults.fault_values_everywhere
        if faults.finite_fault_defined or not np.isfinite(values).all() or \
                not np.isclose(values.min(), 0, atol=1e-12) or not np.isclose(values.max(), 1, atol=1e-12):
            raise ValueError('unverified_fault_banks: shifted production drift must span 0 and 1')
    fault_output = solution.root_output.outputs[0]
    fault_scalar = np.asarray(fault_output.exported_fields.scalar_field)
    coordinates = captures[0][0].xyz_to_interpolate[:len(fault_scalar)]
    design = np.column_stack((coordinates, np.ones(len(coordinates))))
    fit = np.linalg.lstsq(design, fault_scalar, rcond=None)[0]
    if np.max(np.abs(design @ fit-fault_scalar)) > 1e-10:
        raise ValueError('unsupported_nonplanar_fault: bounded infinite planar separator required')
    separator = fault_scalar-fault_output.scalar_field_at_sp[0]
    away = np.abs(separator) > 1e-10
    for eval_input, _, _ in captures[1:]:
        drift = eval_input.fault_internal.fault_values_everywhere[0, :len(separator)]
        if not np.array_equal(drift[away] > .5, separator[away] < 0):
            raise ValueError('unsupported_bank_orientation: Model7 drift 0 must be positive scalar bank')
    return solution, captures


def query_bank(capture, xyz, bank=None):
    """Change ONLY evaluation coordinates and query drift, preserving solved data.

    Reference/rest/SP fault values remain bitwise copies of the original solve.
    Actual engine kernel gradients are evaluated with constant bank drift.
    Coordinates are not displaced to either side of the geometric separator.
    """
    from gempy_engine.API.interp_single._interp_scalar_field import _evaluate_sys_eq

    original, weights, options = capture
    xyz = np.array(xyz, dtype=np.float64, copy=True)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('invalid_query_points')
    view = copy.copy(original)
    view.xyz_to_interpolate = xyz
    if original.fault_internal.n_faults:
        if bank not in (0, 1):
            raise ValueError('invalid_bank: dependent queries require explicit 0 or 1')
        if isinstance(view, EvaluatorInput):
            view.solver_input = copy.copy(original.solver_input)
            view.solver_input.fault_internal = copy.deepcopy(original.fault_internal)
        else:
            view.fault_internal = copy.deepcopy(original.fault_internal)
        view.fault_internal.fault_values_everywhere = np.full((1, len(xyz)), bank, dtype=np.float64)
    elif bank is not None:
        raise ValueError('invalid_bank: independent fault field has no drift')
    exported = _evaluate_sys_eq(view, weights, options)
    scalar = np.array(exported._scalar_field, dtype=np.float64, copy=True)
    gradient = np.stack([np.array(g, dtype=np.float64, copy=True) for g in
                         (exported._gx_field, exported._gy_field, exported._gz_field)], axis=-1)
    if not np.isfinite(scalar).all() or not np.isfinite(gradient).all():
        raise ValueError('nonfinite_bank_query')
    return scalar, gradient


def extract_bank_skin(axes, samples, gradients, levels, bank, *, include_reference=True, center_samples=None):
    """Joint bank-side separator/horizon incidence; never reinterpret fault as erosion.

    Input fields are the independent fault and two forced-bank dependent fields.
    Directed fault edges and erosion edges are separate metadata. Composite
    bank/erosion ownership is resolved before quads/triangles, or rejected.
    Finite faults, multiple faults and adaptive grids are not supported.
    """
    from gempy_engine.API.dual_contouring.topology_aware_extraction import _extract_topology_aware

    if bank not in (0, 1):
        raise ValueError('invalid_bank')
    if len(samples) != 3 or [len(v) for v in levels] != [1, 1, 2]:
        raise ValueError('unsupported_surface_structure')
    coordinates = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    design = np.column_stack((coordinates, np.ones(len(coordinates))))
    fault = np.asarray(samples[0]).reshape(-1)
    if len(fault) != len(coordinates) or not np.isfinite(fault).all():
        raise ValueError('invalid_fault_samples')
    fit = np.linalg.lstsq(design, fault, rcond=None)[0]
    if np.max(np.abs(design @ fit-fault)) > 1e-10:
        raise ValueError('unsupported_nonplanar_fault')
    # Separator is positive on the excluded bank, so all controller constraints
    # use negative fields. Flipping its orientation does not change its geometry.
    # Model7's reversed unit IDs make shifted drift 0 the positive scalar bank.
    sign = -1 if bank == 0 else 1
    separator = sign * (samples[0] - levels[0][0])
    erosion = samples[1] - levels[1][0]
    fields = np.array([separator, samples[1], samples[2], samples[2]])
    normals = np.array([sign * gradients[0], gradients[1], gradients[2], gradients[2]])
    ownership = np.array([np.ones_like(separator), -separator,
                          np.minimum(-separator, -erosion), np.minimum(-separator, -erosion)])
    groups, indices = [0, 1, 2, 2], [0, 0, 0, 1]
    fault_pairs = {(0, 1), (0, 2), (0, 3)}
    erosion_pairs = {(1, 2), (1, 3)}
    allowed = np.zeros((4, 4), dtype=bool)
    for c, t in fault_pairs | erosion_pairs:
        allowed[c, t] = allowed[t, c] = True
    interior = None
    if center_samples is not None:
        interior = np.array([sign*(center_samples[0]-levels[0][0]),
                             center_samples[1]-levels[1][0],
                             center_samples[2]-levels[2][0], center_samples[2]-levels[2][1]])
    result = _extract_topology_aware(
        axes, fields, groups, indices, [R.FAULT, R.ERODE, R.BASEMENT],
        [np.array([0.]), levels[1], levels[2]], ownership=ownership, gradient_samples=normals,
        sampled_junctions=True, include_reference=include_reference,
        interior_samples=interior,
        separator_contacts=(allowed, fault_pairs | erosion_pairs),
    )
    # Disjoint per-bank index spaces are intentional, including coincident skins.
    result['bank'] = bank
    result['tagged_vertex_keys'] = [(bank, key) for key in result['vertex_keys']]
    if include_reference:
        result['reference']['tagged_vertex_keys'] = [(bank, key) for key in result['reference']['vertex_keys']]
    result['surface_ids'] = [(bank, g, s) for g, s in zip(groups, indices)]
    result['directed_fault_pairs'] = sorted(fault_pairs)
    result['directed_erosion_pairs'] = sorted(erosion_pairs)
    result['diagnostics'].update(interface='bank_side_fault_skin_not_single_conforming_fault_mesh',
                                 bank=bank, finite_faults='unsupported', adaptive='unsupported',
                                 field_contract='bank_conditioned_sampled_monotone_or_bilinear_extrusion',
                                 fault_separator='explicit_directed_geometric_controller_not_erosion')
    return result
