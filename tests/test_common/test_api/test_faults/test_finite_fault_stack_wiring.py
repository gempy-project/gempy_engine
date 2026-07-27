import copy

import numpy as np
import pytest

from gempy_engine import compute_model
from gempy_engine.API.interp_single._aux_faults_ops import (
    _modify_faults_values_output,
    _options_with_finite_fault_gradients,
)
from gempy_engine.API.interp_single._multi_scalar_field_manager import _compute_independent_chunks
from gempy_engine.core.data import FiniteFault, InterpolationOptions
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.scalar_field_output import ScalarFieldOutput
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure


def test_modify_fault_values_applies_projected_taper():
    finite_fault = FiniteFault(center=(0.0, 0.0, 0.0), strike_radius=1.0, dip_radius=1.0)
    fault_input = FaultsData.from_user_input(thickness=None, finite_fault=finite_fault)
    points = np.array([
            [2.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
    ])
    exported_fields = ExportedFields(
        _scalar_field=np.full(3, 2.0),
        _gx_field=np.zeros(3),
        _gy_field=np.zeros(3),
        _gz_field=np.ones(3),
        _grid_size=3,
        _scalar_field_at_surface_points=np.array([0.0]),
    )
    output = ScalarFieldOutput(
        weights=None,
        grid=None,
        exported_fields=exported_fields,
        stack_relation=StackRelationType.FAULT,
        values_block=np.array([[1.0, 3.0, 3.0]]),
    )

    tapered_values = _modify_faults_values_output(fault_input, output, points)

    assert np.allclose(tapered_values, [[0.0, 2.0, 0.0]])


def test_finite_fault_gradient_options_do_not_mutate_input():
    options = InterpolationOptions.from_args(range=1.0, c_o=1.0, mesh_extraction=False)
    fault_input = FaultsData.from_user_input(
        thickness=None,
        finite_fault=FiniteFault(center=(0.0, 0.0, 0.0)),
    )

    finite_fault_options = _options_with_finite_fault_gradients(options, fault_input)

    assert finite_fault_options is not options
    assert finite_fault_options.evaluation_options is not options.evaluation_options
    assert finite_fault_options.evaluation_options.compute_scalar_gradient is True
    assert options.evaluation_options.compute_scalar_gradient is False


def test_finite_fault_stack_is_isolated_before_dependent_stack():
    finite_fault_data = FaultsData.from_user_input(
        thickness=None,
        finite_fault=FiniteFault(center=(0.0, 0.0, 0.0)),
    )
    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([3, 3, 3]),
        number_of_orientations_per_stack=np.array([1, 1, 1]),
        number_of_surfaces_per_stack=np.array([1, 1, 1]),
        masking_descriptor=[StackRelationType.FAULT, StackRelationType.ERODE, StackRelationType.ERODE],
        faults_relations=np.array([
                [False, False, True],
                [False, False, False],
                [False, False, False],
        ]),
        faults_input_data=[finite_fault_data, None, None],
    )

    chunks = _compute_independent_chunks(stack_structure, len_grid=8)

    assert [0] in chunks
    assert chunks.index([0]) < next(i for i, chunk in enumerate(chunks) if 2 in chunk)
    assert all(0 not in chunk or chunk == [0] for chunk in chunks)


def test_finite_fault_is_wired_into_dependent_stack(one_fault_model):
    interpolation_input, data_descriptor, options = copy.deepcopy(one_fault_model)
    options.evaluation_options.number_octree_levels = 1
    options.evaluation_options.mesh_extraction = False
    options.evaluation_options.compute_scalar_gradient = False

    baseline = compute_model(
        copy.deepcopy(interpolation_input),
        options,
        copy.deepcopy(data_descriptor),
    )

    fault_points = interpolation_input.surface_points.sp_coords[:9]
    finite_fault = FiniteFault(
        center=tuple(np.mean(fault_points, axis=0)),
        strike_radius=0.75,
        dip_radius=0.75,
    )
    data_descriptor.stack_structure.faults_input_data = [
        FaultsData.from_user_input(thickness=None, finite_fault=finite_fault),
        None,
        None,
    ]

    finite = compute_model(interpolation_input, options, data_descriptor)

    finite_fault_output = finite.octrees_output[0].outputs[0].exported_fields
    baseline_dependent = baseline.octrees_output[0].outputs[2].exported_fields.scalar_field
    finite_dependent = finite.octrees_output[0].outputs[2].exported_fields.scalar_field
    assert finite_fault_output.gx_field is not None
    assert finite_fault_output.gy_field is not None
    assert finite_fault_output.gz_field is not None
    assert options.evaluation_options.compute_scalar_gradient is False
    assert not np.allclose(finite_dependent, baseline_dependent)


def test_finite_fault_flat_stack_matches_serial(one_fault_model, monkeypatch):
    pytest.importorskip("pykeops")
    interpolation_input, data_descriptor, options = copy.deepcopy(one_fault_model)
    options.evaluation_options.number_octree_levels = 1
    options.evaluation_options.mesh_extraction = False
    options.evaluation_options.compute_scalar_gradient = False

    fault_points = interpolation_input.surface_points.sp_coords[:9]
    data_descriptor.stack_structure.faults_input_data = [
        FaultsData.from_user_input(
            thickness=None,
            finite_fault=FiniteFault(
                center=tuple(np.mean(fault_points, axis=0)),
                strike_radius=0.75,
                dip_radius=0.75,
            ),
        ),
        None,
        None,
    ]

    monkeypatch.setenv("GEMPY_FLAT_STACKS", "False")
    serial = compute_model(
        copy.deepcopy(interpolation_input),
        options,
        copy.deepcopy(data_descriptor),
    )

    original_use_pykeops = BackendTensor.use_pykeops
    BackendTensor.use_pykeops = True
    monkeypatch.setenv("GEMPY_FLAT_STACKS", "True")
    try:
        flat = compute_model(
            copy.deepcopy(interpolation_input),
            options,
            copy.deepcopy(data_descriptor),
        )
    finally:
        BackendTensor.use_pykeops = original_use_pykeops

    serial_dependent = serial.octrees_output[0].outputs[2].exported_fields.scalar_field
    flat_dependent = flat.octrees_output[0].outputs[2].exported_fields.scalar_field
    # Dense and PyKeOps routes use different reduction orders.
    assert np.allclose(flat_dependent, serial_dependent, atol=5e-3, rtol=1e-2)
    assert flat.octrees_output[0].outputs[0].exported_fields.gx_field is not None
    assert options.evaluation_options.compute_scalar_gradient is False
