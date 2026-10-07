"""Characterize finite drift support, not physical displacement or mesh correctness."""

import copy

import numpy as np
import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.API.interp_single._aux_faults_ops import _modify_faults_values_output
from gempy_engine.core.backend_tensor import AvailableBackends, BackendTensor
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.finite_fault import FiniteFault
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.modules.faults.finite_faults import (
    get_ellipsoid_distance,
    get_local_frame,
    project_points_onto_surface,
)


def test_finite_fault_extraction_support(one_fault_model, monkeypatch):
    if (BackendTensor.engine_backend is not AvailableBackends.numpy
            or BackendTensor.use_gpu or BackendTensor.use_pykeops):
        pytest.skip("Characterization requires the NumPy CPU backend without PyKeOps")

    monkeypatch.setenv("GEMPY_FLAT_STACKS", "False")
    monkeypatch.setenv("SET_RAW_SCALAR_FIELDS_IN_SOLUTION", "True")
    interpolation_input, descriptor, options = copy.deepcopy(one_fault_model)
    options.evaluation_options.number_octree_levels = 2
    options.evaluation_options.number_octree_levels_surface = 2
    options.evaluation_options.mesh_extraction = True
    options.evaluation_options.compute_scalar_gradient = True

    baseline = compute_model(
        copy.deepcopy(interpolation_input), copy.deepcopy(options), copy.deepcopy(descriptor)
    )
    fault_point_count = int(descriptor.stack_structure.number_of_points_per_stack[0])
    finite_fault = FiniteFault(
        center=tuple(np.mean(interpolation_input.surface_points.sp_coords[:fault_point_count], axis=0)),
        strike_radius=0.75,
        dip_radius=0.75,
    )
    finite_descriptor = copy.deepcopy(descriptor)
    finite_descriptor.stack_structure.faults_input_data = [
        FaultsData.from_user_input(thickness=None, finite_fault=finite_fault), None, None
    ]
    finite = compute_model(
        copy.deepcopy(interpolation_input), copy.deepcopy(options), finite_descriptor
    )
    assert len(baseline.octrees_output) == len(finite.octrees_output) == 2

    # Compare the shared root queries; dependent-stack refinement can change leaf grids.
    baseline_root = baseline.octrees_output[0]
    finite_root = finite.octrees_output[0]
    np.testing.assert_allclose(baseline_root.grid.values, finite_root.grid.values)
    baseline_fault = baseline_root.outputs[0]
    finite_root_fault = finite_root.outputs[0]
    np.testing.assert_allclose(
        baseline_fault.exported_fields.scalar_field_everywhere,
        finite_root_fault.exported_fields.scalar_field_everywhere,
        rtol=1e-10, atol=1e-10,
    )
    np.testing.assert_allclose(baseline_fault.scalar_field_at_sp, finite_root_fault.scalar_field_at_sp)
    isovalue = float(finite_root_fault.scalar_field_at_sp[0])
    np.testing.assert_array_equal(
        baseline_fault.exported_fields.scalar_field > isovalue,
        finite_root_fault.exported_fields.scalar_field > isovalue,
    )
    root_taper = finite_root_fault.scalar_fields.finite_fault_scalar
    assert root_taper is not None
    assert np.all(np.isfinite(root_taper))
    assert np.all((root_taper >= 0) & (root_taper <= 1))
    assert np.any(root_taper == 0) and np.any(root_taper > 0)
    # Exported values_block is untapered; the drift transform returns a separate array.
    np.testing.assert_allclose(baseline_fault.values_block, finite_root_fault.values_block)
    root_xyz = np.concatenate([finite_root.grid.values, interpolation_input.surface_points.sp_coords])
    baseline_drift = _modify_faults_values_output(
        FaultsData(), copy.deepcopy(baseline_fault.scalar_fields), root_xyz
    )
    finite_drift = _modify_faults_values_output(
        finite_descriptor.stack_structure.faults_input_data[0],
        copy.deepcopy(finite_root_fault.scalar_fields), root_xyz,
    )
    np.testing.assert_allclose(finite_drift, baseline_drift * root_taper, rtol=1e-10, atol=1e-10)
    drift_delta = float(np.max(np.abs(baseline_drift - finite_drift)))
    dependent_delta = float(np.max(np.abs(
        baseline_root.outputs[2].exported_fields.scalar_field
        - finite_root.outputs[2].exported_fields.scalar_field
    )))
    assert drift_delta > 1e-6
    assert dependent_delta > 1e-6

    baseline_meshes = [mesh for mesh in baseline.dc_meshes if mesh.stack_index == 0]
    finite_meshes = [mesh for mesh in finite.dc_meshes if mesh.stack_index == 0]
    assert len(baseline_meshes) == len(finite_meshes) == 1
    baseline_mesh, finite_mesh = baseline_meshes[0], finite_meshes[0]
    for mesh in (baseline_mesh, finite_mesh):
        assert mesh.surface_index == 0
        assert len(mesh.vertices) > 0 and len(mesh.edges) > 0
        assert np.all(np.isfinite(mesh.vertices))
        assert np.all((mesh.edges >= 0) & (mesh.edges < len(mesh.vertices)))
        assert mesh.isovalue == pytest.approx(isovalue)
    np.testing.assert_allclose(baseline_mesh.vertices, finite_mesh.vertices, rtol=1e-6, atol=1e-5)
    np.testing.assert_array_equal(baseline_mesh.edges, finite_mesh.edges)

    # Recover the frame selected by production from the finite leaf's macro queries.
    leaf_fault = finite.octrees_output[-1].outputs[0]
    fields = leaf_fault.exported_fields
    reference_size = fields._macro_reference_size
    reference_xyz = np.concatenate([
        leaf_fault.grid.values, interpolation_input.surface_points.sp_coords
    ])
    assert reference_xyz.shape == (reference_size, 3)
    gradients = (fields.gx_field_everywhere, fields.gy_field_everywhere, fields.gz_field_everywhere)
    reference_gradients = tuple(component[:reference_size] for component in gradients)
    projected_reference = project_points_onto_surface(
        reference_xyz, fields.scalar_field_everywhere[:reference_size],
        reference_gradients, target_scalar_value=isovalue,
    )
    gradient_matrix = np.stack(reference_gradients, axis=-1)
    valid_gradient = np.linalg.norm(gradient_matrix, axis=1) > 1e-12
    center_distances = np.linalg.norm(projected_reference - finite_fault.center, axis=1)
    center_index = np.argmin(np.where(valid_gradient, center_distances, np.inf))
    normal = gradient_matrix[center_index]
    np.testing.assert_allclose(
        finite_fault.calculate_slip(projected_reference, normal),
        leaf_fault.scalar_fields.finite_fault_scalar[:reference_size],
        rtol=1e-10, atol=1e-10,
    )

    # Re-evaluate real extracted vertices, not edge intersections or a synthetic plane.
    # The first fault stack is independent, so an untapered probe preserves its field.
    probe_input = copy.deepcopy(interpolation_input)
    probe_input.set_temp_grid(EngineGrid.from_xyz_coords(finite_mesh.vertices.copy()))
    probe_options = copy.deepcopy(options)
    probe_options.evaluation_options.mesh_extraction = False
    probe_options.evaluation_options.number_octree_levels = 1
    probe = compute_model(probe_input, probe_options, copy.deepcopy(descriptor))
    vertex_fields = probe.octrees_output[0].outputs[0].exported_fields
    projected_vertices = project_points_onto_surface(
        finite_mesh.vertices, vertex_fields.scalar_field,
        (vertex_fields.gx_field, vertex_fields.gy_field, vertex_fields.gz_field),
        target_scalar_value=isovalue,
    )
    u, v, _ = get_local_frame(normal, finite_fault.rotation_deg)
    distance = get_ellipsoid_distance(
        projected_vertices, np.asarray(finite_fault.center), u, v,
        a=finite_fault.strike_radius, b=finite_fault.dip_radius,
    )
    vertex_taper = finite_fault.calculate_slip(projected_vertices, normal)
    unprojected_distance = get_ellipsoid_distance(
        finite_mesh.vertices, np.asarray(finite_fault.center), u, v,
        a=finite_fault.strike_radius, b=finite_fault.dip_radius,
    )
    assert np.all(np.isfinite(distance)) and np.all(np.isfinite(vertex_taper))
    assert np.all((vertex_taper >= 0) & (vertex_taper <= 1))
    assert np.any(vertex_taper > 0)
    outside = distance > 1 + 1e-6
    assert np.any(outside)
    assert np.any(outside & (unprojected_distance > 1 + 1e-6))
    np.testing.assert_array_equal(vertex_taper[outside], np.zeros(np.count_nonzero(outside)))
    referenced_vertices = np.unique(finite_mesh.edges)
    assert np.any(outside[referenced_vertices])

    print(f"\nFinite-fault characterization: levels=2, center={finite_fault.center}, radii=(0.75, 0.75)")
    print(f"Root fault isovalue={isovalue:.9g}; max scalar delta="
          f"{np.max(np.abs(baseline_fault.exported_fields.scalar_field - finite_root_fault.exported_fields.scalar_field)):.9g}")
    print(f"Root taper: zero={np.count_nonzero(root_taper == 0)}/{root_taper.size}, "
          f"range=[{root_taper.min():.9g}, {root_taper.max():.9g}]; "
          f"max drift delta={drift_delta:.9g}, max dependent scalar delta={dependent_delta:.9g}")
    for label, mesh in (("baseline", baseline_mesh), ("finite", finite_mesh)):
        print(f"{label} fault mesh: vertices={len(mesh.vertices)}, triangles={len(mesh.edges)}, "
              f"bounds={np.stack([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)]).tolist()}")
    print(f"Max baseline/finite fault vertex delta={np.max(np.abs(baseline_mesh.vertices - finite_mesh.vertices)):.9g}; "
          f"unprojected support distance=[{unprojected_distance.min():.9g}, {unprojected_distance.max():.9g}]")
    print(f"Finite mesh projected support distance=[{distance.min():.9g}, {distance.max():.9g}]; "
          f"outside={np.count_nonzero(outside)}/{len(distance)}, "
          f"referenced outside={np.count_nonzero(outside[referenced_vertices])}/{len(referenced_vertices)}, "
          f"vertex taper=[{vertex_taper.min():.9g}, {vertex_taper.max():.9g}]")
    print(f"Max extracted-vertex scalar residual={np.max(np.abs(vertex_fields.scalar_field - isovalue)):.9g}; "
          f"max one-step projection distance={np.max(np.linalg.norm(projected_vertices - finite_mesh.vertices, axis=1)):.9g}")
