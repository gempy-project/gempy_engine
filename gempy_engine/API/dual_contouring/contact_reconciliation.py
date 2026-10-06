"""API orchestration for the initial affine, two-stack contact mode.

Computational modules receive explicit NumPy arrays and never call each other.
Extraction metadata and vertices_tensor describe the pre-reconciliation solve,
not the cell-to-vertex correspondence of the returned, retriangulated meshes.
"""

import numpy as np

from ...core.backend_tensor import BackendTensor
from ...config import AvailableBackends
from ...core.data.options.evaluation_options import MeshExtractionMaskingOptions, MeshExtentCapping
from ...core.data.stack_relation_type import StackRelationType


def prepare_planar_contact(data_descriptor, interpolation_input, options, octree_leaves):
    """Validate support before extraction and fit planes from leaf corner samples."""
    if BackendTensor.engine_backend != AvailableBackends.numpy or BackendTensor.use_gpu:
        raise ValueError("contact_aware initially supports only the NumPy CPU backend; PyTorch QEF is unsupported")
    if np.dtype(BackendTensor.dtype) != np.dtype("float64"):
        raise ValueError("contact_aware initially requires float64 extraction; float32 is unsupported")
    stacks = data_descriptor.stack_structure
    relations = tuple(stacks.masking_descriptor)
    if stacks.n_stacks not in (1, 2):
        raise ValueError("contact_aware supports only one or two stacks")
    if len(octree_leaves.outputs) != stacks.n_stacks:
        raise ValueError("contact_aware requires one leaf output per stack")
    if any(r in (StackRelationType.FAULT, StackRelationType.NULL_SPACE) for r in relations):
        raise ValueError("contact_aware does not support faults or null-space stacks")
    if (stacks.faults_relations is not None and np.any(BackendTensor.t.to_numpy(stacks.faults_relations))) or \
            (stacks.faults_input_data is not None and any(f is not None for f in stacks.faults_input_data)) or \
            not interpolation_input.not_fault_input:
        raise ValueError("contact_aware does not support fault data or fault relations")
    if np.any(np.asarray(stacks.number_of_surfaces_per_stack) != 1) or any(
            output.scalar_field_at_sp.shape[0] != 1 for output in octree_leaves.outputs):
        raise ValueError("contact_aware requires exactly one surface per stack")
    if options.evaluation_options.mesh_extraction_masking_options != MeshExtractionMaskingOptions.INTERSECT:
        raise ValueError("contact_aware requires INTERSECT extraction masking")
    if MeshExtentCapping(options.evaluation_options.mesh_extraction_extent_capping) != MeshExtentCapping.NONE:
        raise ValueError("contact_aware does not support extent capping")
    if stacks.n_stacks == 1:
        return None
    match relations:
        case (StackRelationType.ERODE, StackRelationType.BASEMENT):
            controller, truncated, retained_sign = 0, 1, -1
        case (StackRelationType.ONLAP, StackRelationType.BASEMENT):
            controller, truncated, retained_sign = 1, 0, 1
        case _:
            raise ValueError("contact_aware supports only [ERODE, BASEMENT] or [ONLAP, BASEMENT]")

    from ...modules.dual_contouring.contact_planes import fit_contact_plane

    points = BackendTensor.t.to_numpy(octree_leaves.grid.corners_grid.values)
    planes = []
    for stack_index, output in enumerate(octree_leaves.outputs):
        values = BackendTensor.t.to_numpy(output.exported_fields.scalar_field[output.grid.corners_grid_slice])
        isovalue = float(BackendTensor.t.to_numpy(output.scalar_field_at_sp)[0])
        try:
            planes.append(fit_contact_plane(points, values, isovalue, tolerance=1e-8))
        except ValueError as error:
            raise ValueError(f"contact_aware stack {stack_index}: {error}") from error
    return controller, truncated, retained_sign, planes


def reconcile_contact_meshes(meshes, contact):
    """Replace only final mesh arrays; preserve extraction and support metadata."""
    if contact is None:
        return
    from ...modules.dual_contouring.contact_geometry import reconcile_planar_contact

    controller, truncated, retained_sign, planes = contact
    if len(meshes) != 2:
        raise ValueError("contact_aware requires two extracted meshes for a two-stack contact")
    cm, tm = meshes[controller], meshes[truncated]
    try:
        cv, cf, tv, tf, report = reconcile_planar_contact(
            cm.vertices, cm.edges, tm.vertices, tm.edges,
            planes[controller], planes[truncated], retained_sign, tolerance=1e-8
        )
    except ValueError as error:
        raise ValueError(f"contact_aware geometry/support error: {error}") from error
    cm.vertices, cm.edges = cv, cf
    tm.vertices, tm.edges = tv, tf
    for mesh, role in ((cm, "controller"), (tm, "truncated")):
        mesh.contact_report = dict(
            report, role=role, controller_stack_index=controller,
            truncated_stack_index=truncated, retained_sign=retained_sign,
            extraction_metadata="dc_data, support_report and vertices_tensor describe pre-reconciliation extraction",
            cell_vertex_correspondence=False,
            differentiable_topology=False,
        )
