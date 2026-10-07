"""Orchestrate detached, cell-local contact operations without changing legacy modes."""

import numpy as np

from ...modules.dual_contouring.contact_cells import finalize_cell_vertices, reconcile_cell_vertices
from ...modules.dual_contouring.contact_topology import build_contact_relations, contact_surface_roles, reconcile_cell_faces
from ...modules.dual_contouring.weighted_qef_setup_multicore import find_and_inject_multi_surface_constraints_multicore


def _contact_relations(surface_metadata, stacks):
    levels = [np.array([value for stack, _, value in sorted(surface_metadata) if stack == index])
              for index in range(stacks.n_stacks)]
    return build_contact_relations(
        [item[0] for item in surface_metadata], [item[1] for item in surface_metadata],
        stacks.masking_descriptor, stacks.faults_relations, levels,
    )


def prepare_contact_constraints(dc_data, cell_coordinates, surface_metadata, data_descriptor, base_number):
    """Reuse fault QEF constraints, with explicit empty partner sets for other surfaces."""
    stacks = data_descriptor.stack_structure
    surface_to_stack = [item[0] for item in surface_metadata]
    relations = _contact_relations(surface_metadata, stacks)
    _, fault_pairs, _ = relations
    if len(fault_pairs):
        fault_surfaces, _ = contact_surface_roles(surface_to_stack, stacks.masking_descriptor, stacks.faults_relations)
        partners = [set() for _ in dc_data]
        for controller, target in fault_pairs:
            if fault_surfaces[controller] and fault_surfaces[target]:
                continue
            partners[controller].add(target)
            partners[target].add(controller)
        find_and_inject_multi_surface_constraints_multicore(
            dc_data_list=dc_data, left_right_per_mesh=cell_coordinates,
            base_number=base_number, surface_to_stack=surface_to_stack,
            faults_relations=stacks.faults_relations,
            allowed_partners_per_surface=partners,
        )
    return relations


def reconcile_contact_meshes(all_meshes, cell_coordinates, surface_metadata, data_descriptor,
                             corner_ownership, *, contact_relations=None):
    """Keep vertex/cell correspondence while applying relation-directed local topology."""
    stacks = data_descriptor.stack_structure
    surface_to_stack = [item[0] for item in surface_metadata]
    if contact_relations is None:
        contact_relations = _contact_relations(surface_metadata, stacks)
    allowed_pairs, fault_pairs, truncation_pairs = contact_relations
    _, ordinary_surfaces = contact_surface_roles(surface_to_stack, stacks.masking_descriptor, stacks.faults_relations)
    surface_ids = [(item[0], item[1]) for item in surface_metadata]
    vertices, contact_ids, position_report = reconcile_cell_vertices(
        [mesh.vertices for mesh in all_meshes], cell_coordinates, surface_to_stack,
        surface_ids, allowed_pairs, fault_pairs,
        contact_eligible=[np.any(owned, axis=1) for owned in corner_ownership],
    )
    faces, topology_report = reconcile_cell_faces(
        [mesh.edges for mesh in all_meshes], contact_ids, corner_ownership,
        position_report['fault_overlap_vertices'], truncation_pairs,
        ownership_targets=np.flatnonzero(ordinary_surfaces),
    )
    vertices, contact_ids, finalization_report = finalize_cell_vertices(
        [mesh.vertices for mesh in all_meshes], vertices, contact_ids, faces, surface_ids, fault_pairs,
    )
    position_report.update(finalization_report)
    for index, mesh in enumerate(all_meshes):
        mesh.vertices, mesh.edges = vertices[index], faces[index]
        mesh.contact_report = dict(
            position_report, topology=topology_report, contact_ids=contact_ids[index],
            status='reconciled' if np.any(contact_ids[index] >= 0) else 'no_contact',
            cell_vertex_correspondence=True, differentiable_topology=False,
            extraction_metadata='dc_data and vertices_tensor describe the original solve; support_report describes pre-contact triangulation',
        )
