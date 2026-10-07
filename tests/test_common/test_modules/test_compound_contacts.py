"""Compound topology/finalization regressions with nondegenerate support faces."""

from itertools import permutations

import numpy as np
import pytest

from gempy_engine.API.dual_contouring.contact_reconciliation import reconcile_contact_meshes
from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("competitor", [False, True])
@pytest.mark.parametrize("near_fault", [False, True])
def test_removed_middle_patch_keeps_supported_attachment_without_regrouping(dtype, competitor, near_fault):
    # A controls B and C; B's redundant patch disappears, but C retains an
    # attachment along A's first edge. The third shared cell becomes unsupported.
    triangle = np.array([[0, 1, 2]], dtype=np.int64)
    positions = [np.array([[0., 0., z], [1., 0., z], [0., 1., z]], dtype=dtype)
                 for z in (.4, .5, .6)]
    positions[2] = np.vstack([positions[2], np.array([[.5, -1., .6]], dtype=dtype)])
    common = np.array([[4, 4, 4], [5, 4, 4], [4, 5, 4]], dtype=np.int64)
    cells = [common.copy(), common.copy(), np.vstack([common, [[4, 3, 4]]])]
    faces = [triangle.copy(), triangle.copy(), np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int64)]
    metadata = [(0, 0, .4), (1, 0, .5), (2, 0, .6)]
    relations = [StackRelationType.ERODE, StackRelationType.ERODE, StackRelationType.BASEMENT]
    counts = [1, 1, 1]
    if competitor:
        positions.append(np.array([[0., 0., .9], [1., 0., .9], [0., 1., .9]], dtype=dtype))
        cells.append(common.copy())
        faces.append(triangle.copy())
        metadata.append((1, 1, .9))
        counts[1] = 2
    if near_fault:
        positions.append(np.array([[.5, -1., .62], [1.5, -1., .62], [.5, -2., .62]], dtype=dtype))
        cells.append(np.array([[4, 3, 4], [5, 3, 4], [4, 2, 4]], dtype=np.int64))
        faces.append(triangle.copy())
        metadata.append((3, 0, .62))
        counts.append(1)
        relations.append(StackRelationType.FAULT)
    faults = np.zeros((len(counts), len(counts)), dtype=bool)
    if near_fault:
        faults[3, 2] = True
    descriptor = InputDataDescriptor(
        TensorsStructure(np.array([], dtype=int)),
        StacksStructure(np.zeros(len(counts), dtype=int), np.zeros(len(counts), dtype=int),
                        np.array(counts), relations, faults_relations=faults),
    )
    baseline = None
    for order in permutations(range(len(positions))):
        meshes = [DualContouringMesh(positions[i].copy(), faces[i].copy()) for i in order]
        owned = [np.ones((len(cells[i]), 8), dtype=bool) for i in order]
        reconcile_contact_meshes(meshes, [cells[i] for i in order], [metadata[i] for i in order],
                                 descriptor, owned)
        result = {i: mesh for i, mesh in zip(order, meshes)}
        assert len(result[1].edges) == 0
        np.testing.assert_array_equal(result[1].vertices, positions[1])
        assert np.all(result[1].contact_report['contact_ids'] == -1)
        assert result[0].contact_report['unsupported_contact_member_count'] == 4
        assert result[0].contact_report['dissolved_contact_count'] == 1
        assert sum(item['shared_patch_removed_count']
                   for item in result[0].contact_report['topology']['per_surface']) == 2
        mean = ((positions[0][:2].astype(np.float64) + positions[2][:2]) / 2).astype(dtype)
        for surface in (0, 2):
            np.testing.assert_array_equal(result[surface].vertices[:2], mean)
            np.testing.assert_array_equal(result[surface].vertices[2], positions[surface][2])
            assert result[surface].contact_report['contact_ids'][2] == -1
        a_ids = result[0].contact_report['contact_ids'][:2]
        c_ids = result[2].contact_report['contact_ids'][:2]
        assert np.all(a_ids >= 0)
        np.testing.assert_array_equal(a_ids, c_ids)
        if competitor:
            np.testing.assert_array_equal(result[3].vertices, positions[3])
            assert np.all(result[3].contact_report['contact_ids'] == -1)
            np.testing.assert_array_equal(result[3].edges, faces[3])
        if near_fault:
            fault = len(positions) - 1
            np.testing.assert_array_equal(result[2].vertices[3], positions[fault][0])
            assert result[2].contact_report['contact_ids'][3] == result[fault].contact_report['contact_ids'][0]
        # Both supported meshes contain the exact attachment edge. Retained
        # faces keep their original indices and orientation after the new mean.
        for surface, mesh in result.items():
            if not len(mesh.edges):
                continue
            expected_faces = faces[surface][1:] if surface == 2 else faces[surface]
            np.testing.assert_array_equal(mesh.edges, expected_faces)
            before, after = positions[surface][mesh.edges], mesh.vertices[mesh.edges]
            n0 = np.cross(before[:, 1] - before[:, 0], before[:, 2] - before[:, 0])
            n1 = np.cross(after[:, 1] - after[:, 0], after[:, 2] - after[:, 0])
            assert np.all(np.einsum('ij,ij->i', n0, n1) > 0)
        current = [(result[i].vertices.copy(), result[i].edges.copy(),
                    result[i].contact_report['contact_ids'].copy()) for i in range(len(positions))]
        if baseline is None:
            baseline = current
        for expected, actual in zip(baseline, current):
            for before, after in zip(expected, actual):
                np.testing.assert_array_equal(after, before)
