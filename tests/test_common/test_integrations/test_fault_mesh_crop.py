"""Faults and the stacks around them.

Fault meshes are cropped to the rock older than the fault: younger erosions and null spaces cut the plane. Faults are
transparent to lithology ownership: an onlap stack above a fault onlaps the stack below the fault, on both sides.
"""

import numpy as np
import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.generic_grid import GenericGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine.core.data.stacks_structure import StacksStructure

EXTENT = [0, 10., 0, 2., 0, 5.]
RESOLUTION = [10, 2, 10]
LEVELS = 3
VOXEL_Z = (EXTENT[5] - EXTENT[4]) / RESOLUTION[2] / 2 ** (LEVELS - 1)
UNCONFORMITY = 3.5
#: none/pretty crop per voxel (within one voxel past the contact); joint_contacts cuts the fault on the contact.
OVERLAPS = ["none", "pretty", "joint_contacts"]


def _horizontal(z):
    return np.array([[2., .5, z], [8., .5, z]]), np.array([[5., 1., z]]), np.array([[0., 0., 1.]])


def _vertical(x):
    return np.array([[x, .5, 1.], [x, 1.5, 4.]]), np.array([[x, 1., 2.5]]), np.array([[1., 0., 0.]])


def _compute(stacks, faults_relations, overlap, null_space=None, custom=None):
    """``stacks``: (relation, (points, orientation positions, gradients)), youngest first, one surface each.

    Returns the meshes' vertices, or with ``custom`` points the unit ids there.
    """
    points = np.concatenate([s[1][0] for s in stacks])
    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([len(s[1][0]) for s in stacks]),
        number_of_orientations_per_stack=np.array([len(s[1][1]) for s in stacks]),
        number_of_surfaces_per_stack=np.ones(len(stacks), dtype=int),
        masking_descriptor=[s[0] for s in stacks],
        faults_relations=np.asarray(faults_relations, dtype=bool),
        null_space_per_stack=null_space,
    )
    tensors = TensorsStructure(number_of_points_per_surface=np.array([len(s[1][0]) for s in stacks]))
    orientations = Orientations(np.concatenate([s[1][1] for s in stacks]), np.concatenate([s[1][2] for s in stacks]))
    options = InterpolationOptions.from_args(86.6, 3571.4, uni_degree=1, i_res=4, gi_res=2, number_dimensions=3,
                                             kernel_function=AvailableKernelFunctions.cubic)
    options.evaluation_options.number_octree_levels = LEVELS
    options.evaluation_options.mesh_extraction = custom is None
    options.evaluation_options.mesh_extraction_overlap = overlap
    grid = EngineGrid(octree_grid=RegularGrid(EXTENT, RESOLUTION),
                      custom_grid=None if custom is None else GenericGrid(values=np.asarray(custom, dtype=float)))
    solutions = compute_model(InterpolationInput(SurfacePoints(points), orientations, grid, np.arange(1, 10)),
                              options, InputDataDescriptor(tensors, stack_structure))
    if custom is not None:
        return np.rint(np.asarray(BackendTensor.t.to_numpy(solutions.raw_arrays.custom))).astype(int)
    return [np.asarray(mesh.vertices) for mesh in solutions.dc_meshes]


@pytest.mark.parametrize("overlap", OVERLAPS)
def test_younger_erosion_crops_the_fault_plane(overlap):
    unconformity, fault, layer = _compute(
        [(R.ERODE, _horizontal(UNCONFORMITY)), (R.FAULT, _vertical(5.)), (R.BASEMENT, _horizontal(2.))],
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]], overlap)
    assert len(fault) and fault[:, 2].min() < 2.
    assert fault[:, 2].max() <= UNCONFORMITY + _tolerance(overlap)


@pytest.mark.parametrize("overlap", OVERLAPS)
def test_fault_without_younger_rock_spans_the_domain(overlap):
    fault, layer = _compute([(R.FAULT, _vertical(5.)), (R.BASEMENT, _horizontal(2.))], [[0, 1], [0, 0]], overlap)
    assert fault[:, 2].max() > EXTENT[5] - 2 * VOXEL_Z


@pytest.mark.parametrize("overlap", OVERLAPS)
def test_younger_fault_does_not_crop_an_older_fault(overlap):
    younger, older, layer = _compute(
        [(R.FAULT, _vertical(3.)), (R.FAULT, _vertical(7.)), (R.BASEMENT, _horizontal(2.))],
        [[0, 1, 1], [0, 0, 1], [0, 0, 0]], overlap)
    assert older[:, 2].max() > EXTENT[5] - 2 * VOXEL_Z


@pytest.mark.parametrize("overlap", OVERLAPS)
def test_younger_null_space_crops_the_fault_plane(overlap):
    fault, layer = _compute(
        [(R.ERODE, _horizontal(UNCONFORMITY)), (R.FAULT, _vertical(5.)), (R.BASEMENT, _horizontal(2.))],
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]], overlap, null_space=[True, False, False])
    assert fault[:, 2].max() <= UNCONFORMITY + _tolerance(overlap)


def _tolerance(overlap):
    return 1e-6 if overlap == "joint_contacts" else VOXEL_Z


@pytest.mark.parametrize("overlap", OVERLAPS)
def test_younger_onlap_crops_the_fault_plane(overlap):
    """The cover onlaps the basement below the fault, so the fault ends on the basement top inside the cover's
    reach: where the cover lies, above its own surface and above the basement top."""
    cover, fault, base = _compute(
        [(R.ONLAP, _horizontal(UNCONFORMITY)), (R.FAULT, _vertical(5.)), (R.BASEMENT, _horizontal(2.))],
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]], overlap)
    assert len(fault) and fault[:, 2].min() < 1.5
    assert fault[:, 2].max() <= 2. + VOXEL_Z + .5  # The faulted basement top, offset up to .5 on one side
    assert fault[:, 2].max() < UNCONFORMITY


def test_onlap_over_a_fault_onlaps_the_stack_below_on_both_sides():
    """Before, the onlap chain took the fault's field: the cover existed on one side, the fault owned the other."""
    cover = [[x, 1., z] for z in (3.2, 4.5) for x in (2., 8.)]  # Above the basement top (2), both sides of x = 5
    faulted = _compute([(R.ONLAP, _horizontal(UNCONFORMITY)), (R.FAULT, _vertical(5.)), (R.BASEMENT, _horizontal(2.))],
                       [[0, 0, 0], [0, 0, 1], [0, 0, 0]], "none", custom=cover)
    unfaulted = _compute([(R.ONLAP, _horizontal(UNCONFORMITY)), (R.BASEMENT, _horizontal(2.))],
                         [[0, 0], [0, 0]], "none", custom=cover)
    np.testing.assert_array_equal(faulted, unfaulted)
    assert faulted[0] == faulted[1] and faulted[2] == faulted[3] and faulted[0] != faulted[2]
