from typing import List

import numpy as np
import pytest

from gempy_engine.API.interp_single._multi_scalar_field_manager import _interpolate_stack
from gempy_engine.API.interp_single.interp_features import interpolate_all_fields_no_octree
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.core.data.scalar_field_output import ScalarFieldOutput
from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.core.data.options import MeshExtractionMaskingOptions
from gempy_engine.modules.activator.activator_interface import activate_formation_block
from gempy_engine.plugins.plotting import helper_functions_pyvista
from tests.conftest import TEST_SPEED
from gempy_engine.plugins.plotting.helper_functions import plot_block

PLOT = False


def test_compute_mask_components_all_erode_implicit_sphere(unconformity_complex_implicit):
    """Plot each individual mask compontent"""
    # TODO:
    interpolation_input, options, structure = unconformity_complex_implicit
    outputs: List[ScalarFieldOutput] = _interpolate_stack(structure, interpolation_input, options)

    if PLOT or False:
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].mask_components_erode, grid)
        plot_block(outputs[1].mask_components_erode, grid)
        plot_block(outputs[2].mask_components_erode, grid)


def test_final_block_implicit(unconformity_complex_implicit):
    interpolation_input, options, structure = unconformity_complex_implicit
    outputs: List[InterpOutput] = interpolate_all_fields_no_octree(interpolation_input, options, structure)

    if PLOT or False:
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].final_block, grid)
        plot_block(outputs[1].final_block, grid)
        plot_block(outputs[2].final_block, grid)
        plot_block(outputs[3].final_block, grid)


def test_implicit_function(unconformity_complex):
    def implicit_sphere(xyz: np.ndarray, extent: np.ndarray):
        x_dir = np.minimum(xyz[:, 0] - extent[0], extent[1] - xyz[:, 0])
        y_dir = np.minimum(xyz[:, 1] - extent[2], extent[3] - xyz[:, 1])
        z_dir = np.minimum(xyz[:, 2] - extent[4], extent[5] - xyz[:, 2])
        return x_dir ** 2 + y_dir ** 2 + z_dir ** 2

    interpolation_input, options, structure = unconformity_complex
    grid = interpolation_input.grid.octree_grid 
    xyz = grid.values
    scalar = implicit_sphere(xyz, grid.orthogonal_extent)

    from gempy_engine.core.backend_tensor import BackendTensor
    exported_fields = ExportedFields(
        _scalar_field=BackendTensor.t.array(scalar),
        _scalar_field_at_surface_points=BackendTensor.t.array([20])
    )
    values_block = activate_formation_block(exported_fields, np.array([0, 1]), 100000)

    if PLOT or False:
        plot_block(scalar, grid)
        plot_block(values_block, grid)


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_dual_contouring_multiple_independent_fields(unconformity_complex_implicit, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex_implicit
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT

    solutions: Solutions = compute_model(interpolation_input, options, structure)

    if PLOT or False:
        dc_data = solutions.dc_meshes[0].dc_data  # * Scalar field where to show gradients
        intersection_xyz = dc_data.xyz_on_edge
        gradients = dc_data.gradients

        center_mass = dc_data.bias_center_mass
        normals = dc_data.bias_normals

        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes,
            gradient_pos=intersection_xyz, gradients=gradients,  # * Uncomment for more detailed plots
            a=center_mass, b=normals
        )


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_tent_topography_external_function(n_oct_levels=2):
    """Use Warp's mesh_query_point_sign_normal on a tent topography mesh
    as an external implicit function, replacing the built-in SPHERE function.
    Runs compute_model with dual contouring mesh extraction.

    The gradient is computed analytically in the same Warp kernel pass:
    grad = (query_point - closest_surface_point) / distance.

    This validates that arbitrary GPU-based external geometry can drive
    the full GemPy modeling pipeline end-to-end.
    """
    pytest.importorskip("warp")
    import warp as wp

    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA device required")

    # ------------------------------------------------------------------
    # region: Warp kernel — SDF + gradient in a single launch
    # ------------------------------------------------------------------

    @wp.kernel
    def _sdf_with_grad(
        mesh_id: wp.uint64,
        mesh_vertices: wp.array(dtype=wp.vec3),
        mesh_indices: wp.array(dtype=wp.int32),
        query_points: wp.array(dtype=wp.vec3),
        out_distances: wp.array(dtype=wp.float32),
        out_gx: wp.array(dtype=wp.float32),
        out_gy: wp.array(dtype=wp.float32),
        out_gz: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        p = query_points[tid]
        hit = wp.mesh_query_point_sign_normal(mesh_id, p, 1.0e6, 1.0e-6)

        i0 = mesh_indices[hit.face * 3]
        i1 = mesh_indices[hit.face * 3 + 1]
        i2 = mesh_indices[hit.face * 3 + 2]
        v0 = mesh_vertices[i0]
        v1 = mesh_vertices[i1]
        v2 = mesh_vertices[i2]

        w = 1.0 - hit.u - hit.v
        closest = v0 * hit.u + v1 * hit.v + v2 * w

        diff = p - closest
        dist = wp.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
        inv_dist = wp.where(dist < 1.0e-6, 0.0, 1.0 / dist)

        out_distances[tid] = wp.where(hit.sign < 0.0, -dist, dist)
        out_gx[tid] = diff[0] * inv_dist
        out_gy[tid] = diff[1] * inv_dist
        out_gz[tid] = diff[2] * inv_dist

    # endregion

    # ------------------------------------------------------------------
    # region: Build tent topography mesh in model coordinates
    #   Model extent: X=[0,10], Y=[0,2], Z=[0,5]
    #   Tent: peaked ridge at the center-top of the domain
    # ------------------------------------------------------------------

    vertices = wp.array(np.array([
        [2.0, 0.0, 1.0],
        [8.0, 0.0, 1.0],
        [8.0, 2.0, 1.0],
        [2.0, 2.0, 1.0],
        [5.0, 1.0, 4.5],
    ], dtype=np.float32), dtype=wp.vec3)
    indices = wp.array(np.array(
        [0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4], dtype=np.int32
    ), dtype=wp.int32)
    mesh = wp.Mesh(points=vertices, indices=indices)

    # ------------------------------------------------------------------
    # region: Build model — same setup as unconformity_complex_implicit
    #         but with the Warp tent SDF as the external function
    # ------------------------------------------------------------------

    import os
    import pandas as pd

    from gempy_engine.core.data.engine_grid import EngineGrid
    from gempy_engine.core.data.regular_grid import RegularGrid
    from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
    from gempy_engine.core.data.stacks_structure import StacksStructure
    from gempy_engine.core.data import TensorsStructure
    from gempy_engine.core.data.stack_relation_type import StackRelationType
    from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
    from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
    from gempy_engine.core.data.kernel_classes.orientations import Orientations
    from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
    from gempy_engine.core.data.interpolation_input import InterpolationInput
    from gempy_engine.core.data.options import InterpolationOptions
    from gempy_engine.core.data.options import EvaluationOptions
    from gempy_engine.plugins.plotting.helper_functions import calculate_gradient

    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "fixtures", "simple_geometries"
    )

    orientations_df = pd.read_csv(
        os.path.join(data_path, "05_toy_fold_unconformity_orientations.csv"))
    sp_df = pd.read_csv(
        os.path.join(data_path, "05_toy_fold_unconformity_interfaces.csv"))

    sp_coords = sp_df[["X", "Y", "Z"]].values
    dip_positions = orientations_df[["X", "Y", "Z"]].values
    dip_gradients_ = calculate_gradient(
        orientations_df["dip"], orientations_df["azimuth"],
        orientations_df["polarity"])
    dip_gradients = np.vstack(dip_gradients_).T

    extent = [0, 10.0, 0, 2.0, 0, 5.0]
    resolution = [15, 2, 15]
    regular_grid = RegularGrid(extent, resolution)
    grid = EngineGrid(octree_grid=regular_grid)

    # --- SDF + gradient evaluator backed by the Warp kernel ---
    def implicit_tent(xyz: np.ndarray) -> np.ndarray:
        n = len(xyz)
        query_wp = wp.array(xyz.astype(np.float32), dtype=wp.vec3)
        out_d = wp.zeros(n, dtype=wp.float32)
        wp.launch(_sdf_with_grad, dim=n,
                  inputs=[mesh.id, vertices, indices, query_wp,
                          out_d, wp.zeros(n, dtype=wp.float32),
                          wp.zeros(n, dtype=wp.float32), wp.zeros(n, dtype=wp.float32)])
        wp.synchronize()
        return out_d.numpy().astype(np.float64)

    def tent_grad(xyz: np.ndarray) -> tuple:
        n = len(xyz)
        query_wp = wp.array(xyz.astype(np.float32), dtype=wp.vec3)
        out_d = wp.zeros(n, dtype=wp.float32)
        out_gx = wp.zeros(n, dtype=wp.float32)
        out_gy = wp.zeros(n, dtype=wp.float32)
        out_gz = wp.zeros(n, dtype=wp.float32)
        wp.launch(_sdf_with_grad, dim=n,
                  inputs=[mesh.id, vertices, indices, query_wp,
                          out_d, out_gx, out_gy, out_gz])
        wp.synchronize()
        return out_gx.numpy().astype(np.float64), out_gy.numpy().astype(np.float64), out_gz.numpy().astype(np.float64)

    _grad_cache = None

    def gx_func(xyz: np.ndarray) -> np.ndarray:
        return tent_grad(xyz)[0]

    def gy_func(xyz: np.ndarray) -> np.ndarray:
        return tent_grad(xyz)[1]

    def gz_func(xyz: np.ndarray) -> np.ndarray:
        return tent_grad(xyz)[2]

    custom_func = CustomInterpolationFunctions(
        scalar_field_at_surface_points=np.array([0.0]),
        implicit_function=implicit_tent,
        gx_function=gx_func,
        gy_function=gy_func,
        gz_function=gz_func,
    )

    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([3, 2, 6]),
        number_of_orientations_per_stack=np.array([2, 1, 6]),
        number_of_surfaces_per_stack=np.array([1, 1, 2]),
        masking_descriptor=[
            StackRelationType.ERODE,
            StackRelationType.ERODE,
            StackRelationType.BASEMENT,
        ],
        interp_functions_per_stack=[custom_func, None, None, None],
    )

    tensor_struct = TensorsStructure(
        number_of_points_per_surface=np.array([3, 2, 3, 3]))
    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)

    range_ = 0.8660254 * 100
    c_o = 35.71428571 * 100
    i_r = 4
    gi_r = 2

    options = InterpolationOptions.from_args(
        range_, c_o, uni_degree=0, i_res=i_r, gi_res=gi_r,
        number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.evaluation_options.mesh_extraction_masking_options = (
        MeshExtractionMaskingOptions.INTERSECT)

    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_positions, dip_gradients)
    ids = np.array([0, 1, 2, 3, 4, 5, 6])

    interpolation_input = InterpolationInput(spi, ori, grid, ids)

    # endregion

    # ------------------------------------------------------------------
    # region: Run compute_model
    # ------------------------------------------------------------------

    solutions: Solutions = compute_model(
        interpolation_input, options, input_data_descriptor)

    # ------------------------------------------------------------------
    # region: Assertions
    # ------------------------------------------------------------------

    assert solutions is not None
    assert solutions.dc_meshes is not None
    assert len(solutions.dc_meshes) > 0
    mesh = solutions.dc_meshes[0]
    assert mesh.vertices is not None
    assert mesh.edges is not None
    assert len(mesh.vertices) > 0
    assert len(mesh.edges) > 0

    # endregion

    # ------------------------------------------------------------------
    # region: Plot (guarded)
    # ------------------------------------------------------------------

    if PLOT or True:
        dc_data = solutions.dc_meshes[0].dc_data
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes,
        )
