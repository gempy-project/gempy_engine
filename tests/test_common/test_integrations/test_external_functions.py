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
from gempy_engine.core.data.stack_relation_type import StackRelationType
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
    #   Tent: peaked ridge at the center-top of the domain (open topography surface)
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


def test_tent_topography_compute_solutions(n_oct_levels=2):
    """Tent topography external function → compute_model → Solutions.

    Same setup as test_tent_topography_external_function but goes deeper
    into the Solutions object to extract and validate the computed scalar
    fields and lithology blocks — matching the pattern of
    test_dual_contouring_multiple_independent_fields.
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
    # region: Build tent mesh + model (same as test_tent_topography_external_function)
    # ------------------------------------------------------------------

    vertices = wp.array(np.array([
        [2.0, 0.0, 1.0],
        [8.0, 0.0, 1.0],
        [8.0, 2.0, 1.0],
        [2.0, 2.0, 1.0],
        [5.0, 1.0, 4.5],
    ], dtype=np.float32), dtype=wp.vec3)
    indices_flat = np.array([0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4], dtype=np.int32)
    indices_wp = wp.array(indices_flat, dtype=wp.int32)
    mesh = wp.Mesh(points=vertices, indices=indices_wp)

    import os
    import pandas as pd
    from gempy_engine.core.backend_tensor import BackendTensor
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
    from gempy_engine.core.data.options import InterpolationOptions, MeshExtractionMaskingOptions
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

    def implicit_tent(xyz: np.ndarray) -> np.ndarray:
        n = len(xyz)
        query_wp = wp.array(xyz.astype(np.float32), dtype=wp.vec3)
        out_d = wp.zeros(n, dtype=wp.float32)
        wp.launch(_sdf_with_grad, dim=n,
                  inputs=[mesh.id, vertices, indices_wp, query_wp,
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
                  inputs=[mesh.id, vertices, indices_wp, query_wp,
                          out_d, out_gx, out_gy, out_gz])
        wp.synchronize()
        return out_gx.numpy().astype(np.float64), out_gy.numpy().astype(np.float64), out_gz.numpy().astype(np.float64)

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
    # region: compute_model → Solutions
    # ------------------------------------------------------------------

    solutions: Solutions = compute_model(
        interpolation_input, options, input_data_descriptor)

    # endregion

    # ------------------------------------------------------------------
    # region: Extract scalar fields & lithology blocks from Solutions
    # ------------------------------------------------------------------

    last_level = solutions.octrees_output[-1]
    # 
    stack_0_output = last_level.outputs[0]
    scalar_0 = BackendTensor.t.to_numpy(stack_0_output.exported_fields.scalar_field)

    litho_ids = solutions.raw_arrays.lith_block

    unique_litho = np.unique(litho_ids)
    assert 1 in unique_litho, f"Expected litho 1 in final block, got {unique_litho[:10]}"

    assert np.any(scalar_0 > 0), "SDF should have positive values"
    assert np.any(scalar_0 < 0), "SDF should have negative values"

    # endregion

    # ------------------------------------------------------------------
    # region: Mesh assertions (same as test_tent_topography_external_function)
    # ------------------------------------------------------------------

    assert solutions.dc_meshes is not None
    assert len(solutions.dc_meshes) > 0
    mesh_out = solutions.dc_meshes[0]
    assert mesh_out.vertices is not None and len(mesh_out.vertices) > 0
    assert mesh_out.edges is not None and len(mesh_out.edges) > 0

    # endregion

    # ------------------------------------------------------------------
    # region: Plot (same pattern as test_dual_contouring_multiple_independent_fields)
    # ------------------------------------------------------------------

    if PLOT or True:
        tent_verts = np.array([[2.0, 0.0, 1.0], [8.0, 0.0, 1.0], [8.0, 2.0, 1.0], [2.0, 2.0, 1.0], [5.0, 1.0, 4.5]])
        tent_faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int32)
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes[1:],
            vertices=tent_verts,
            indices=tent_faces,
            clip=(0.5, 4.5),
        )


def test_tent_topography_scalar_field_3d():
    """3D visualization: tent topography mesh + below-surface lithology region.

    Computes the Warp SDF on a high-resolution grid, segments it with GemPy's
    activate_formation_block into above-surface (lith 0) and below-surface
    (lith 1), then thresholds the 3D plot to hide everything above the tent.
    Shows:
      - Below-surface cells colored by SDF value (RdBu_r colormap)
      - Zero-isosurface (light blue — the tent surface itself)
      - Tent mesh edges (green wireframe)
    """
    pytest.importorskip("warp")
    import warp as wp

    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA device required")

    pytest.importorskip("pyvista")
    import pyvista as pv

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
    # region: Build tent mesh in model coordinates
    # ------------------------------------------------------------------

    vertices = wp.array(np.array([
        [2.0, 0.0, 1.0],
        [8.0, 0.0, 1.0],
        [8.0, 2.0, 1.0],
        [2.0, 2.0, 1.0],
        [5.0, 1.0, 4.5],
    ], dtype=np.float32), dtype=wp.vec3)
    indices_flat = np.array(
        [0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4], dtype=np.int32)
    indices_wp = wp.array(indices_flat, dtype=wp.int32)
    mesh = wp.Mesh(points=vertices, indices=indices_wp)

    # endregion

    # ------------------------------------------------------------------
    # region: Build high-resolution regular grid and evaluate SDF
    # ------------------------------------------------------------------

    from gempy_engine.core.data.regular_grid import RegularGrid

    extent = [0, 10.0, 0, 2.0, 0, 5.0]
    res_xy = 60
    res_z = 60
    resolution = [res_xy, max(10, res_xy // 6), res_z]

    grid = RegularGrid(extent, resolution)
    xyz_flat = grid.values_vtk_format  # (nx+1)*(ny+1)*(nz+1), 3 in VTK order
    nx, ny, nz = resolution[0] + 1, resolution[1] + 1, resolution[2] + 1
    n_pts = nx * ny * nz

    xyz_vtk = xyz_flat.astype(np.float32)
    query_wp = wp.array(xyz_vtk, dtype=wp.vec3)
    out_d = wp.zeros(n_pts, dtype=wp.float32)
    wp.launch(_sdf_with_grad, dim=n_pts,
              inputs=[mesh.id, vertices, indices_wp, query_wp,
                      out_d, wp.zeros(n_pts, dtype=wp.float32),
                      wp.zeros(n_pts, dtype=wp.float32),
                      wp.zeros(n_pts, dtype=wp.float32)])
    wp.synchronize()
    scalar_flat = out_d.numpy()
    # VTK ordering: Z-fastest, then Y, then X → reshape directly as (nx, ny, nz)
    scalar = scalar_flat.reshape(nx, ny, nz)

    assert np.all(np.isfinite(scalar_flat)), "SDF contains NaN/Inf"
    assert np.any(scalar > 0), "Expected some points on the 'outside' side of the topography"
    assert np.any(scalar < 0), "Expected some points below/behind the topography surface"

    # The zero-isosurface should intersect the grid centrally — verify the field
    # has both positive and negative values near the tent apex
    apex_ix = int((5.0 - extent[0]) / (extent[1] - extent[0]) * (nx - 1))
    apex_iy = int((1.0 - extent[2]) / (extent[3] - extent[2]) * (ny - 1))
    apex_iz = int((4.5 - extent[4]) / (extent[5] - extent[4]) * (nz - 1))
    assert 0 <= apex_ix < nx and 0 <= apex_iy < ny and 0 <= apex_iz < nz
    # Points just above and below the apex should have opposite signs
    assert scalar[apex_ix, apex_iy, min(apex_iz + 1, nz - 1)] * scalar[apex_ix, apex_iy, max(apex_iz - 1, 0)] < 0, (
        f"Expected sign change across the tent surface near apex, got: "
        f"above={scalar[apex_ix, apex_iy, min(apex_iz + 1, nz - 1)]:.3f}, "
        f"below={scalar[apex_ix, apex_iy, max(apex_iz - 1, 0)]:.3f}")

    # endregion

    # ------------------------------------------------------------------
    # region: Segment SDF into above/below-surface lithology IDs
    # ------------------------------------------------------------------

    from gempy_engine.core.backend_tensor import BackendTensor
    from gempy_engine.modules.activator.activator_interface import activate_formation_block

    exported_fields = ExportedFields(
        _scalar_field=BackendTensor.t.array(scalar_flat.astype(np.float64)),
        _scalar_field_at_surface_points=BackendTensor.t.array([0.0]),
    )
    lith_block = activate_formation_block(exported_fields, np.array([0, 1]), 50000)
    lith_flat = lith_block.ravel()  # (N,) soft values ~0 or ~1
    lith_int = np.round(lith_flat).astype(np.uint8)

    assert np.any(lith_int == 0), "Expected some points above the surface (lith 0)"
    assert np.any(lith_int == 1), "Expected some points below the surface (lith 1)"
    assert not np.any(lith_int > 1), f"Unexpected lith values: {np.unique(lith_int)[:10]}"

    # endregion

    # ------------------------------------------------------------------
    # region: 3D plot — only below-surface region + tent mesh
    # ------------------------------------------------------------------

    pv.global_theme.show_edges = True

    plotter = pv.Plotter(off_screen=not PLOT)

    x_arr = xyz_flat[:, 0].reshape(nx, ny, nz)
    y_arr = xyz_flat[:, 1].reshape(nx, ny, nz)
    z_arr = xyz_flat[:, 2].reshape(nx, ny, nz)
    grid_3d = pv.StructuredGrid(x_arr, y_arr, z_arr)
    grid_3d.point_data["sdf"] = scalar.ravel(order="F")

    # Compute per-cell lithology: cell is "below surface" if majority of its 8
    # corner points are lith 1. StructuredGrid has (nx-1, ny-1, nz-1) cells.
    lith_3d = lith_int.reshape(nx, ny, nz)
    cx, cy, cz = nx - 1, ny - 1, nz - 1
    cell_lith = np.empty((cx, cy, cz), dtype=np.uint8)
    for i in range(cx):
        for j in range(cy):
            for k in range(cz):
                corners = lith_3d[i:i+2, j:j+2, k:k+2]
                cell_lith[i, j, k] = 1 if corners.sum() >= 4 else 0

    grid_3d.cell_data["lith"] = cell_lith.ravel(order="F")

    # Keep only cells fully below the surface
    below = grid_3d.threshold(value=[0.5, 1.5], scalars="lith")
    plotter.add_mesh(below, scalars="sdf", cmap="RdBu_r",
                     clim=[-3, 3], opacity=0.7,
                     label="Below-surface region")

    iso = grid_3d.contour(isosurfaces=[0.0], scalars="sdf")
    plotter.add_mesh(iso, color="lightblue", opacity=0.7,
                     label="SDF zero-isosurface")

    verts_np = np.array([
        [2.0, 0.0, 1.0],
        [8.0, 0.0, 1.0],
        [8.0, 2.0, 1.0],
        [2.0, 2.0, 1.0],
        [5.0, 1.0, 4.5],
    ], dtype=np.float32)
    faces_np = indices_flat.reshape(-1, 3)
    faces_pv = np.insert(faces_np, 0, 3, axis=1).ravel()
    tent_poly = pv.PolyData(verts_np, faces_pv)
    plotter.add_mesh(tent_poly, color="green", opacity=0.5,
                     label="Tent mesh", show_edges=True, line_width=2)

    plotter.add_axes()
    plotter.add_legend()

    out_path = "tent_topography_below_surface_3d.png"
    plotter.show(title="Tent topography — below-surface region",
                  screenshot=out_path)

    # endregion


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_null_space_external_function_no_gradients(n_oct_levels=2):
    """Null-space external function (sphere SDF, no gradients) with
    zero surface points and zero orientations. Validates that:
    - compute_model succeeds without gradients
    - null-space stack is excluded from dual contouring meshes
    - final block has -1 (null_space_id) in masked regions
    """

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

    extent = [0, 10.0, 0, 2.0, 0, 5.0]
    resolution = [15, 2, 15]
    regular_grid = RegularGrid(extent, resolution)
    grid = EngineGrid(octree_grid=regular_grid)

    def sphere_sdf(xyz: np.ndarray) -> np.ndarray:
        center = (extent[1] - extent[0]) / 2
        radius = center / 2
        return -((xyz[:, 0] - center) ** 2 + (xyz[:, 1] - 1.0) ** 2 + (xyz[:, 2] - 2.5) ** 2 - radius ** 2)

    null_space_func = CustomInterpolationFunctions(
        scalar_field_at_surface_points=np.array([0.0]),
        implicit_function=sphere_sdf,
    )

    n_sp_geo = 4
    n_ori_geo = 2

    sp_coords = np.array([[2.0, 0.5, 2.5], [8.0, 0.5, 2.5], [2.0, 0.5, 1.0], [8.0, 0.5, 1.0]])
    dip_positions = np.array([[5.0, 1.0, 2.5], [5.0, 1.0, 1.0]])
    dip_gradients = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([0, n_sp_geo]),
        number_of_orientations_per_stack=np.array([0, n_ori_geo]),
        number_of_surfaces_per_stack=np.array([0, 2]),
        masking_descriptor=[
            StackRelationType.NULL_SPACE,
            StackRelationType.BASEMENT,
        ],
        interp_functions_per_stack=[null_space_func, None],
        null_space_id=-1,
    )

    tensor_struct = TensorsStructure(
        number_of_points_per_surface=np.array([2, 2]))
    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)

    range_ = 0.8660254 * 100
    c_o = 35.71428571 * 100
    options = InterpolationOptions.from_args(
        range_, c_o, uni_degree=0, i_res=4, gi_res=2,
        number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.evaluation_options.mesh_extraction_masking_options = \
        MeshExtractionMaskingOptions.INTERSECT

    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_positions, dip_gradients)
    ids = np.array([0, 1, 2])

    interpolation_input = InterpolationInput(spi, ori, grid, ids)

    solutions: Solutions = compute_model(
        interpolation_input, options, input_data_descriptor)

    assert solutions is not None
    assert solutions.dc_meshes is not None

    n_non_null_stacks = 1
    assert len(solutions.dc_meshes) == n_non_null_stacks * 2

    final_block = solutions.octrees_output[-1].last_output_center.final_block
    final_block_np = np.array(final_block).reshape(-1)
    has_minus_one = (final_block_np == -1).any()
    assert has_minus_one, "Expected some cells to be -1 (null-space masked)"

    has_non_negative = (final_block_np >= 0).any()
    assert has_non_negative, "Expected some cells to have non-negative IDs"


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_external_function_stack_with_dummy_element():
    """External function stack with one empty structural element (no SPs, no orientations).

    Simulates the server mesh-field wiring where a NULL_SPACE group is added with
    a single dummy StructuralElement that has zero surface points and zero orientations.
    Validates that:
    - compute_model succeeds without requiring fake SP/orientation rows
    - the surface-count disparity between StacksStructure and TensorsStructure
      (due to TensorsStructure filtering zero-point surfaces) is handled correctly
    """
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
    from gempy_engine.API.model.model_api import _check_input_validity

    extent = [0, 10.0, 0, 2.0, 0, 5.0]
    resolution = [15, 2, 15]
    regular_grid = RegularGrid(extent, resolution)
    grid = EngineGrid(octree_grid=regular_grid)

    def sphere_sdf(xyz: np.ndarray) -> np.ndarray:
        center = (extent[1] - extent[0]) / 2
        radius = center / 2
        return -((xyz[:, 0] - center) ** 2 + (xyz[:, 1] - 1.0) ** 2 + (xyz[:, 2] - 2.5) ** 2 - radius ** 2)

    null_space_func = CustomInterpolationFunctions(
        scalar_field_at_surface_points=np.array([0.0]),
        implicit_function=sphere_sdf,
    )

    n_sp_geo = 4
    n_ori_geo = 2

    sp_coords = np.array([[2.0, 0.5, 2.5], [8.0, 0.5, 2.5], [2.0, 0.5, 1.0], [8.0, 0.5, 1.0]])
    dip_positions = np.array([[5.0, 1.0, 2.5], [5.0, 1.0, 1.0]])
    dip_gradients = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([0, n_sp_geo]),
        number_of_orientations_per_stack=np.array([0, n_ori_geo]),
        number_of_surfaces_per_stack=np.array([0, 2]),
        masking_descriptor=[
            StackRelationType.NULL_SPACE,
            StackRelationType.BASEMENT,
        ],
        interp_functions_per_stack=[null_space_func, None],
        null_space_id=-1,
    )

    tensor_struct = TensorsStructure(
        number_of_points_per_surface=np.array([2, 2]))
    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)

    range_ = 0.8660254 * 100
    c_o = 35.71428571 * 100
    options = InterpolationOptions.from_args(
        range_, c_o, uni_degree=0, i_res=4, gi_res=2,
        number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)
    options.number_octree_levels = 2
    options.debug = True

    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_positions, dip_gradients)
    ids = np.array([0, 1, 2])

    interpolation_input = InterpolationInput(spi, ori, grid, ids)

    solutions: Solutions = compute_model(
        interpolation_input, options, input_data_descriptor)

    assert solutions is not None
    assert solutions.dc_meshes is not None

    final_block = solutions.octrees_output[-1].last_output_center.final_block
    final_block_np = np.array(final_block).reshape(-1)
    has_minus_one = (final_block_np == -1).any()
    assert has_minus_one, "Expected some cells to be -1 (null-space masked)"


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_external_function_stack_with_dummy_element_validation_only():
    """Validation-only: external stack with an empty dummy element passes input checks.

    Simulates what InputDataDescriptor.from_structural_frame produces after
    zeroing out the surface count for an external-function stack with no points:
    number_of_surfaces_per_stack becomes [0, 2] instead of [1, 2].
    """
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
    from gempy_engine.API.model.model_api import _check_input_validity, _stack_uses_external_function

    extent = [0, 10.0, 0, 2.0, 0, 5.0]
    resolution = [15, 2, 15]
    regular_grid = RegularGrid(extent, resolution)
    grid = EngineGrid(octree_grid=regular_grid)

    null_space_func = CustomInterpolationFunctions(
        scalar_field_at_surface_points=np.array([0.0]),
        implicit_function=lambda xyz: np.zeros(len(xyz)),
    )

    n_sp_geo = 4
    n_ori_geo = 2
    sp_coords = np.array([[2.0, 0.5, 2.5], [8.0, 0.5, 2.5], [2.0, 0.5, 1.0], [8.0, 0.5, 1.0]])
    dip_positions = np.array([[5.0, 1.0, 2.5], [5.0, 1.0, 1.0]])
    dip_gradients = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    # External stack has 1 dummy surface, but from_structural_frame zeroes it to 0
    # because it's an external-function stack with zero points
    stack_structure = StacksStructure(
        number_of_points_per_stack=np.array([0, n_sp_geo]),
        number_of_orientations_per_stack=np.array([0, n_ori_geo]),
        number_of_surfaces_per_stack=np.array([0, 2]),
        masking_descriptor=[StackRelationType.NULL_SPACE, StackRelationType.BASEMENT],
        interp_functions_per_stack=[null_space_func, None],
        null_space_id=-1,
    )
    tensor_struct = TensorsStructure(
        number_of_points_per_surface=np.array([2, 2]))
    input_data_descriptor = InputDataDescriptor(tensor_struct, stack_structure)

    options = InterpolationOptions.from_args(
        86.6, 3571.4, uni_degree=0, i_res=4, gi_res=2,
        number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)

    spi = SurfacePoints(sp_coords)
    ori = Orientations(dip_positions, dip_gradients)
    ids = np.array([0, 1, 2])
    interpolation_input = InterpolationInput(spi, ori, grid, ids)

    # This should not raise
    _check_input_validity(interpolation_input, options, input_data_descriptor)

    # Verify the helper works correctly
    assert _stack_uses_external_function(stack_structure, 0) is True
    assert _stack_uses_external_function(stack_structure, 1) is False

    # Remove the external function → should now raise
    stack_structure.interp_functions_per_stack[0] = None
    with pytest.raises(Exception):
        _check_input_validity(interpolation_input, options, input_data_descriptor)
