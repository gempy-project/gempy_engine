import os
import numpy as np
import pytest
from gempy_engine import compute_model
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.options import MeshExtractionMaskingOptions
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.plugins.plotting import helper_functions_pyvista
from tests.conftest import plot_pyvista

np.random.seed(42)


def test_dual_contouring_on_fault_model(one_fault_model, n_oct_levels=5):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model

    import numpy as np
    interpolation_input.surface_points.sp_coords[:, 2] += np.random.uniform(-0.02, 0.02, interpolation_input.surface_points.sp_coords[:, 2].shape)
    options.compute_scalar_gradient = False
    options.evaluation_options.dual_contouring = True
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
    options.evaluation_options.mesh_extraction_fancy = True

    options.evaluation_options.number_octree_levels = n_oct_levels
    options.evaluation_options.number_octree_levels_surface = n_oct_levels

    solutions: Solutions = compute_model(interpolation_input, options, structure)

    outputs: list[OctreeLevel] = solutions.octrees_output

    if plot_pyvista or False:
        helper_functions_pyvista.plot_pyvista(
            # octree_list=solutions.octrees_output,
            octree_list=None,
            dc_meshes=solutions.dc_meshes
        )


def test_dual_contouring_on_fault_model_anisotropic_octree(one_fault_model, n_oct_levels=3):
    import numpy as np
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model
    interpolation_input.grid.octree_grid = RegularGrid(
        orthogonal_extent=(np.array([-500, 500., -500, 500, -450, 550]) / 240),
        regular_grid_shape=[5, 2, 4])

    options.compute_scalar_gradient = False
    options.evaluation_options.dual_contouring = True
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
    options.evaluation_options.mesh_extraction_fancy = True

    options.evaluation_options.number_octree_levels = n_oct_levels
    options.evaluation_options.number_octree_levels_surface = n_oct_levels

    solutions: Solutions = compute_model(interpolation_input, options, structure)

    outputs: list[OctreeLevel] = solutions.octrees_output

    if plot_pyvista or False:
        helper_functions_pyvista.plot_pyvista(
            # octree_list=solutions.octrees_output,
            octree_list=None,
            dc_meshes=solutions.dc_meshes
        )


@pytest.mark.skip(reason="Run only explicit")
def test_dual_contouring_serialization(one_fault_model, n_oct_levels=3):
    import os
    import numpy as np
    from gempy_engine.core.backend_tensor import BackendTensor

    # 1. Set the flag to skip triangulation
    os.environ['GEMPY_SKIP_TRIANGULATION'] = '1'

    try:
        interpolation_input: InterpolationInput
        structure: InputDataDescriptor
        options: InterpolationOptions

        interpolation_input, structure, options = one_fault_model
        
        options.compute_scalar_gradient = False
        options.evaluation_options.dual_contouring = True
        options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
        options.evaluation_options.mesh_extraction_fancy = True

        options.evaluation_options.number_octree_levels = n_oct_levels
        options.evaluation_options.number_octree_levels_surface = n_oct_levels

        # 2. Compute the model
        solutions: Solutions = compute_model(interpolation_input, options, structure)

        # 3. Serialize data for C# triangulation
        for i, mesh in enumerate(solutions.dc_meshes):
            dc_data = mesh.dc_data
            if dc_data is None:
                continue
            
            # Prepare the data same way as _process_one_surface
            valid_voxels = dc_data.valid_voxels
            left_right_per_surface = dc_data.left_right_codes[valid_voxels]
            valid_voxels_per_surface = dc_data.valid_edges[valid_voxels]
            tree_depth_per_surface = dc_data.tree_depth
            
            # Reconstruct edges_normals (as in _process_one_surface)
            valid_edges = dc_data.valid_edges
            edges_normals = BackendTensor.t.zeros((valid_edges.shape[0], 12, 3), dtype=BackendTensor.dtype_obj)
            edges_normals[:] = 0
            edges_normals[valid_edges] = dc_data.gradients
            
            voxels_normals = edges_normals[valid_voxels]
            vertices = mesh.vertices
            base_number = np.array(dc_data.base_number)

            # Serialization
            output_dir = f"test_serialization_surface_{i}"
            os.makedirs(output_dir, exist_ok=True)
            
            np.save(f"{output_dir}/left_right_array.npy", left_right_per_surface)
            np.save(f"{output_dir}/valid_edges.npy", valid_voxels_per_surface)
            np.save(f"{output_dir}/voxel_normals.npy", BackendTensor.t.to_numpy(voxels_normals))
            np.save(f"{output_dir}/vertices.npy", vertices)
            np.save(f"{output_dir}/base_number.npy", base_number)
            with open(f"{output_dir}/tree_depth.txt", "w") as f:
                f.write(str(tree_depth_per_surface))

            print(f"Surface {i} data serialized to {output_dir}")

    finally:
        # Clean up
        os.environ['GEMPY_SKIP_TRIANGULATION'] = '0'


"""
C# Deserialization and Triangulation Logic:
------------------------------------------
The following C# code shows how to deserialize the saved numpy arrays without using NumSharp.
It uses a basic BinaryReader to read the .npy file format (assuming version 1.0 and standard headers).

using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Text;

public class DualContouringTriangulator {
    public void DeserializeAndTriangulate(string surfaceDir) {
        // 1. Load data
        int[] leftRightArray = LoadNpy<int>(Path.Combine(surfaceDir, "left_right_array.npy"));
        bool[] validEdges = LoadNpy<bool>(Path.Combine(surfaceDir, "valid_edges.npy"));
        float[] voxelNormals = LoadNpy<float>(Path.Combine(surfaceDir, "voxel_normals.npy"));
        float[] vertices = LoadNpy<float>(Path.Combine(surfaceDir, "vertices.npy"));
        int[] baseNumber = LoadNpy<int>(Path.Combine(surfaceDir, "base_number.npy"));
        int treeDepth = int.Parse(File.ReadAllText(Path.Combine(surfaceDir, "tree_depth.txt")));

        // Note: The 1D arrays need to be interpreted as MD arrays based on their shape.
        // leftRightArray: (n_voxels, 3)
        // validEdges: (n_voxels, 12)
        // voxelNormals: (n_voxels, 12, 3)
        // vertices: (n_vertices, 3)

        // 2. Triangulate (C# implementation of triangulate function)
        // var indices = Triangulate(leftRightArray, validEdges, treeDepth, voxelNormals, vertices, baseNumber);
    }

    /// <summary>
    /// Simple .npy loader for standard types. 
    /// Note: This is a simplified version and might need adjustments for specific byte orders or header variations.
    /// </summary>
    public static T[] LoadNpy<T>(string filePath) where T : struct {
        using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read))
        using (var br = new BinaryReader(fs)) {
            // Check Magic Number
            byte[] magic = br.ReadBytes(6);
            if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' || magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y')
                throw new Exception("Invalid .npy file");

            byte major = br.ReadByte();
            byte minor = br.ReadByte();
            ushort headerLen = br.ReadUInt16();
            byte[] headerBytes = br.ReadBytes(headerLen);
            string header = Encoding.ASCII.GetString(headerBytes);

            // Extract shape and dtype from header if needed. 
            // For simple use cases, we can just read the rest of the file as data.
            long dataLen = fs.Length - fs.Position;
            int elementSize = System.Runtime.InteropServices.Marshal.SizeOf(typeof(T));
            int count = (int)(dataLen / elementSize);
            
            T[] data = new T[count];
            byte[] buffer = br.ReadBytes((int)dataLen);
            Buffer.BlockCopy(buffer, 0, data, 0, (int)dataLen);
            return data;
        }
    }
}
"""
