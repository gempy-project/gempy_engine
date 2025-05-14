import os
from typing import Tuple

import pytest
import numpy as np
import pandas as pd

import tests.conftest
from gempy_engine.core.data import InterpolationOptions, SurfacePoints, Orientations, SurfacePointsInternals, OrientationsInternals, TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.kernel_classes.solvers import Solvers
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions

params = {
    "VeryFewInputOctLvl3"         : pytest.param((16, 3, Solvers.DEFAULT, 100), marks=pytest.mark.skipif(True, reason="Manually skip")),
    "VeryFewInputOctLvl3_SCIPY_GC": pytest.param((16, 3, Solvers.SCIPY_CG, 100), marks=pytest.mark.skipif(True, reason="Manually skip")),
    "FewInputOctLvl2"             : pytest.param((8, 2, Solvers.DEFAULT, 100), marks=pytest.mark.skipif(False, reason="Manually skip")),
    "FewInputOctLvl3"             : pytest.param((8, 3, Solvers.DEFAULT, 100), marks=pytest.mark.skipif(True, reason="Manually skip")),
    "FewInputOctLvl4"             : pytest.param((8, 4, Solvers.DEFAULT, 100), marks=pytest.mark.skipif(True, reason="Manually skip")),
    "FewInputOctLvl5"             : pytest.param((8, 5, Solvers.DEFAULT, 100), marks=pytest.mark.skipif(True, reason="Manually skip")),
    "FewInputOctLvl6"             : pytest.param((8, 5, Solvers.DEFAULT, 100), marks=pytest.mark.skipif(True, reason="Manually skip")),
    "MidInputOctLvl3"             : pytest.param((4, 3, Solvers.DEFAULT, 100), marks=pytest.mark.skipif(True, reason="Manually skip")),
    "AllInputOctLvl3"             : pytest.param((1, 3, Solvers.DEFAULT, 100), marks=pytest.mark.skipif(True, reason="Manually skip")),
    "AllInputOctLvl3_SCIPY_GC"    : pytest.param((1, 3, Solvers.SCIPY_CG, 100), marks=pytest.mark.skipif(True, reason="Manually skip")),
}


@pytest.fixture(scope="session", params=params.values(), ids=list(params.keys()))
def moureze_model(request, tests_root) -> Tuple[InterpolationInput, InterpolationOptions, InputDataDescriptor]:
    model = moureze_model_factory(
        path_to_root=tests_root,
        pick_every=request.param[0],
        octree_lvls=request.param[1],
        solver=request.param[2],
        nugget=request.param[3]
    )
    model[1].evaluation_options.mesh_extraction_fancy = True  # ! This is the Opt3
    return model


def moureze_model_factory(path_to_root: str, pick_every=8, octree_lvls=3, solver: Solvers = Solvers.DEFAULT, nugget=0.1) \
        -> Tuple[InterpolationInput, InterpolationOptions, InputDataDescriptor]:
    
    # region: Pull data from cloud
    Moureze_points = pd.read_csv(
        filepath_or_buffer=os.path.join(path_to_root, "data", "Moureze_Points.csv"),
        sep=';',
        names=['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', '_'],
        header=0
    )

    if MORE_DATA := False:
        Sections_EW = pd.read_csv(
            'https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Sections_EW.csv',
            sep=';',
            names=['X', 'Y', 'Z', 'ID', '_'],
            header=1
        ).dropna()
        Sections_NS = pd.read_csv(
            'https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Sections_NS.csv',
            sep=';',
            names=['X', 'Y', 'Z', 'ID', '_'],
            header=1
        ).dropna()

    mask_surfpoints = Moureze_points['G_x'] < -9999
    sp = Moureze_points[mask_surfpoints]
    orientations_raw = Moureze_points[~mask_surfpoints]
    # endregion
    # region: Set up GemPy Data
    # * LowInput pick_every=8 | MidInput pick_every=4 | HighInput pick_every=1
    surface_points: SurfacePoints = SurfacePoints(
        sp_coords=sp[['X', 'Y', 'Z']].values[::pick_every],
        nugget_effect_scalar=nugget
    )
    orientations: Orientations = Orientations(
        dip_positions=orientations_raw[['X', 'Y', 'Z']].values[::pick_every],
        dip_gradients=orientations_raw[['G_x', 'G_y', 'G_z']].values[::pick_every],
        nugget_effect_grad=nugget
    )
    # Get extent from sp[['X', 'Y', 'Z']].values
    extent = np.array([
        sp['X'].min(), sp['X'].max(),
        sp['Y'].min(), sp['Y'].max(),
        sp['Z'].min(), sp['Z'].max()
    ])
    regular_grid = RegularGrid(
        orthogonal_extent=extent,
        regular_grid_shape=[2, 2, 2]
    )
    grid: EngineGrid = EngineGrid.from_regular_grid(regular_grid)
    interpolation_input: InterpolationInput = InterpolationInput(
        surface_points=surface_points,
        orientations=orientations,
        grid=grid
    )
    # endregion

    # region InterpolationOptions
    interpolation_options: InterpolationOptions = InterpolationOptions.from_args(
        range=100.,
        c_o=10.,
        number_octree_levels=octree_lvls,
        kernel_function=AvailableKernelFunctions.cubic,
        uni_degree=0,
    )
    
    interpolation_options.number_octree_levels_surface = octree_lvls # * For now we set the same octree levels for dual contouring

    # TODO: Add solver parameter
    interpolation_options.kernel_options.kernel_solver = solver

    from gempy_engine.core.data.options import MeshExtractionMaskingOptions
    interpolation_options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.RAW

    # endregion
    # region InputDataDescriptor
    tensor_struct: TensorsStructure = TensorsStructure(
        number_of_points_per_surface=np.array([surface_points.n_points]),
    )
    stack_structure: StacksStructure = StacksStructure(
        number_of_points_per_stack=np.array([surface_points.n_points]),
        number_of_orientations_per_stack=np.array([orientations.n_items]),
        number_of_surfaces_per_stack=np.array([1]),
        masking_descriptor=[StackRelationType.ERODE]
    )
    input_data_descriptor: InputDataDescriptor = InputDataDescriptor(
        tensors_structure=tensor_struct,
        stack_structure=stack_structure
    )
    # endregion

    # endregion
    return interpolation_input, interpolation_options, input_data_descriptor
