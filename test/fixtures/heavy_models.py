from typing import Tuple

import pytest
import numpy as np
import pandas as pd

from gempy_engine.core.data import InterpolationOptions, SurfacePoints, Orientations, SurfacePointsInternals, OrientationsInternals
from gempy_engine.core.data.grid import Grid, RegularGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor, TensorsStructure, StacksStructure, StackRelationType
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions


@pytest.fixture(scope="session")
def moureze_model() -> Tuple[InterpolationInput, InterpolationOptions, InputDataDescriptor]:
    
    # region: Pull data from cloud
    Moureze_points = pd.read_csv(
        'https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Moureze_Points.csv', sep=';',
        names=['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', '_'], header=0, )
    Sections_EW = pd.read_csv(
        'https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Sections_EW.csv',
        sep=';',
        names=['X', 'Y', 'Z', 'ID', '_'], header=1).dropna()
    Sections_NS = pd.read_csv(
        'https://raw.githubusercontent.com/Loop3D/ImplicitBenchmark/master/Moureze/Sections_NS.csv',
        sep=';',
        names=['X', 'Y', 'Z', 'ID', '_'], header=1).dropna()

    mask_surfpoints = Moureze_points['G_x'] < -9999
    
    sp = Moureze_points[mask_surfpoints]
    orientations_raw = Moureze_points[~mask_surfpoints]
    # endregion
    
    # region: Set up GemPy Data
    
    pick_every = 4
    surface_points: SurfacePoints = SurfacePoints(sp_coords=sp[['X', 'Y', 'Z']].values[::pick_every])

    orientations: Orientations = Orientations(
        dip_positions=orientations_raw[['X', 'Y', 'Z']].values,
        dip_gradients=orientations_raw[['G_x', 'G_y', 'G_z']].values
    )
    
    # Get extent from sp[['X', 'Y', 'Z']].values
    extent = np.array([
        sp['X'].min(), sp['X'].max(),
        sp['Y'].min(), sp['Y'].max(),
        sp['Z'].min(), sp['Z'].max()
    ])
    
    regular_grid = RegularGrid(
        extent=extent,
        regular_grid_shape=[2, 2, 2]
    )
    grid: Grid = Grid.from_regular_grid(regular_grid)

    interpolation_input: InterpolationInput = InterpolationInput(
        surface_points=surface_points,
        orientations=orientations,
        grid=grid
    )

    # endregion

    # region InterpolationOptions

    interpolation_options: InterpolationOptions = InterpolationOptions(
        range=10.,  
        c_o=1., 
        number_octree_levels=3,
        kernel_function=AvailableKernelFunctions.cubic,
        uni_degree=1
    )

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
