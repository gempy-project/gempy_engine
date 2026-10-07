"""Engine inputs for https://docs.gempy.org/examples/geometries/g07_combination.html.

The existing model7 CSVs in simple_geometries are the upstream jan_models data.
Only numerical resolution is reduced from the gallery example; all observations,
the fault and the unconformity are retained.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gempy_engine.core.data import InterpolationOptions, Orientations, SurfacePoints, TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.generic_grid import GenericGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.core.data.transforms import Transform
from gempy_engine.plugins.plotting.helper_functions import calculate_gradient


MODEL7_EXTENT = (0, 2500, 0, 1000, 0, 1000)
MODEL7_SURFACES = ("fault", "rock3", "rock2", "rock1")
MODEL7_UNITS = (*MODEL7_SURFACES, "basement")


def model7_combination_factory(
    *, number_octree_levels: int = 3, mesh_extraction: bool = False,
    custom_grid: np.ndarray | None = None,
) -> tuple[InterpolationInput, InterpolationOptions, InputDataDescriptor]:
    """Return fresh engine input, options and descriptor for actual Model 7.

    Stacks are Fault_Series(fault), Strat_Series1(rock3), and
    Strat_Series2(rock2, rock1). Unit IDs 1..5 follow MODEL7_UNITS; ID 1
    describes the fault, not a lithology. The youngest stratigraphy erodes
    the folded older series, and the fault affects both following stacks.

    custom_grid is an optional (N, 3) array in original model coordinates.
    Engine coordinates are (world_xyz - [1150, 500, 525]) / 4600, using
    Transform.from_input_points rather than normalizing to the grid extent.
    Gradients retain unit magnitude. Scalar fields are implicit potential
    values in this normalized interpolation system, not elevations or IDs.
    Sampling results retain custom_grid row order in the custom grid slice.

    The initial octree has 2^3 cells; three levels reach at most 8^3 cells.
    Mesh extraction is opt-in so the default only computes model fields.
    Import model7_combination explicitly into a consuming test module to
    register the fixture without changing the shared conftest.
    """
    data_path = Path(__file__).parent / "simple_geometries"
    points = pd.read_csv(data_path / "model7_surface_points.csv")
    orientations = pd.read_csv(data_path / "model7_orientations.csv")
    # Both arrays must be contiguous by surface/stack, not CSV appearance order.
    points = pd.concat([points.loc[points.formation == name] for name in MODEL7_SURFACES])
    orientations = pd.concat([
        orientations.loc[orientations.formation == name] for name in MODEL7_SURFACES
    ])
    xyz = points[["X", "Y", "Z"]].to_numpy(dtype=np.float64)
    dip_positions = orientations[["X", "Y", "Z"]].to_numpy(dtype=np.float64)
    transform = Transform.from_input_points(
        SimpleNamespace(xyz=xyz), SimpleNamespace(xyz=dip_positions)
    )
    gradients = np.vstack(calculate_gradient(
        orientations["dip"], orientations["azimuth"], orientations["polarity"]
    )).T
    bounds = np.array(MODEL7_EXTENT, dtype=np.float64).reshape(3, 2).T
    extent = transform.apply(bounds).T.ravel()
    grid = EngineGrid(
        octree_grid=RegularGrid(extent, [2, 2, 2]),
        custom_grid=None if custom_grid is None else GenericGrid(
            values=transform.apply(np.asarray(custom_grid, dtype=np.float64))
        ),
    )
    interpolation_input = InterpolationInput(
        surface_points=SurfacePoints(transform.apply(xyz)),
        orientations=Orientations(
            transform.apply(dip_positions), transform.transform_gradient(gradients)
        ),
        grid=grid,
        unit_values=np.arange(1, 6),
    )
    descriptor = InputDataDescriptor(
        TensorsStructure(number_of_points_per_surface=np.array([3, 15, 39, 45])),
        StacksStructure(
            number_of_points_per_stack=np.array([3, 15, 84]),
            number_of_orientations_per_stack=np.array([1, 1, 6]),
            number_of_surfaces_per_stack=np.array([1, 1, 2]),
            masking_descriptor=[
                StackRelationType.FAULT, StackRelationType.ERODE, StackRelationType.BASEMENT
            ],
            faults_relations=np.array([
                [False, True, True], [False, False, False], [False, False, False]
            ]),
        ),
    )
    options = InterpolationOptions.init_octree_options(refinement=number_octree_levels)
    options.evaluation_options.mesh_extraction = mesh_extraction
    options.evaluation_options.number_octree_levels_surface = max(2, number_octree_levels)
    return interpolation_input, options, descriptor


@pytest.fixture
def model7_combination() -> tuple[InterpolationInput, InterpolationOptions, InputDataDescriptor]:
    return model7_combination_factory()
