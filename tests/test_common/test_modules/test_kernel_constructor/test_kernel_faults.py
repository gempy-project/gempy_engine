import matplotlib.pyplot as plt
import numpy as np

from gempy_engine.core.data.custom_segmentation_functions import _implicit_3d_ellipsoid_to_slope
from gempy_engine.core.data.engine_grid import RegularGrid
from tests.conftest import plot_pyvista

PLOT = False


def test_implicit_ellipsoid():
    rescaling_factor = 240
    resolution = np.array([20, 4, 20])
    extent = np.array([-500, 500., -500, 500, -450, 550]) / rescaling_factor
    regular_grid = RegularGrid(extent, resolution)

    from gempy_engine.core.backend_tensor import BackendTensor
    scalar = _implicit_3d_ellipsoid_to_slope(
        xyz=BackendTensor.t.array(regular_grid.values),
        center= np.array([0, 0, 0]),
        radius= np.array([1, 1, 2])
    )
    
    if plot_pyvista or False:
        import pyvista as pv
        p = pv.Plotter()
        regular_grid_values = regular_grid.values_vtk_format
        shape = regular_grid_values.shape

        grid_3d = regular_grid_values.reshape(*(resolution + 1), 3).T
        regular_grid_mesh = pv.StructuredGrid(*grid_3d)
        regular_grid_mesh["lith"] = scalar
        p.add_mesh(regular_grid_mesh, show_edges=False, opacity=.5)
        p.show()


def test_transforming_implicit_ellipsoid():
    rescaling_factor = 240
    resolution = np.array([20, 20, 20])
    extent = np.array([-500, 500., -500, 500, -450, 550]) / rescaling_factor
    regular_grid = RegularGrid(extent, resolution)

    xyz = regular_grid.values
    center = np.array([0, 0, 0])
    radius = np.array([1, 1, 2]) * 1
    f = 1
    f2 = 1
    scalar = - np.sum((xyz - center) ** 2.00 / (radius ** 2), axis=1) - 1.0
    # scalar = (((xyz[:, 0] - center[0])) ** 2) / (radius[0] ** 2)
    scalar = scalar - scalar.min()

    sigmoid_slope = 10
    Z_x = scalar
    drift_0 = 8
    scale_0 = 1000
    scalar = scale_0 / (1 + np.exp(-sigmoid_slope * (Z_x - drift_0)))

    if False:
        plt.plot(xyz[:, 0], scalar)
        plt.show()

        plt.plot(xyz[:, 0], foo)
        plt.show()

    # max_slope = 1000
    # min_slope = 0
    # scalar_slope = (scalar - scalar.min()) / (scalar.max() - scalar.min()) * (max_slope - min_slope) + min_slope
    #
    # # cap scalar
    # scalar[scalar > 10] = 10
    #
    # # map scalar between 0 and 1 but heavily skewed with high values
    # scalar2 = np.power(scalar, 15)
    # scalar_slope2 = (scalar2 - scalar2.min()) / (scalar2.max() - scalar2.min()) * (max_slope - min_slope) + min_slope
    #
    if False:
        plt.hist(scalar, bins=100)
        plt.show()
        # plt.hist(scalar_slope)
        # plt.show()
        # plt.hist(scalar2, log=True, bins=100)
        # plt.show()
        # plt.hist(scalar_slope2, log=True, bins=100)
        # plt.show()
        # 

    if plot_pyvista or False:
        import pyvista as pv

        p = pv.Plotter()
        regular_grid_values = regular_grid.values_vtk_format
        shape = regular_grid_values.shape

        grid_3d = regular_grid_values.reshape(*(resolution + 1), 3).T
        regular_grid_mesh = pv.StructuredGrid(*grid_3d)
        regular_grid_mesh["lith"] = scalar
        regular_grid_mesh = regular_grid_mesh.threshold([10, 10000])

        p.add_mesh(regular_grid_mesh, show_edges=False, opacity=.5)
        p.show()


