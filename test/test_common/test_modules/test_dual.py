import numpy as np
import os
import matplotlib.pyplot as plt
import pytest

from gempy_engine.core.data.exported_structs import OctreeLevel
from gempy_engine.modules.dual_contouring.dual_contouring_interface import solve_qef_3d, QEF

dir_name = os.path.dirname(__file__)

@pytest.mark.skip(reason="Not Implemented yet")
def test_dual_contouring_from_octree(simple_model_output):
    octrees = simple_model_output.octrees
    last_level: OctreeLevel = octrees[1]

    x = last_level.xyz_coords[0, 0]
    y = last_level.xyz_coords[0, 1]
    z = last_level.xyz_coords[0, 2]


    qef = QEF.make_3d(positions, normals)

    residual, v = qef.solve()

    solve_qef_3d()




def test_dual_contouring(plot=True):
    zx = np.load(dir_name + "/solutions/zx.npy")
    gx = np.load(dir_name + "/solutions/gx.npy")
    gy = np.load(dir_name + "/solutions/gy.npy")
    gz = np.load(dir_name + "/solutions/gz.npy")

    # TODO: Compute normal at edges



    if plot:
        plt.contourf(zx[:-7].reshape(50, 5, 50)[:, 2, :].T, N=40, cmap="autumn")
        plt.quiver(
            gx[:-7].reshape(50, 5, 50)[:, 2, :].T,
            gz[:-7].reshape(50, 5, 50)[:, 2, :].T,
            scale=10
        )

        plt.colorbar()


        plt.show()