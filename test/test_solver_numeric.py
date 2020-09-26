import pytest
import numpy as np

from gempy_engine.data_structures.private_structures import SurfacePointsInternals, OrientationsInternals
from gempy_engine.data_structures.public_structures import OrientationsInput, InterpolationOptions
from gempy_engine.systems.generators import tile_dip_positions
from gempy_engine.systems.kernel.kernel import vectors_preparation


@pytest.fixture()
def simple_model():
    spi = SurfacePointsInternals(
        ref_surface_points=np.array(
            [[4, 0],
             [4, 0],
             [4, 0],
             [3, 3],
             [3, 3]]),
        rest_surface_points=np.array([[0, 0],
                                      [2, 0],
                                      [3, 0],
                                      [0, 2],
                                      [2, 2]]),
        nugget_effect_ref_rest=0
    )

    ori_i = OrientationsInput(
        dip_positions=np.array([[0, 6],
                                [2, 13]]),
        nugget_effect_grad=0.0000001
    )
    dip_tiled = tile_dip_positions(ori_i.dip_positions, 2)
    ori_int = OrientationsInternals(dip_tiled)
    kri = InterpolationOptions(5, 5 ** 2 / 14 / 3, 0, i_res=1, gi_res=1,
                               number_dimensions=2)

    return spi, ori_int, kri


def test_vector_preparation(simple_model):
    kri = vectors_preparation(*simple_model)
    print(kri)