import pytest
import tensorflow as tf

from gempy_engine.data_structures.public_structures import OrientationsInput
from gempy_engine.graph_model import GemPyEngine


@pytest.fixture
def moureze_orientations(moureze):
    _, ori = moureze
    ori_t = OrientationsInput(
        ori[['X', 'Y', 'Z']],
        dip_gradients=ori[['G_x', 'G_y', 'G_z']])

    return ori_t


@pytest.fixture
def ge():
    return GemPyEngine()


def test_dips_position_tiled(moureze_orientations):
    a = tf.tile(moureze_orientations.dip_positions, (1, 3))
    print(a)


def test_dips_position_tiled2(ge, moureze_orientations):
    tf.debugging.set_log_device_placement(True)

    s = ge.tile_dip_positions(moureze_orientations.dip_positions, 3)
    print(s)


def test_dips_position_tiled_jacobian(ge, moureze_orientations):
    # tf.debugging.set_log_device_placement(True)
    dp = tf.Variable(moureze_orientations.dip_positions.values)
    with tf.GradientTape(persistent=True) as tape:
        s = ge.tile_dip_positions(dp, 3)
        su = tf.reduce_sum(s)

    print(su)
    foo = tape.jacobian(su, dp)
    print(foo)


def test_squared_euclidean_distances():
    raise NotImplementedError


def test_cov_gradients():
    raise NotImplementedError


def test_covariance_matrix():
    raise NotImplementedError


def test_solver():
    """Here we need to test all the different methods"""
    raise NotImplementedError