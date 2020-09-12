import pytest
import tensorflow as tf
from pandas import np

from gempy_engine.config import use_tf
from gempy_engine.data_structures.public_structures import OrientationsInput, KrigingParameters, SurfacePointsInput
from gempy_engine.graph_model import GemPyEngine, GemPyEngineTF, squared_euclidean_distances, cartesian_distances, \
    compute_perpendicular_matrix, compute_cov_gradients, get_ref_rest


@pytest.fixture
def moureze_orientations(moureze):
    _, ori = moureze
    ori_t = OrientationsInput(
        ori[['X', 'Y', 'Z']].values,
        dip_gradients=ori[['G_x', 'G_y', 'G_z']].values)

    return ori_t

@pytest.fixture
def moureze_sp(moureze):
    sp, ori = moureze
    sp_t = SurfacePointsInput(sp[['X', 'Y', 'Z']].values,
                              sp['smooth'])
    return sp_t

@pytest.fixture()
def moureze_kriging():
    return KrigingParameters(10000, 50000)


@pytest.fixture
def ge():
    return GemPyEngineTF()


def test_dips_position_tiled(moureze_orientations):
    a = tf.tile(moureze_orientations.dip_positions, (1, 3))
    print(a)


def test_dips_position_tiled2(ge, moureze_orientations):
    tf.debugging.set_log_device_placement(True)

    s = ge.tile_dip_positions(moureze_orientations.dip_positions, 3)
    print(s)
    return s


def test_dips_position_tiled_jacobian(ge, moureze_orientations):
    # tf.debugging.set_log_device_placement(True)
    dp = tf.Variable(moureze_orientations.dip_positions)
    with tf.GradientTape(persistent=True) as tape:
        s = ge.tile_dip_positions(dp, 3)
        su = tf.reduce_sum(s)

    print(su)
    foo = tape.jacobian(su, dp)
    print(foo)


def test_get_ref_rest(moureze_sp):
    s = get_ref_rest(
        moureze_sp,
        np.array([10, 50, moureze_sp.sp_positions.shape[0] - 60 - 3],
                 dtype='int32')
    )
    print(s)
    return s


def test_squared_euclidean_distances(moureze_orientations):

    # Test numpy/eager
    np_array = moureze_orientations.dip_positions
    s = squared_euclidean_distances(np_array,
                                    np_array)
    print(s, type(s))

    if use_tf:
        # Test TF and gradients?
        tf_variable = tf.Variable(np_array)
        with tf.GradientTape(persistent=True) as tape:
            s = squared_euclidean_distances(tf_variable, tf_variable)
            su = tf.reduce_sum(s)

        ja = tape.jacobian(su, tf_variable)
        print(ja)

    return s


def test_cartesian_distances(moureze_orientations):
    np_array = moureze_orientations.dip_positions
    s = cartesian_distances(np_array, np_array, 3)
    print(s)
    return s


def test_perpendicularity_matrix():
    s = compute_perpendicular_matrix(100)
    print(s)
    return s


def test_cov_gradients(ge, moureze_orientations, moureze_kriging):

    dip_tiled = test_dips_position_tiled2(ge, moureze_orientations)
    s = ge.cov_gradients(
        moureze_orientations,
        dip_tiled,
        moureze_kriging,
        3
    )
    print(s)
    return s

def test_cov_sp():
    raise NotImplementedError


def test_covariance_matrix():
    raise NotImplementedError


def test_solver():
    """Here we need to test all the different methods"""
    raise NotImplementedError
