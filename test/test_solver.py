import pytest
import tensorflow as tf
from pandas import np

from gempy_engine.config import use_tf
from gempy_engine.data_structures.private_structures import SurfacePointsInternals,\
    OrientationsGradients
from gempy_engine.data_structures.public_structures import OrientationsInput, InterpolationOptions, SurfacePointsInput
from gempy_engine.graph_model import GemPyEngineTF, squared_euclidean_distances, get_ref_rest, tile_dip_positions
from gempy_engine.systems.kernel.aux_functions import cartesian_distances, compute_perpendicular_matrix, \
    covariance_assembly, b_scalar_assembly
from gempy_engine.systems.kernel.kernel_legacy import cov_gradients_f, cov_sp_f, cov_sp_grad_f, drift_uni_f, create_covariance_legacy
from gempy_engine.systems.reductions import solver


@pytest.fixture
def moureze_orientations(moureze):
    _, ori = moureze
    ori_t = OrientationsInput(
        ori[['X', 'Y', 'Z']].values,
        dip_gradients=ori[['G_x', 'G_y', 'G_z']].values)

    return ori_t


@pytest.fixture
def moureze_orientations_heavy(moureze):
    _, ori = moureze
    n = 2
    ori_poss = ori[['X', 'Y', 'Z']].values,
    ori_pos = ori_poss[0]
    ori_grad = ori[['G_x', 'G_y', 'G_z']].values

    for i in range(n):
        ori_pos = np.vstack([ori_pos, ori_pos + np.array([i * 100, i * 100, i * 100])])
        ori_grad = np.vstack([ori_grad, ori_grad])

    ori_t = OrientationsInput(ori_pos,
                              dip_gradients=ori_grad)

    return ori_t


@pytest.fixture
def moureze_orientations_lite(moureze):
    _, ori = moureze
    ori_t = OrientationsInput(
        ori[['X', 'Y', 'Z']].values[:50],
        dip_gradients=ori[['G_x', 'G_y', 'G_z']].values[:50])

    return ori_t


@pytest.fixture
def moureze_sp(moureze):
    sp, ori = moureze
    sp_t = SurfacePointsInput(sp[['X', 'Y', 'Z']].values,
                              sp['smooth'].values)
    return sp_t


@pytest.fixture
def moureze_sp_lite(moureze):
    sp, ori = moureze
    sp_t = SurfacePointsInput(sp[['X', 'Y', 'Z']].values[:100],
                              sp['smooth'].values[:100])
    return sp_t


@pytest.fixture()
def moureze_kriging():
    return InterpolationOptions(10000, 50000)


@pytest.fixture
def ge():
    return GemPyEngineTF()


def test_dips_position_tiled(moureze_orientations):
    a = tf.tile(moureze_orientations.dip_positions, (1, 3))
    print(a)


def test_dips_position_tiled2(moureze_orientations):
    tf.debugging.set_log_device_placement(True)

    s = tile_dip_positions(moureze_orientations.dip_positions, 3)
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


def test_cov_gradients(moureze_orientations, moureze_kriging):
    dip_tiled = test_dips_position_tiled2(moureze_orientations)
    s = cov_gradients_f(
        moureze_orientations,
        dip_tiled,
        moureze_kriging,
        3
    )
    print(s)
    return s


def test_cov_gradients_simple():
    ori = OrientationsInput(np.array([[0., 6.], [2., 13.]]),
                            nugget_effect_grad=0.0000001)
    kri = InterpolationOptions(5, 5 ** 2 / 14 / 3)
    dip_tiled = tile_dip_positions(ori.dip_positions, 2)
    s = cov_gradients_f(
        ori,
        dip_tiled,
        kri,
        2
    )
    print(s)
    return s


def test_cov_sp(moureze_sp, moureze_kriging):
    sp_i = SurfacePointsInternals(*get_ref_rest(
        moureze_sp,
        np.array([moureze_sp.sp_positions.shape[0] - 1], dtype='int32')
    ))

    s = cov_sp_f(
        sp_i,
        moureze_kriging
    )
    print(s)
    return s


def test_cov_sp_grad(moureze_sp, moureze_orientations, moureze_kriging):
    dip_tiled = test_dips_position_tiled2(moureze_orientations)

    sp_i = SurfacePointsInternals(*get_ref_rest(
        moureze_sp,
        np.array([moureze_sp.sp_positions.shape[0] - 1], dtype='int32')
    ))

    s = cov_sp_grad_f(moureze_orientations,
                      dip_tiled,
                      sp_i,
                      moureze_kriging,
                      1)
    print(s)
    return s


def test_drift_uni(moureze_orientations, moureze_sp):
    sp_i = SurfacePointsInternals(*get_ref_rest(
        moureze_sp,
        np.array([moureze_sp.sp_positions.shape[0] - 1], dtype='int32')
    ))

    s = drift_uni_f(
        moureze_orientations.dip_positions,
        sp_i,
        4,
        1
    )

    print(s)
    return s


def test_creat_covariance():
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
    kri = InterpolationOptions(5, 5 ** 2 / 14 / 3, 0, i_res=1, gi_res=1,
                               number_dimensions=2)

    # ci
    cov_sp = cov_sp_f(
        spi,
        kri
    )
    print(cov_sp, '\n')

    cov = create_covariance_legacy(spi, ori_i, dip_tiled, kri)
    np.set_printoptions(precision=2)
    print('\n', cov)
    #print(cov.sum(axis=1), '\n', cov.sum(axis=0))
    return cov

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
    kri = InterpolationOptions(5, 5 ** 2 / 14 / 3, 0, i_res=1, gi_res=1,
                               number_dimensions=2)


def test_covariance_matrix(moureze_sp, moureze_orientations, moureze_kriging):
    cov_g = test_cov_gradients(moureze_orientations, moureze_kriging)
    cov_g_sp = test_cov_sp_grad(moureze_sp, moureze_orientations, moureze_kriging)
    cov_sp = test_cov_sp(moureze_sp, moureze_kriging)
    drift_grad, drift_sp = test_drift_uni(moureze_orientations, moureze_sp)
    cov_matrix = covariance_assembly(cov_sp, cov_g, cov_g_sp,
                                     drift_grad, drift_sp)

    print(cov_matrix, cov_matrix.shape[0])
    return cov_matrix


def test_b_scalar_assembly(moureze_orientations, c_size=4783):
    s = b_scalar_assembly(
        OrientationsGradients(
            moureze_orientations.dip_gradients[0],
            moureze_orientations.dip_gradients[1],
            moureze_orientations.dip_gradients[2],
        ),
        c_size
    )
    print(s)
    return s


def test_solver_lite(moureze_sp_lite, moureze_orientations_lite, moureze_kriging):
    """Here we need to test all the different methods"""
    cov_matrix = test_covariance_matrix(
        moureze_sp_lite,
        moureze_orientations_lite,
        moureze_kriging
    )

    b = test_b_scalar_assembly(moureze_orientations_lite, cov_matrix.shape[0])

    s = solver(cov_matrix, b)
    print(s)


def test_solver(moureze_sp, moureze_orientations, moureze_kriging):
    """Here we need to test all the different methods"""
    cov_matrix = test_covariance_matrix(
        moureze_sp,
        moureze_orientations,
        moureze_kriging
    )

    b = test_b_scalar_assembly(moureze_orientations, cov_matrix.shape[0])

    s = solver(cov_matrix, b)
    print(s)


def test_solver_heavy(moureze_sp, moureze_orientations_heavy, moureze_kriging):
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')
    tf.debugging.set_log_device_placement(True)
    # with tf.device('/device:GPU:0'):
    tf.profiler.experimental.start('logdir')
    for step in range(1):
        with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
            """Here we need to test all the different methods"""
            cov_matrix = test_covariance_matrix(
                moureze_sp,
                moureze_orientations_heavy,
                moureze_kriging
            )

            b = test_b_scalar_assembly(moureze_orientations_heavy, cov_matrix.shape[0])

            s = solver(cov_matrix, b)
        tf.profiler.experimental.stop()

    print(s)
