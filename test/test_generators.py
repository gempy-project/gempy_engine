import numpy as np
import pytest
np.set_printoptions(1)


@pytest.fixture
def input():
    layer1 = np.array([[0, 0], [2, 0]])
    layer2 = np.array([[0, 2], [2, 2]])

    layer1 = np.array([[0, 0], [2, 0], [3, 0], [4, 0]])
    layer2 = np.array([[0, 2], [2, 2], [3, 3]])

    number_of_layer = 2
    number_of_points_per_surface = np.array([layer1.shape[0], layer2.shape[0]])

    def set_rest_ref_matrix(number_of_points_per_surface):
        ref_layer_points = np.repeat(np.stack([layer1[-1], layer2[-1]], axis=0),
                                     repeats=number_of_points_per_surface - 1, axis=0)
        rest_layer_points = np.concatenate([layer1[0:-1], layer2[0:-1]], axis=0)
        return ref_layer_points, rest_layer_points

    ref_layer_points, rest_layer_points = set_rest_ref_matrix(number_of_points_per_surface)
    ## defining the dips position
    G_1 = np.array([[0., 6.], [2., 13.]])

    G_1_x = 1
    G_1_y = 1

    G_1_tiled = np.tile(G_1, [2, 1])

    dipsref = np.vstack((G_1_tiled, ref_layer_points))
    dipsrest = np.vstack((G_1_tiled, rest_layer_points))
    n_dips = G_1.shape[0]
    n_points = dipsref.shape[0]
    return dipsref, dipsrest, n_dips


def test_create_r(input):
    dipsref, dipsrest = input

    a = dipsref[:, None, :]
    b = dipsref[None, :, :]
    a2 = dipsrest[:, None, :]
    b2 = dipsrest[None, :, :]

    sed_ref_ref = np.sqrt(((a - b) ** 2).sum(-1))
    sed_rest_rest = np.sqrt(((a2 - b2) ** 2).sum(-1))
    # This is completely unnecessary since it is the same as sed_rest_ref
    #sed_ref_rest = np.sqrt(((a - b2) ** 2).sum(-1))

    sed_rest_ref = np.sqrt(((a2 - b) ** 2).sum(-1))

    print(sed_ref_ref)
    # print(sed_ref_rest)
    print(sed_rest_rest)
    print(sed_rest_ref)


def test_create_hu_dips(input):
    n_dim = 2
    dipsref, dipsrest, n_dips = input
    dips = np.zeros_like(dipsref)
    sel_0 = np.zeros_like(dipsref)
    sel_0[:n_dips, 0] = 1
    sel_0[n_dips:n_dips*2, 1] = 1

    sel_1 = np.zeros_like(dipsref)
    sel_1[:n_dips*n_dim, :] = 1

    dips[:n_dips * n_dim] = dipsref[:n_dips * n_dim]
    dips_0 = dips[:, None, :].copy()
    dips_1 = dips[None, :, :].copy()

    hu = (dips_0 - dips_1) * sel_0 * sel_1

    print(hu.sum(axis=-1))


def test_create_h(input):
    dipsref, dipsrest = input

    a = dipsref[:, None, :]
    b = dipsref[None, :, :]
    a2 = dipsrest[:, None, :]
    b2 = dipsrest[None, :, :]

    hu_ref = a - b
    s = hu_ref[:, :, 1]
    print(s)

