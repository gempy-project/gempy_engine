import numpy as np
import pytest
np.set_printoptions(1)


@pytest.fixture
def input():
    layer1 = np.array([[0, 0], [2, 0]])
    layer2 = np.array([[0, 2], [2, 2]])

    layer1 = np.array([[5, 7], [2, 0], [3, 0], [4, 0]])
    layer2 = np.array([[6, 12], [2, 2], [3, 3]])

    number_of_layer = 2
    number_of_points_per_surface = np.array([layer1.shape[0], layer2.shape[0]])

    def set_rest_ref_matrix(number_of_points_per_surface):
        ref_layer_points = np.repeat(np.stack([layer1[0], layer2[0]], axis=0),
                                     repeats=number_of_points_per_surface - 1, axis=0)
        rest_layer_points = np.concatenate([layer1[1:], layer2[1:]], axis=0)
        return ref_layer_points, rest_layer_points

    ref_layer_points, rest_layer_points = set_rest_ref_matrix(number_of_points_per_surface)
    ## defining the dips position
    G_1 = np.array([[0., 6.], [2., 13.]])

    G_1_x = 1
    G_1_y = 1

    G_1_tiled = np.tile(G_1, [2, 1])
    uni = np.array([[1, 0],
                    [0, 1],
                    [1, 0],
                    [0, 1],
                    [1, 0],
                    [1, 0]])

    dipsref = np.vstack((G_1_tiled, ref_layer_points))
    dipsrest = np.vstack((G_1_tiled, rest_layer_points))
    n_dips = G_1.shape[0]
    n_points = dipsref.shape[0]
    return dipsref, dipsrest, n_dips


def test_create_r(input):
    dipsref, dipsrest, _ = input

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


def test_perpendicular_matrix(input):
    _, _ , n_dips = input
    perp_m = np.zeros_like(_)
    perp_m[:n_dips, 0] = 1
    perp_m[n_dips:2*n_dips, 1] = 1

    perp_i = perp_m[:, None, :].copy()
    perp_j = perp_m[None, :, :].copy()

    perp_ma = perp_i*perp_j
    perp_m, perp_ma
    print(perp_m, perp_ma)



def test_create_hu_dips(input):
    n_dim = 2
    dipsref, dipsrest, n_dips = input

    dips = np.zeros_like(dipsref) # This is unnecessary with sel

    sel_0 = np.zeros_like(dipsref)
    sel_0[:n_dips, 0] = 1
    sel_0[n_dips:n_dips*2, 1] = 1

    sel_1 = np.zeros_like(dipsref)
    sel_1[:n_dips*n_dim, :] = 1

    # dips[:n_dips * n_dim] = dipsref[:n_dips * n_dim] Unnecessary with sel
    dips = dipsref
    dips_0 = dips[:, None, :].copy()
    dips_1 = dips[None, :, :].copy()

    sel_01 = sel_0[:, None, :].copy()
    sel_11 = sel_1[None, :, :].copy()

    sel_02 = sel_1[:, None, :].copy()
    sel_12 = sel_0[None, :, :].copy()

    hu = (dips_0 - dips_1) * (sel_01 * sel_11)
    hv = (dips_0 - dips_1) * (sel_02 * sel_12)
    print(hu.sum(axis=-1))
    print(hv.sum(axis=-1))



def test_hu_ref_select(input):
    n_dim = 2
    dipsref, dipsrest, n_dips = input

    a = np.zeros((dipsref.shape[0], 1))
    b = np.zeros((dipsref.shape[0], 1))
    a[:n_dips * 2, 0] = -1
    b[n_dips * 2:, 0] = -1
    a = a[None, :, :]
    b = b[:, None, :]

    c = a-b
    print(c)


def test_uni_select(input):
    n_dim = 2
    dipsref, dipsrest, n_dips = input
    perp_m = np.zeros_like(dipsref)
    perp_m[:n_dips, 0] = 1
    perp_m[-2:, 1] = 1

    perp_i = perp_m[None, :, :].copy()
    perp_j = perp_m[:, None, :].copy()

    perp_i2 = perp_m[None, :, :].copy()
    perp_j2 = perp_m[:, None, :].copy()

    perp_ma = (perp_i*perp_j)+(perp_i2*perp_j2)
    perp_m, perp_ma
    print(perp_m, perp_ma)


def test_uni_select2(input):
    n_dim = 2
    uni_terms = 2
    dipsref, dipsrest, n_dips = input

    #perp_cgi = np.zeros((dipsref.shape[0], 2))
    #perp_cgi[:n_dips * 2, 1] = 1
    #perp_cgi[n_dips * 2:, 0] = 0

    # sel_0 = perp_cgi[:, None, :]
    # sel_1 = perp_cgi[None, :, :]
    # sel = (sel_0 - sel_1).sum(axis=-1)
    #sel = (sel_0 * sel_1 - 1).prod(axis=-1)
    #print(sel)
    dipsref_0 = dipsref[:, None, :].copy()
    dipsref_1 = dipsref[None, :, :].copy()


    sel_0 = np.zeros_like(dipsref)
    sel_0[:-uni_terms, 0] = 1
    sel_0[-uni_terms:, 1] = 1

    sel_1 = np.zeros_like(dipsref)
    sel_1[-uni_terms:, :] = 1

    sel_01 = sel_0[:, None, :].copy()
    sel_11 = sel_1[None, :, :].copy()

    sel_02 = sel_1[:, None, :].copy()
    sel_12 = sel_0[None, :, :].copy()

    hu_ref = (sel_01 * sel_11)
    hv_ref = (sel_02 * sel_12)

    print(hu_ref.sum(axis=-1) - hv_ref.sum(axis=-1))


def test_create_hu_ref_rest(input):
    n_dim = 2
    uni_terms = 2
    dipsref, dipsrest, n_dips = input

    #perp_cgi = np.zeros((dipsref.shape[0], 2))
    #perp_cgi[:n_dips * 2, 1] = 1
    #perp_cgi[n_dips * 2:, 0] = 0

    # sel_0 = perp_cgi[:, None, :]
    # sel_1 = perp_cgi[None, :, :]
    # sel = (sel_0 - sel_1).sum(axis=-1)
    #sel = (sel_0 * sel_1 - 1).prod(axis=-1)
    #print(sel)
    dipsref_0 = dipsref[:, None, :].copy()
    dipsref_1 = dipsref[None, :, :].copy()


    sel_0 = np.zeros_like(dipsref)
    sel_0[:n_dips, 0] = 1
    sel_0[n_dips:n_dips * 2, 1] = 1

    sel_1 = np.zeros_like(dipsref)
    sel_1[n_dips * n_dim:-uni_terms, :] = 1

    sel_01 = sel_0[:, None, :].copy()
    sel_11 = sel_1[None, :, :].copy()

    sel_02 = sel_1[:, None, :].copy()
    sel_12 = sel_0[None, :, :].copy()

    hu_ref = (dipsref_0 - dipsref_1) * (sel_01 * sel_11)
    hv_ref = (dipsref_0 - dipsref_1) * (sel_02 * sel_12)

    print(hu_ref.sum(axis=-1) - hv_ref.sum(axis=-1))

def test_create_ug(input):
    n_dim = 2
    uni_terms = 6
    dipsref, dipsrest, n_dips = input

    dipsref[:n_dips, 1] = 0
    dipsref[n_dips:n_dips*n_dim, 0] = 0

    # Remove the sp
    dipsref_d1 = np.zeros_like(dipsref)
    dipsref_d1[:n_dips, 0] = 1
    dipsref_d1[n_dips:n_dips*n_dim, 1] = 1
    # Degree 1

    uni_a = np.array([[1, 0],
                      [0, 1],
                      [0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0]])

    dipsref_a = np.vstack((dipsref_d1, uni_a))

    dipsref_0a = dipsref_a[:, None, :].copy()
    dipsref_1a = dipsref_a[None, :, :].copy()
    (dipsref_0a * dipsref_1a)[:,:, 0]
    ui_a = (dipsref_0a * dipsref_1a).sum(axis=-1)
    print('ui_a: \n', ui_a)

    # Degree 2
    uni_b = np.array([[0, 0],
                      [0, 0],
                      [2, 0],
                      [0, 2],
                      [0, 1],
                      [1, 0]])
    dipsref_b = np.vstack((dipsref, uni_b))
    dipsref_0b = dipsref_b[:, None, :].copy()
    dipsref_1b = dipsref_b[None, :, :].copy()
    (dipsref_0b * dipsref_1b)[:,:, 0]
    ui_b = (dipsref_0b * dipsref_1b).sum(axis=-1)
    print('ui_b: \n', ui_b)

    print('ug: \n', ui_a+ui_b)


def test_create_us(input):
    n_dim = 2
    uni_terms = 6
    dipsref, dipsrest, n_dips = input

    # Remove the grads
    dipsref[:n_dips*n_dim, :] = 0

    # Degree 1:
    uni_a = np.array([[1, 0],
                      [0, 1],
                      [0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0]])

    dipsref_a = np.vstack((dipsref, uni_a))
    dipsref_0a = dipsref_a[:, None, :].copy()
    dipsref_1a = dipsref_a[None, :, :].copy()
    ui_a = (dipsref_0a * dipsref_1a).sum(axis=-1)
    print('ui_a: \n', ui_a)

    # Degree 2
    uni_b1 = np.array([[0, 0],
                       [0, 0],
                       [1, 0],
                       [0, 1],
                       [1, 0],
                       [1, 0]])

    dipsref_b1 = np.vstack((dipsref, uni_b1))
    dipsref_0b1 = dipsref_b1[:, None, :].copy()
    dipsref_1b1 = dipsref_b1[None, :, :].copy()
    ui_b1 = (dipsref_0b1 * dipsref_1b1).sum(axis=-1)
    print('ui_b2: \n', ui_b1)


    uni_b2 = np.array([[0, 0],
                       [0, 0],
                       [1, 0],
                       [0, 1],
                       [0, 1],
                       [0, 1]])

    dipsref_b2 = np.vstack((dipsref, uni_b2))
    dipsref_0b2 = dipsref_b2[:, None, :].copy()
    dipsref_1b2 = dipsref_b2[None, :, :].copy()
    ui_b2 = (dipsref_0b2 * dipsref_1b2).sum(axis=-1)
    print('ui_b2: \n', ui_b2)

    print('ui_ref: \n', ui_a+(ui_b1*ui_b2))

    # perp_cgi = np.zeros((dipsref.shape[0], 2))
    # perp_cgi[:n_dips * 2, 1] = 1
    # perp_cgi[n_dips * 2:, 0] = 0

    # sel_0 = perp_cgi[:, None, :]
    # sel_1 = perp_cgi[None, :, :]
    # sel = (sel_0 - sel_1).sum(axis=-1)
    # sel = (sel_0 * sel_1 - 1).prod(axis=-1)
    # print(sel)
    # dipsref_0 = dipsref[:, None, :].copy()
    # dipsref_0[:n_dips*n_dim, :, :] = 0
    #
    # dipsref_1 = dipsref[None, :, :].copy()
    # dipsref_1[:, :n_dips * n_dim, :] = 0
    #
    # sel_0 = np.zeros_like(dipsref)
    # sel_0[:n_dips, 0] = 1
    # sel_0[n_dips:n_dips * 2, 1] = 1
    #
    # sel_1 = np.zeros_like(dipsref)
    # sel_1[-uni_terms:, :] = 1
    #
    # sel_01 = sel_0[:, None, :].copy()
    # sel_11 = sel_1[None, :, :].copy()
    #
    # sel_02 = sel_1[:, None, :].copy()
    # sel_12 = sel_0[None, :, :].copy()
    #
    # u_z = dipsref_0 * dipsref_1 #* (sel_01 * sel_11)
    # print(u_z.sum(axis=-1))
    #
    # dipsref_01 = dipsref[:, None, :].copy()
    # dipsref_01[:n_dips * n_dim, :, :] = 0
    # dipsref_01[n_dips * n_dim, :, :] = 0
    #
    # dipsref_11 = dipsref[None, :, :].copy()
    # dipsref_11[:, :n_dips * n_dim, :] = 0
    #
    # # hv_ref = (dipsref_0 - dipsref_1) * (sel_02 * sel_12)
    # print('\n')
    # #print(hu_ref.sum(axis=-1) - hv_ref.sum(axis=-1))


def test_create_h(input):
    dipsref, dipsrest, _ = input

    a = dipsref[:, None, :]
    b = dipsref[None, :, :]
    a2 = dipsrest[:, None, :]
    b2 = dipsrest[None, :, :]

    hu_ref = a - b
    s = hu_ref[:, :, 1]
    print(s)



