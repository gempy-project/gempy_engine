import pytest
import sys
sys.path.append("/home/miguel/libkeops")

pykeops = pytest.importorskip("pykeops")
from pykeops.numpy import LazyTensor, Genred, Vi, Vj, Pm
import numpy as np

pykeops.config.verbose = True


#@pytest.mark.skip('Only trigger manually when there is something wrong with pykeops compilation', )
def test_keops_run():
    import pykeops
    pykeops.verbose = True
    import os
    os.environ["PYKEOPS_VERBOSE"] = "3"  # Maximum verbosity
    os.environ["PYKEOPS_DEBUG"] = "1"    # Enables detailed debug mode
    print(pykeops.get_build_folder())  # display new build_folder
    pykeops.set_build_folder("/home/miguel/.s")

    pykeops.clean_pykeops()  # just in case old build files are still present
    pykeops.test_numpy_bindings()


# @pytest.mark.skip('Only trigger manually when there is something wrong with'
#                   'pykeops compilation', )
def test_basic_op():

    M, N = 1000, 20000
    x = np.random.rand(M, 2)
    y = np.random.rand(N, 2)
    from pykeops.numpy import LazyTensor
    pykeops.clean_pykeops()
    x_i = LazyTensor(
        x[:, None, :]
    )  # (M, 1, 2) KeOps LazyTensor, wrapped around the numpy array x
    
    y_j = LazyTensor(
        y[None, :, :]
    )  # (1, N, 2) KeOps LazyTensor, wrapped around the numpy array y

    D_ij = ((x_i - y_j) ** 2)  # **Symbolic** (M, N) matrix of squared distances
    foo = D_ij.sum_reduction(axis=0, backend="CPU")

    print(foo)





@pytest.mark.skip('DEP test')
def test_pykeops_wrong():
    layer1 = np.array([[0, 0], [2, 0]])
    layer2 = np.array([[0, 2], [2, 2]])

    layer1 = np.array([[0, 0], [2, 0], [3, 0], [4, 0]])
    layer2 = np.array([[0, 2], [2, 2], [3, 3]])

    number_of_layer = 2
    number_of_points_per_surface = np.array([layer1.shape[0], layer2.shape[0]])

    def set_rest_ref_matrix(number_of_points_per_surface):
        ref_layer_points = np.repeat(np.stack([layer1[-1], layer2[-1]], axis=0),
                                     repeats=number_of_points_per_surface - 1,
                                     axis=0)
        rest_layer_points = np.concatenate([layer1[0:-1], layer2[0:-1]], axis=0)
        return ref_layer_points, rest_layer_points

    ref_layer_points, rest_layer_points = set_rest_ref_matrix(
        number_of_points_per_surface)
    ## defining the dips position
    G_1 = np.array([[0., 6.], [2., 13.]])

    G_1_x = 1
    G_1_y = 1

    G_1_tiled = np.tile(G_1, [2, 1])

    dipsref = np.vstack((G_1_tiled, ref_layer_points))
    dipsrest = np.vstack((G_1_tiled, rest_layer_points))
    n_dips = G_1.shape[0]
    n_points = dipsref.shape[0]
    dipsref, dipsrest, n_dips, n_points

    z = np.zeros((n_points, 1))
    z[:n_dips] = 1
    # z[n_dips*2:n_dips*3] = 1
    z

    z2 = np.zeros((n_points, 1))
    z2[n_dips:n_dips * 2] = 1
    z2

    perp_m = np.zeros((n_points, 2))
    perp_m[:n_dips, 0] = 1
    perp_m[n_dips:2 * n_dips, 1] = 1
    perp_ma = perp_m @ perp_m.T
    perp_m, perp_ma

    perp_cgi = np.zeros((n_points, 2))
    perp_cgi[:n_dips * 2, 1] = 1
    perp_cgi[n_dips * 2:, 0] = 1
    perp_cgi_m = -1 * (perp_cgi @ perp_cgi.T - 1)
    perp_cgi, perp_cgi_m

    perp_cg = np.zeros((n_points, 1))
    perp_cg[:n_dips * 2, 0] = 1
    # perp_m[n_dips:2*n_dips, 1] = 1
    perp_cg_ma = perp_cg @ perp_cg.T
    perp_cg, perp_cg_ma

    perp_ci = np.zeros((n_points, 1))
    perp_ci[n_dips * 2:, 0] = 1
    # perp_m[n_dips:2*n_dips, 1] = 1
    perp_ci_ma = perp_ci @ perp_ci.T
    perp_ci_ma

    a_T = 5
    c_o_T = a_T ** 2 / 14 / 3
    a_T, c_o_T

    # ==================================

    # x_i = Vi(0, 2)
    # x_j = Vj(1, 2)
    x_iref = Vi(0, 2)
    x_jref = Vj(1, 2)

    z_i = Vi(2, 1)  # Used for selecting hu1
    z_i2 = Vi(3, 1)  # Used for selecting hu2
    z_j = Vj(4, 1)  # Used for selecting hv1
    z_j2 = Vj(5, 1)  # Used for selecting hv2
    # p_cg_i = Vi(6,1)
    # p_cg_j= Vj(7,1)
    nugget = Pm(8, 1)  # Avoid a/0
    f1g = Pm(9, 1)
    f2g = Pm(10, 1)
    f3g = Pm(11, 1)
    p_i = Vi(12, 2)  # Select matrix
    p_j2 = Vj(13, 2)  # Select matrix
    c_o = Pm(14, 1)
    range_ = Pm(15, 1)

    # ===============

    x_irest = Vi(16, 2)
    x_jrest = Vj(17, 2)
    i_res = Pm(18, 1)
    f1i = Pm(19, 1)
    f2i = Pm(20, 1)
    f3i = Pm(21, 1)
    # x_idip = Vi(0, 2)
    # x_jref = Vj(1, 2)
    # x_jrest = Vj(2, 2)
    # z_i = Vi(3, 1)
    # z_i2 = Vi(4, 1)

    # ===============

    gi_res = Pm(22, 1)
    f1gi = Pm(23, 1)
    f2gi = Pm(6, 1)
    f3gi = Pm(7, 1)
    # p_cg_i = Vi(24,1)
    # p_cg_j= Vj(25,1)
    # p_cgi_i = Vi(26, 2)
    # p_cgi_j = Vj(27, 2)

    sed_ref_ref = x_iref.sqdist(x_jref).sqrt()
    sed_rest_rest = x_irest.sqdist(x_jrest).sqrt()
    sed_ref_rest = x_iref.sqdist(x_jrest).sqrt()
    sed_rest_ref = x_irest.sqdist(x_jref).sqrt()
    # huke = x_i - x_j

    hu_temp_ref = x_iref - x_jref

    hu_0 = hu_temp_ref[0] * z_i
    hu_1 = hu_temp_ref[1] * z_i2
    hu_dip = hu_0 + hu_1

    hv_0 = hu_temp_ref[0] * z_j
    hv_1 = hu_temp_ref[1] * z_j2
    hv_dip = -(hv_0 + hv_1)

    # perp_cg_m = p_cg_i * p_cg_j

    # hu1 = huke[0] * z_i
    # hv1 = - huke[0] * z_j
    # hu1.ranges = ranges_ij0
    # hu2 = huke[1] * z_i2
    # hv2 = - huke[1] * z_j2
    # hu2.ranges = ranges_ij1

    # hu = hu1+hu2
    # hv = hv1+hv2
    t1 = (hu_dip * hv_dip) / (sed_ref_ref ** 2 + nugget)

    t2 = (-c_o * ((-14 / range_ ** 2) + f1g * sed_ref_ref / range_ ** 3 -
                  f2g * sed_ref_ref ** 3 / range_ ** 5 +
                  f3g * sed_ref_ref ** 5 / range_ ** 7)
          ) + (
                 c_o * 7 * (
                     9 * sed_ref_ref ** 5 - 20 * range_ ** 2 * sed_ref_ref ** 3 +
                     15 * range_ ** 4 * sed_ref_ref - 4 * range_ ** 5) / (
                             2 * range_ ** 7))

    t3 = (p_i * p_j2).sum(-1) * c_o * (
                (-14 / range_ ** 2) + f1g * sed_ref_ref / range_ ** 3 -
                f2g * sed_ref_ref ** 3 / range_ ** 5 + f3g * sed_ref_ref ** 5 / range_ ** 7)

    cg = (t1 * t2 - t3)  # *perp_cg_m

    ci = (c_o * i_res * (
        # (sed_rest_rest < range) *  # Rest - Rest Covariances Matrix
            (1 - 7 * (sed_rest_rest / range_) ** 2 +
             f1i * (sed_rest_rest / range_) ** 3 -
             f2i * (sed_rest_rest / range_) ** 5 +
             f3i * (sed_rest_rest / range_) ** 7) -
            # ((sed_ref_rest < range) *  # Reference - Rest
            ((1 - 7 * (sed_ref_rest / range_) ** 2 +
              f1i * (sed_ref_rest / range_) ** 3 -
              f2i * (sed_ref_rest / range_) ** 5 +
              f3i * (sed_ref_rest / range_) ** 7)) -
            # ((sed_rest_ref < range) *  # Rest - Reference
            ((1 - 7 * (sed_rest_ref / range_) ** 2 +
              f1i * (sed_rest_ref / range_) ** 3 -
              f2i * (sed_rest_ref / range_) ** 5 +
              f3i * (sed_rest_ref / range_) ** 7)) +
            # ((sed_ref_ref < range) *  # Reference - References
            ((1 - 7 * (sed_ref_ref / range_) ** 2 +
              f1i * (sed_ref_ref / range_) ** 3 -
              f2i * (sed_ref_ref / range_) ** 5 +
              f3i * (sed_ref_ref / range_) ** 7))))

    hu_ref = (hu_dip + hv_dip)

    hu_temp_rest = x_irest - x_jrest

    hu_0rest = hu_temp_rest[0] * z_i
    hu_1rest = hu_temp_rest[1] * z_i2
    hu_rest = hu_0rest + hu_1rest

    hv_0rest = hu_temp_rest[0] * z_j
    hv_1rest = hu_temp_rest[1] * z_j2
    hv_rest = -(hv_0rest + hv_1rest)

    hu_rest = hu_rest + hv_rest
    # hu_0 = hu_temp_rest[0] * z_i
    # hu_1 = hu_temp_ref[1] * z_i2
    # hu_dip = hu_0 + hu_1

    # hv_0 = hu_temp_ref[0] * z_j
    # hv_1 = hu_temp_ref[1] * z_j2
    # hv_dip = -(hv_0 + hv_1)

    # h_rest1 = hu_temp_rest[0] * (z_i + z_j)
    # h_rest2 = hu_temp_rest[1] * (z_i2 + z_j2)
    # hu_rest = h_rest1+h_rest2

    # hu_temp_ref2 = x_iref - x_jref

    # hu_ref1 = hu_temp_ref2[0] * (z_i + z_j)
    # hu_ref2 = hu_temp_ref2[1] * (z_i2 + z_j2)
    # hu_ref = hu_ref1 + hu_ref2

    # perp_cgi_ke = -1 * (p_cgi_i*p_cgi_j-1)

    cgi = gi_res * (
            (hu_rest *
             (- c_o * ((-14 / range_ ** 2) + f1gi * sed_rest_rest / range_ ** 3 -
                       f2gi * sed_rest_rest ** 3 / range_ ** 5 +
                       f3gi * sed_rest_rest ** 5 / range_ ** 7))) -
            (hu_ref *
             (- c_o * ((-14 / range_ ** 2) + f1gi * sed_ref_ref / range_ ** 3 -
                       f2gi * sed_ref_ref ** 3 / range_ ** 5 +
                       f3gi * sed_ref_ref ** 5 / range_ ** 7)))
    )

    # foo = (- c_o * ((-14 / range_ ** 2) + f1gi
    #                 * sed_rest_rest / range_ ** 3 #-
    #                 -  f2gi * sed_rest_rest ** 3 / range_ ** 5
    #                       + f3gi * sed_rest_rest ** 5 / range_ ** 7
    #                )
    #       )
    #
    # bar = (- c_o * ((-14 / range_ ** 2) + f1gi * sed_ref_ref / range_ ** 3 -
    #                        f2gi * sed_ref_ref ** 3 / range_ ** 5 +
    #                        f3gi * sed_ref_ref ** 5 / range_ ** 7))

    cov = cg + ci + cgi  # gi_res *  foo  - gi_res *  bar +  hu_ref + hu_rest# hu_rest # + hu_ref
    # f = cov.sum_reduction(axis=0)

    b = Vi(24, 1)

    f = cov.solve(b)

    s = f(dipsref, dipsref,
          z, z2, z, z2,
          # perp_cg, perp_cg,
          np.ones((1, 1)) * 35 / 2,
          np.ones((1, 1)) * 21 / 4,
          np.ones((1, 1)) * 1e-3,
          np.ones((1, 1)) * 105 / 4,
          np.ones((1, 1)) * 35 / 2,
          np.ones((1, 1)) * 21 / 4,
          perp_m, perp_m,
          np.ones((1, 1)) * c_o_T,
          np.ones((1, 1)) * a_T,
          dipsrest, dipsrest,
          np.ones((1, 1)) * 1,
          np.ones((1, 1)) * 35 / 4,
          np.ones((1, 1)) * 7 / 2,
          np.ones((1, 1)) * 3 / 4,
          np.ones((1, 1)) * 1,
          np.ones((1, 1)) * 105 / 4,

          # perp_cgi, perp_cgi
          )

    print(s)


@pytest.mark.skip('DEP test')
def test_pykeops():
    layer1 = np.array([[0, 0], [2, 0]])
    layer2 = np.array([[0, 2], [2, 2]])

    layer1 = np.array([[0, 0], [2, 0], [3, 0], [4, 0]])
    layer2 = np.array([[0, 2], [2, 2], [3, 3]])

    number_of_layer = 2
    number_of_points_per_surface = np.array([layer1.shape[0], layer2.shape[0]])

    def set_rest_ref_matrix(number_of_points_per_surface):
        ref_layer_points = np.repeat(np.stack([layer1[-1], layer2[-1]], axis=0),
                                     repeats=number_of_points_per_surface - 1,
                                     axis=0)
        rest_layer_points = np.concatenate([layer1[0:-1], layer2[0:-1]], axis=0)
        return ref_layer_points, rest_layer_points

    ref_layer_points, rest_layer_points = set_rest_ref_matrix(
        number_of_points_per_surface)
    ## defining the dips position
    G_1 = np.array([[0., 6.], [2., 13.]])

    G_1_x = 1
    G_1_y = 1

    G_1_tiled = np.tile(G_1, [2, 1])

    dipsref = np.vstack((G_1_tiled, ref_layer_points))
    dipsrest = np.vstack((G_1_tiled, rest_layer_points))
    n_dips = G_1.shape[0]
    n_points = dipsref.shape[0]
    dipsref, dipsrest, n_dips, n_points

    z = np.zeros((n_points, 1))
    z[:n_dips] = 1
    # z[n_dips*2:n_dips*3] = 1
    z

    z2 = np.zeros((n_points, 1))
    z2[n_dips:n_dips * 2] = 1
    z2

    perp_m = np.zeros((n_points, 2))
    perp_m[:n_dips, 0] = 1
    perp_m[n_dips:2 * n_dips, 1] = 1
    perp_ma = perp_m @ perp_m.T
    perp_m, perp_ma

    perp_cgi = np.zeros((n_points, 2))
    perp_cgi[:n_dips * 2, 1] = 1
    perp_cgi[n_dips * 2:, 0] = 1
    perp_cgi_m = -1 * (perp_cgi @ perp_cgi.T - 1)
    perp_cgi, perp_cgi_m

    perp_cg = np.zeros((n_points, 1))
    perp_cg[:n_dips * 2, 0] = 1
    # perp_m[n_dips:2*n_dips, 1] = 1
    perp_cg_ma = perp_cg @ perp_cg.T
    perp_cg, perp_cg_ma

    perp_ci = np.zeros((n_points, 1))
    perp_ci[n_dips * 2:, 0] = 1
    # perp_m[n_dips:2*n_dips, 1] = 1
    perp_ci_ma = perp_ci @ perp_ci.T
    perp_ci_ma

    a_T = 5
    c_o_T = a_T ** 2 / 14 / 3
    a_T, c_o_T

    # ==================================

    # x_i = Vi(0, 2)
    # x_j = Vj(1, 2)
    x_iref = Vi(dipsref)
    x_jref = Vj(dipsref)

    z_i = Vi(z)  # Used for selecting hu1
    z_i2 = Vi(z2)  # Used for selecting hu2
    z_j = Vj(z)  # Used for selecting hv1
    z_j2 = Vj(z2)  # Used for selecting hv2
    # p_cg_i = Vi(6,1)
    # p_cg_j= Vj(7,1)
    nugget = Pm(np.ones((1, 1)) * 1e-3)  # Avoid a/0
    f1g = Pm(np.ones((1, 1)) * 105 / 4)
    f2g = Pm(np.ones((1, 1)) * 35 / 2)
    f3g = Pm(np.ones((1, 1)) * 21 / 4)
    p_i = Vi(perp_m)  # Select matrix
    p_j2 = Vj(perp_m)  # Select matrix
    c_o = Pm(np.ones((1, 1)) * c_o_T)
    range_ = Pm(np.ones((1, 1)) * a_T, )

    # ===============

    x_irest = Vi(dipsrest)
    x_jrest = Vj(dipsrest)
    i_res = Pm(np.ones((1, 1)) * 1)
    f1i = Pm(np.ones((1, 1)) * 35 / 4)
    f2i = Pm(np.ones((1, 1)) * 7 / 2)
    f3i = Pm(np.ones((1, 1)) * 3 / 421)
    # x_idip = Vi(0, 2)
    # x_jref = Vj(1, 2)
    # x_jrest = Vj(2, 2)
    # z_i = Vi(3, 1)
    # z_i2 = Vi(4, 1)

    # ===============

    gi_res = Pm(np.ones((1, 1)) * 1)
    f1gi = Pm(35 / 4)
    f2gi = Pm(7 / 2)
    f3gi = Pm(3 / 4)
    p_cg_i = Vi(perp_cg)
    p_cg_j = Vj(perp_cg)
    p_cgi_i = Vi(perp_cgi)
    p_cgi_j = Vj(perp_cgi)

    sed_ref_ref = x_iref.sqdist(x_jref).sqrt()
    sed_rest_rest = x_irest.sqdist(x_jrest).sqrt()
    sed_ref_rest = x_iref.sqdist(x_jrest).sqrt()
    sed_rest_ref = x_irest.sqdist(x_jref).sqrt()
    # huke = x_i - x_j

    hu_temp_ref = x_iref - x_jref

    hu_0 = hu_temp_ref[0] * z_i
    hu_1 = hu_temp_ref[1] * z_i2
    hu_dip = hu_0 + hu_1

    hv_0 = hu_temp_ref[0] * z_j
    hv_1 = hu_temp_ref[1] * z_j2
    hv_dip = -(hv_0 + hv_1)

    perp_cg_m = p_cg_i * p_cg_j

    # hu1 = huke[0] * z_i
    # hv1 = - huke[0] * z_j
    # hu1.ranges = ranges_ij0
    # hu2 = huke[1] * z_i2
    # hv2 = - huke[1] * z_j2
    # hu2.ranges = ranges_ij1

    # hu = hu1+hu2
    # hv = hv1+hv2
    t1 = (hu_dip * hv_dip) / (sed_ref_ref ** 2 + nugget)

    t2 = (-c_o * ((-14 / range_ ** 2) + f1g * sed_ref_ref / range_ ** 3 -
                  f2g * sed_ref_ref ** 3 / range_ ** 5 +
                  f3g * sed_ref_ref ** 5 / range_ ** 7)
          ) + (
                 c_o * 7 * (
                     9 * sed_ref_ref ** 5 - 20 * range_ ** 2 * sed_ref_ref ** 3 +
                     15 * range_ ** 4 * sed_ref_ref - 4 * range_ ** 5) / (
                             2 * range_ ** 7))

    t3 = (p_i * p_j2).sum(-1) * c_o * (
                (-14 / range_ ** 2) + f1g * sed_ref_ref / range_ ** 3 -
                f2g * sed_ref_ref ** 3 / range_ ** 5 + f3g * sed_ref_ref ** 5 / range_ ** 7)

    cg = (t1 * t2 - t3) * perp_cg_m

    ci = (c_o * i_res * (
        # (sed_rest_rest < range) *  # Rest - Rest Covariances Matrix
            (1 - 7 * (sed_rest_rest / range_) ** 2 +
             f1i * (sed_rest_rest / range_) ** 3 -
             f2i * (sed_rest_rest / range_) ** 5 +
             f3i * (sed_rest_rest / range_) ** 7) -
            # ((sed_ref_rest < range) *  # Reference - Rest
            ((1 - 7 * (sed_ref_rest / range_) ** 2 +
              f1i * (sed_ref_rest / range_) ** 3 -
              f2i * (sed_ref_rest / range_) ** 5 +
              f3i * (sed_ref_rest / range_) ** 7)) -
            # ((sed_rest_ref < range) *  # Rest - Reference
            ((1 - 7 * (sed_rest_ref / range_) ** 2 +
              f1i * (sed_rest_ref / range_) ** 3 -
              f2i * (sed_rest_ref / range_) ** 5 +
              f3i * (sed_rest_ref / range_) ** 7)) +
            # ((sed_ref_ref < range) *  # Reference - References
            ((1 - 7 * (sed_ref_ref / range_) ** 2 +
              f1i * (sed_ref_ref / range_) ** 3 -
              f2i * (sed_ref_ref / range_) ** 5 +
              f3i * (sed_ref_ref / range_) ** 7))))

    hu_ref = (hu_dip + hv_dip)

    hu_temp_rest = x_irest - x_jrest

    hu_0rest = hu_temp_rest[0] * z_i
    hu_1rest = hu_temp_rest[1] * z_i2
    hu_rest = hu_0rest + hu_1rest

    hv_0rest = hu_temp_rest[0] * z_j
    hv_1rest = hu_temp_rest[1] * z_j2
    hv_rest = -(hv_0rest + hv_1rest)

    hu_rest = hu_rest + hv_rest
    # hu_0 = hu_temp_rest[0] * z_i
    # hu_1 = hu_temp_ref[1] * z_i2
    # hu_dip = hu_0 + hu_1

    # hv_0 = hu_temp_ref[0] * z_j
    # hv_1 = hu_temp_ref[1] * z_j2
    # hv_dip = -(hv_0 + hv_1)

    # h_rest1 = hu_temp_rest[0] * (z_i + z_j)
    # h_rest2 = hu_temp_rest[1] * (z_i2 + z_j2)
    # hu_rest = h_rest1+h_rest2

    # hu_temp_ref2 = x_iref - x_jref

    # hu_ref1 = hu_temp_ref2[0] * (z_i + z_j)
    # hu_ref2 = hu_temp_ref2[1] * (z_i2 + z_j2)
    # hu_ref = hu_ref1 + hu_ref2

    # perp_cgi_ke = -1 * (p_cgi_i*p_cgi_j-1)

    cgi = gi_res * (
            (hu_rest *
             (- c_o * ((-14 / range_ ** 2) + f1gi * sed_rest_rest / range_ ** 3 -
                       f2gi * sed_rest_rest ** 3 / range_ ** 5 +
                       f3gi * sed_rest_rest ** 5 / range_ ** 7))) -
            (hu_ref *
             (- c_o * ((-14 / range_ ** 2) + f1gi * sed_ref_ref / range_ ** 3 -
                       f2gi * sed_ref_ref ** 3 / range_ ** 5 +
                       f3gi * sed_ref_ref ** 5 / range_ ** 7)))
    )

    # foo = (- c_o * ((-14 / range_ ** 2) + f1gi
    #                 * sed_rest_rest / range_ ** 3 #-
    #                 -  f2gi * sed_rest_rest ** 3 / range_ ** 5
    #                       + f3gi * sed_rest_rest ** 5 / range_ ** 7
    #                )
    #       )
    #
    # bar = (- c_o * ((-14 / range_ ** 2) + f1gi * sed_ref_ref / range_ ** 3 -
    #                        f2gi * sed_ref_ref ** 3 / range_ ** 5 +
    #                        f3gi * sed_ref_ref ** 5 / range_ ** 7))

    cov = cg + ci + cgi  # gi_res *  foo  - gi_res *  bar +  hu_ref + hu_rest# hu_rest # + hu_ref
    # f = cov.sum_reduction(axis=0)

    b = Vi(24, 1)

    s = cov.sum_reduction(axis=0)
    # f = cov.solve(b)

    # s = f(dipsref, dipsref,
    #   z, z2, z, z2,
    #   # perp_cg, perp_cg,
    #   np.ones((1, 1)) * 35 / 2,
    #   np.ones((1, 1)) * 21 / 4,
    #   np.ones((1, 1)) * 1e-3,
    #   np.ones((1, 1)) * 105 / 4,
    #   np.ones((1, 1)) * 35 / 2,
    #   np.ones((1, 1)) * 21 / 4,
    #   perp_m, perp_m,
    #   np.ones((1, 1)) * c_o_T,
    #   np.ones((1, 1)) * a_T,
    #   dipsrest, dipsrest,
    #   np.ones((1, 1)) * 1,
    #   np.ones((1, 1)) * 35 / 4,
    #   np.ones((1, 1)) * 7 / 2,
    #   np.ones((1, 1)) * 3 / 4,
    #   np.ones((1, 1)) * 1,
    #   np.ones((1, 1)) * 105 / 4,
    #
    #  # perp_cgi, perp_cgi
    #   )

    print(s)
