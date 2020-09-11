from gempy_engine.data_structures.public_structures import OrientationsInput


def test_orientation(moureze):
    _, ori = moureze
    ori_t = OrientationsInput(
        ori[['X', 'Y', 'Z']],
        dip_gradients=ori[['G_x', 'G_y', 'G_z']])
    print(ori_t)

    return ori_t