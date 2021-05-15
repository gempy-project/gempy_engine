from gempy_engine.integrations.interp_single.interp_single_interface import interpolate_single_scalar


def interpolate_model(all_the_input):

    # TODO: [ ] Looping scalars
    s = interpolate_single_scalar

    # --------------------
    # The following operations are applied on the lith block:

    # This should happen only on the leaf of an octree
    # TODO: [ ] Dual contouring. This method only make one vertex per voxel. It is possible to make water tight surfaces?

    # ---------------------
    # TODO: [ ] Gravity

    # TODO: [ ] Magnetics
