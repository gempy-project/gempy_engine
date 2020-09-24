from gempy_engine.data_structures.private_structures import OrientationsGradients
from gempy_engine.data_structures.public_structures import OrientationsInput
from gempy_engine.systems.generators import tfnp


def deg2rad(degree_matrix):
    return degree_matrix * tfnp.constant(0.0174533)


def dip_to_gradients(ori: OrientationsInput):
    dip_angles_ = ori.dip
    azimuth_ = ori.azimuth
    polarity_ = ori.polarity

    gx = tfnp.sin(deg2rad(dip_angles_)) * tfnp.sin(deg2rad(azimuth_)) * polarity_
    gy = tfnp.sin(deg2rad(dip_angles_)) * tfnp.cos(deg2rad(azimuth_)) * polarity_
    gz = tfnp.cos(deg2rad(dip_angles_)) * polarity_

    return OrientationsGradients(gx, gy, gz)