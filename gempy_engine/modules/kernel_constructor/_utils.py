from gempy_engine.core.backend_tensor import BackendTensor

tfnp = BackendTensor.tfnp
tensor_types = BackendTensor.tensor_types


def squared_euclidean_distances(x_1: tensor_types,
                                x_2: tensor_types):
    """
    Compute the euclidian distances in 3D between all the points in x_1 and x_2

    Args:
        x_1 (theano.tensor.matrix): shape n_points x number dimension
        x_2 (theano.tensor.matrix): shape n_points x number dimension

    Returns:
        theano.tensor.matrix: Distancse matrix. shape n_points x n_points
    """

    keops = False
    if keops is False:
        x_1 = x_1[:, None, :]
        x_2 = x_2[None, :, :]
    else:
        from pykeops.numpy import LazyTensor as LazyTensor_np

        x_1 = LazyTensor_np(x_1[:, None, :])
        x_2 = LazyTensor_np(x_2[None, :, :])
    sqd = tfnp.sqrt(tfnp.reduce_sum(((x_1 - x_2) ** 2), -1))

    return sqd


def tensor_transpose(tensor):
    return tfnp.transpose(tensor)
