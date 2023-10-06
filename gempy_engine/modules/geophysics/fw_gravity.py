from ...core.backend_tensor import BackendTensor


def compute_gravity(tz: BackendTensor.t, densities: BackendTensor.t) -> BackendTensor.t:
    grav = tz * densities
    return grav