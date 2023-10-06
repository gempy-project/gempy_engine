import numpy as np
import torch
from torch import Tensor

from ...core.data.geophysics_input import GeophysicsInput
from ...core.data.interp_output import InterpOutput
from ...core.backend_tensor import BackendTensor


# ! Torch implementation
def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indicies = torch.clamp(indicies, 0, len(m) - 1)
    return m[indicies] * x + b[indicies]


# ? This seems quite a universal mapper. Probably I should move it somewhere else
def map_densities_to_ids(ids_gravity_grid, densities):
    # Check if range of ids_gravity_grid equals the number of densities
    if int(ids_gravity_grid.max() - ids_gravity_grid.min() + 1.05) != len(densities):
        raise ValueError("The range of ids_gravity_grid must be equal to the number of densities.")

    # Check if number of densities is at least 2 for interpolation
    if len(densities) < 2:
        raise ValueError("At least two densities are required for interpolation.")

    # Create an array of IDs that map to the given densities
    density_ids = np.linspace(ids_gravity_grid.min(), ids_gravity_grid.max(), len(densities))

    # Perform linear interpolation to map densities to ids_gravity_grid
    interpolated_densities = np.interp(ids_gravity_grid, density_ids, densities)

    return interpolated_densities


def compute_gravity(geophysics_input: GeophysicsInput, root_ouput: InterpOutput) -> BackendTensor.t:
    tz = geophysics_input.tz
    # TODO: Add geophysics grid like sections
    # TODO: Add mapping function
    ids_gravity_grid = root_ouput.geophysics_grid_values
    
    densities = map_densities_to_ids(ids_gravity_grid, densities=geophysics_input.densities)
    n_devices = densities.shape[0] // tz.shape[0]
    
    tz = tz.reshape(1, -1)
    densities = densities.reshape(n_devices, -1)
    
    grav = BackendTensor.t.sum(tz * densities, axis=1)
    return grav