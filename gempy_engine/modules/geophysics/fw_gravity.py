import numpy as np

from ...core.data.geophysics_input import GeophysicsInput
from ...core.data.interp_output import InterpOutput
from ...core.backend_tensor import BackendTensor


# ? This seems quite a universal mapper. Probably I should move it somewhere else
def map_densities_to_ids_fancy(ids_gravity_grid, densities):
    raise NotImplementedError("This function is not implemented yet. For this to work the ids have to be better passed here")
    # Check if range of ids_gravity_grid equals the number of densities
    if int(ids_gravity_grid.max() - ids_gravity_grid.min() + 1.05) != len(densities):
        # TODO: Here I need to have the actual range of formations ids
        raise ValueError("The range of ids_gravity_grid must be equal to the number of densities.")

    # Check if number of densities is at least 2 for interpolation
    if len(densities) < 2:
        raise ValueError("At least two densities are required for interpolation.")

    # Create an array of IDs that map to the given densities
    density_ids = np.linspace(ids_gravity_grid.min(), ids_gravity_grid.max(), len(densities))

    # Perform linear interpolation to map densities to ids_gravity_grid
    interpolated_densities = np.interp(ids_gravity_grid, density_ids, densities)

    return interpolated_densities


def map_densities_to_ids_basic(ids_gravity_grid, densities):
    return densities[ids_gravity_grid - 1]


def compute_gravity(geophysics_input: GeophysicsInput, root_ouput: InterpOutput) -> BackendTensor.t:
    tz = geophysics_input.tz
    densities = map_densities_to_ids_basic(
        ids_gravity_grid=root_ouput.ids_geophysics_grid,
        densities=BackendTensor.t.array(geophysics_input.densities)
    )
    
    n_devices = densities.shape[0] // tz.shape[0]
    
    tz = tz.reshape(1, -1)
    densities = densities.reshape(n_devices, -1)
    
    grav = BackendTensor.t.sum(densities * tz, axis=1)
    return grav