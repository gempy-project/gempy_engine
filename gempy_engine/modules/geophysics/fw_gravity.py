from ...core.data.geophysics_input import GeophysicsInput
from ...core.data.interp_output import InterpOutput
from ...core.backend_tensor import BackendTensor


def map_densities_to_ids(ids_gravity_grid, densities):
    pass


def compute_gravity(geophysics_input: GeophysicsInput, output: InterpOutput) -> BackendTensor.t:
    tz = geophysics_input.tz
    # TODO: Add geophysics grid like sections
    # TODO: Add mapping function
    ids_gravity_grid = output.geophysics_grid_values
    
    densities = map_densities_to_ids(ids_gravity_grid, densities=geophysics_input.densities)
    
    grav = tz * densities
    return grav