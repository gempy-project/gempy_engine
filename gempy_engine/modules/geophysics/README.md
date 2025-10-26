# GemPy Geophysics Module

This module implements forward geophysical computations for 3D geological models in GemPy v3. Currently, gravity modeling is fully implemented, with magnetic modeling planned for future development.

## Overview

The geophysics module in GemPy v3 provides an integrated framework for computing geophysical responses directly from 3D geological models. The implementation follows a modular architecture that separates:

1. **Grid management** - Defines observation points and voxelized computational grids
2. **Forward modeling** - Computes geophysical responses from geological models
3. **Physical property mapping** - Maps rock properties (density, susceptibility) to geological units

## Architecture

### General Workflow

The geophysics computation workflow integrates seamlessly with GemPy's interpolation pipeline:Key components:

**Conceptual diagram:**
┌─────────────────┐
│ Observation     │ ← Device 1: xyz position
│ Points (n)      │ ← Device 2: xyz position
└────────┬────────┘ ← Device n: xyz position
│
├─────────────────────────┐
↓                         ↓
┌────────────────┐      ┌────────────────────┐
│ Voxel Grid     │      │ Geological Model   │
│ around each    │      │ (GemPy interpolate)│
│ observation    │      │                    │
│ point          │      │  ┌──┬──┬──┐        │
│                │      │  │ 1│ 2│ 3│ IDs    │
│  ┌─┬─┬─┐       │      │  ├──┼──┼──┤        │
│  │ │ │ │       │      │  │ 2│ 2│ 3│        │
│  ├─┼─┼─┤       │      │  └──┴──┴──┘        │
│  │ │●│ │       │      └────────────────────┘
│  ├─┼─┼─┤       │                 │
│  │ │ │ │       │                 │
│  └─┴─┴─┘       │                 │
└────────┬───────┘                 │
│                         │
↓                         ↓
┌────────────────┐      ┌────────────────────┐
│ Gravity        │      │ Density mapping    │
│ Gradient (tz)  │      │ ID → ρ             │
│ per voxel      │      │ [1→2.67, 2→3.3]    │
└────────┬───────┘      └────────┬───────────┘
│                       │
└───────┬───────────────┘
↓
┌────────────────┐
│   gz = Σ(ρ·tz) │  Forward calculation
└────────┬───────┘
↓
┌────────────────┐
│ Gravity at     │
│ observation    │
│ points         │
└────────────────┘

- **`GeophysicsInput`**: Data class containing physical property arrays (densities, susceptibilities) and pre-computed kernels (e.g., gravity gradients)
- **`CenteredGrid`**: Defines the geometry of the computational grid around observation points
- **Forward modeling functions**: Compute geophysical responses by combining interpolated geological IDs with physical properties

The geophysics calculations are performed within the main `compute_model()` API function, after the octree interpolation completes but before mesh extraction.

## Gravity Implementation

### Physical Basis

The gravity forward calculation computes the vertical component of gravitational acceleration at observation points by summing contributions from voxelized density distributions in 3D space.


Key components:

- **`GeophysicsInput`**: Data class containing physical property arrays (densities, susceptibilities) and pre-computed kernels (e.g., gravity gradients)
- **`CenteredGrid`**: Defines the geometry of the computational grid around observation points
- **Forward modeling functions**: Compute geophysical responses by combining interpolated geological IDs with physical properties

The geophysics calculations are performed within the main `compute_model()` API function, after the octree interpolation completes but before mesh extraction.

## Gravity Implementation
Observation point (gravimeter/sensor)
        ●  ← measures gz (vertical component)
        |
        | gz = Σ(density_i × tz_i)
        ↓
════════════════════  Earth's surface
        |
     .-"─"-.
   .'   |   '.         Density anomaly
  /     |     \        (geological body)
 ;    Δρ > 0   ;       
  \     |     /
   '.   |   .'
     `-─-'
        |
Background density ρ₀

Observation point P (x₀, y₀, z₀)
         ●
        ╱│╲
       ╱ │ ╲  gz contributions from voxel corners
      ╱  │  ╲
     ╱   │   ╲
    ╱    ↓    ╲
┌─────────────────────┐
│    ·       ·       · │  ← Voxel with 8 corners
│  ·    ┌───┐    ·    │     Analytical integration
│    ·  │ Δρ│  ·      │     over rectangular prism
│  ·    └───┘    ·    │
│    ·       ·       · │
└─────────────────────┘

tz = G·Σ[sign_i·(x log(y+r) + y log(x+r) - z arctan(xy/zr))]

### Physical Basis

The gravity forward calculation computes the vertical component of gravitational acceleration at observation points by summing contributions from voxelized density distributions in 3D space.

**Gravity anomaly geometry:**
**Input:**
- `CenteredGrid`: Defines observation point centers, voxel resolutions, and spatial extents

**Output:**
- `tz`: Array of vertical gravity gradient contributions (shape: n_voxels)

**Key parameters:**
- `ugal`: If `True`, uses units of cm³·g⁻¹·s⁻² (10⁻³ mGal), suitable for microgravity surveys. If `False`, uses SI units (m³·kg⁻¹·s⁻²)
- Gravitational constant: G = 6.674e-3 (ugal) or 6.674e-11 (SI)

**Implementation details:**
- Each voxel's contribution is computed by analytical integration over its 8 corners
- Uses the formula: `-G * [x*log(y+r) + y*log(x+r) - z*arctan(xy/zr)]`
- Corner contributions are summed with appropriate sign factors ([1, -1, -1, 1, -1, 1, 1, -1])

#### 2. Forward Gravity Calculation (`fw_gravity.py`)

The `compute_gravity()` function combines the geological model (as interpolated IDs) with density values:

**Input:**
- `GeophysicsInput`: Contains `tz` (gravity gradient kernel) and `densities` array
- `InterpOutput`: Contains `ids_geophysics_grid` (interpolated geological unit IDs at voxel centers)

**Output:**
- `grav`: Gravity response at observation points (shape: n_devices)

**Workflow:**
1. **Map densities to geological IDs**: Using `map_densities_to_ids_basic()`, each voxel is assigned a density based on its geological unit ID
2. **Reshape for multiple devices**: Supports batch computation across multiple observation points
3. **Weighted sum**: Compute `grav = Σ(density_i * tz_i)` for each device

**Property Mapping:**
- `map_densities_to_ids_basic()`: Direct indexing (ID → density), assumes 1-based IDs
- `map_densities_to_ids_fancy()`: Interpolation-based mapping (planned for future use)

### Usage Example

```python
from gempy_engine.core.data.centered_grid import CenteredGrid
from gempy_engine.core.data.geophysics_input import GeophysicsInput
from gempy_engine.modules.geophysics.gravity_gradient import calculate_gravity_gradient
from gempy_engine.API.model.model_api import compute_model
import numpy as np

# Define observation points and voxel grid
geophysics_grid = CenteredGrid(
    centers=np.array([[0.25, 0.5, 0.75], [0.75, 0.5, 0.75]]),  # 2 observation points
    resolution=np.array([10, 10, 15]),  # voxel resolution per device
    radius=np.array([1, 1, 1])  # spatial extent
)

# Calculate gravity gradient kernel
gravity_gradient = calculate_gravity_gradient(geophysics_grid, ugal=True)

# Define physical properties
geophysics_input = GeophysicsInput(
    tz=gravity_gradient,
    densities=np.array([2.67, 3.3, 2.4])  # g/cm³ for each geological unit
)

# Compute model with gravity
interpolation_input.grid.geophysics_grid = geophysics_grid
solutions = compute_model(
    interpolation_input, 
    options, 
    structure, 
    geophysics_input=geophysics_input
)

# Access results
gravity_response = solutions.gravity  # or solutions.fw_gravity
```

