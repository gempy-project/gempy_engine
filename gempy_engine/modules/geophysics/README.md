
# GemPy Geophysics Module

This module implements forward geophysical computations for 3D geological models in GemPy v3. Currently, gravity modeling is fully implemented, with magnetic modeling in development.

## Overview

The geophysics module in GemPy v3 provides an integrated framework for computing geophysical responses directly from 3D geological models. The implementation follows a modular architecture that separates:

1. **Grid management** - Defines observation points and voxelized computational grids
2. **Forward modeling** - Computes geophysical responses from geological models
3. **Physical property mapping** - Maps rock properties (density, susceptibility) to geological units

## Architecture

### General Workflow

The geophysics computation workflow integrates seamlessly with GemPy's interpolation pipeline:

```
Geological Model → Octree Interpolation → Property Mapping → Forward Calculation → Geophysical Response
```

**Conceptual diagram:**

```
    ┌─────────────────┐
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
    │ Geophysical    │      │ Property mapping   │
    │ Gradient       │      │ ID → property      │
    │ Kernel         │      │ [1→ρ, 2→χ]         │
    └────────┬───────┘      └────────┬───────────┘
             │                       │
             └───────┬───────────────┘
                     ↓
            ┌────────────────┐
            │ Forward model  │  
            │ Response = Σ   │
            │ (property·kernel)
            └────────┬───────┘
                     ↓
            ┌────────────────┐
            │ Geophysical    │
            │ response at    │
            │ observation    │
            │ points         │
            └────────────────┘
```

Key components:

- **`GeophysicsInput`**: Data class containing physical property arrays (densities, susceptibilities) and pre-computed kernels (e.g., gravity/magnetic gradients)
- **`CenteredGrid`**: Defines the geometry of the computational grid around observation points
- **Forward modeling functions**: Compute geophysical responses by combining interpolated geological IDs with physical properties

The geophysics calculations are performed within the main `compute_model()` API function, after the octree interpolation completes but before mesh extraction.

## Gravity Implementation

### Physical Basis

The gravity forward calculation computes the vertical component of gravitational acceleration at observation points by summing contributions from voxelized density distributions in 3D space.

**Gravity anomaly geometry:**

```
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
```

### Components

#### 1. Gravity Gradient Calculation (`gravity_gradient.py`)

The `calculate_gravity_gradient()` function computes the gravitational kernel (tz component) for each voxel in the computational grid using analytical integration:

**Voxel contribution geometry:**

```
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
```

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

### Integration with GemPy Pipeline

The gravity calculation occurs in `compute_model()` after octree interpolation:

1. Octree interpolation produces `ids_geophysics_grid` at voxel centers
2. If `geophysics_input` is provided, `compute_gravity()` is called using the finest octree level's output
3. Results are stored in `Solutions.fw_gravity` (aliased as `Solutions.gravity`)

### Key Design Decisions

- **Voxelized approach**: Uses regular grid voxels for computational efficiency and GPU compatibility
- **Pre-computed kernels**: Gravity gradients are calculated once and reused across multiple density configurations
- **Backend agnostic**: Works with NumPy, PyTorch, and TensorFlow through the `BackendTensor` abstraction
- **Batch-friendly**: Supports multiple observation points (devices) in a single computation

## Magnetics Implementation (In Development)

### Physical Basis

Magnetic field modeling is more complex than gravity because it involves **vector fields** rather than scalar fields. The key physical components are:

1. **Earth's magnetic field (IGRF)**: The regional/ambient magnetic field that induces magnetization in rocks
2. **Induced magnetization**: Magnetization caused by susceptibility in Earth's field
3. **Remanent magnetization**: Permanent magnetization "frozen in" when rocks cooled/formed
4. **Total Magnetic Intensity (TMI)**: The measured quantity, representing field strength anomalies

**Magnetic anomaly geometry:**

```
    Observation point (magnetometer)
            ●  ← measures ΔT (TMI anomaly)
            ↓
    ════════════════════  Earth's surface
            
    Earth's Field B₀ (IGRF)
    ─────────────────→ (inclination I, declination D)
            │
         .-"─"-.
       .'   |   '.         Susceptible body
      /  χ, M_r  \         χ = susceptibility
     ;     ●     ;          M_r = remanent magnetization
      \         /
       '.     .'
         `-.-'
            
    Induced: M_ind = χ · B₀ / μ₀
    Total:   M_total = M_ind + M_r
    
    TMI anomaly: ΔT = B_anomaly · B̂₀
```

### Key Differences from Gravity

| Aspect                | Gravity               | Magnetics                               |
|-----------------------|-----------------------|-----------------------------------------|
| **Field type**        | Scalar (gz only)      | Vector (3 components)                   |
| **Physical property** | Density (ρ)           | Susceptibility (χ) + Remanence (Mr)     |
| **Source**            | Mass distribution     | Magnetic dipoles                        |
| **Ambient field**     | None (constant g)     | Earth's field (varies by location/time) |
| **Measurement**       | Vertical acceleration | Total field intensity                   |
| **Kernel complexity** | Single component (tz) | Full tensor (9 components)              |

### Mathematical Framework

For a voxelized magnetic body, the magnetic field anomaly at observation point **r** is:

$$\mathbf{B}(\mathbf{r}) = \frac{\mu_0}{4\pi} \sum_i V_i \nabla \nabla \left(\frac{1}{|\mathbf{r} - \mathbf{r}_i|}\right) \cdot \mathbf{M}_i$$

Where:
- **M**ᵢ = magnetization vector of voxel i (A/m)
- Vᵢ = volume of voxel i
- μ₀ = permeability of free space (4π × 10⁻⁷ H/m)
- ∇∇ = magnetic gradient tensor (3×3 matrix)

The **Total Magnetic Intensity (TMI)** anomaly is the projection onto the ambient field direction:

$$\Delta T = \mathbf{B}_{anomaly} \cdot \hat{\mathbf{B}}_0 = B_{anomaly,x} l_x + B_{anomaly,y} l_y + B_{anomaly,z} l_z$$

Where **l** = direction cosines of Earth's field (from inclination I and declination D).

### Implementation Strategy

The magnetics implementation follows the same pre-computation architecture as gravity:

#### 1. Magnetic Gradient Tensor Calculation (`magnetic_gradient.py` - planned)

```python
def calculate_magnetic_gradient_tensor(centered_grid: CenteredGrid, 
                                       igrf_field: np.ndarray,
                                       compute_tmi: bool = True) -> dict:
    """
    Compute magnetic gradient kernels for voxelized forward modeling.
    
    Args:
        centered_grid: Grid definition with observation points
        igrf_field: Earth's field vector [Bx, By, Bz] in nT or [I, D, F] format
        compute_tmi: If True, pre-compute TMI kernel; else return full tensor
    
    Returns:
        Dictionary containing:
        - 'tmi_kernel': Pre-computed TMI kernel (if compute_tmi=True)
        - 'tensor': Full 3×3 gradient tensor per voxel (if compute_tmi=False)
        - 'field_direction': Unit vector of IGRF field
        - 'inclination': Inclination angle (degrees)
        - 'declination': Declination angle (degrees)
    """
```

**Key components:**
- **IGRF field specification**: Must provide inclination (I), declination (D), and field strength (F)
- **Gradient tensor**: 9 components (∂²/∂x², ∂²/∂x∂y, etc.) computed analytically for each voxel
- **TMI kernel optimization**: Pre-project onto field direction to reduce computation (similar to gravity's tz)

#### 2. Forward Magnetic Calculation (`fw_magnetic.py` - planned)

```python
def compute_tmi(geophysics_input: GeophysicsInput, 
                root_output: InterpOutput) -> BackendTensor.t:
    """
    Compute Total Magnetic Intensity anomaly.
    
    Args:
        geophysics_input: Contains:
            - tmi_kernel: Pre-computed magnetic kernel
            - susceptibilities: Array of susceptibilities per geological unit (SI units)
            - remanence: Optional array of remanent magnetization vectors
            - igrf_params: IGRF field parameters (I, D, F)
        root_output: Interpolation output with geological IDs
    
    Returns:
        TMI anomaly at observation points (nT)
    """
```

**Workflow:**
1. Map susceptibilities (and optionally remanence) to geological IDs
2. Compute induced magnetization: **M**_ind = χ · **B**₀ / μ₀
3. Add remanent magnetization if provided: **M**_total = **M**_ind + **M**_r
4. Apply TMI kernel: TMI = Σ(magnetization · tmi_kernel)

### Handling Earth's Field: IGRF Models

The **International Geomagnetic Reference Field (IGRF)** provides the Earth's magnetic field at any location and time. Key considerations:

#### IGRF Parameters

For a given survey location and date, you need:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| **Inclination (I)** | Dip angle from horizontal | -90° to +90° |
| **Declination (D)** | Angle from true north | -180° to +180° |
| **Total Intensity (F)** | Field strength | 25,000-65,000 nT |

**Geographic variations:**
- Equator: I ≈ 0°, horizontal field
- North pole: I ≈ +90°, vertical downward
- South pole: I ≈ -90°, vertical upward
- Mid-latitudes: I ≈ ±60-70°

#### Obtaining IGRF Values

```python
# Example: Using pyIGRF or similar library
import pyigrf

# For a specific location and date
lat, lon = 37.5, -5.5  # Tharsis region, Spain
year = 2023.66  # August 2023

# Get IGRF components
D, I, H, X, Y, Z, F = pyigrf.igrf_value(lat, lon, alt=0, year=year)

# Create field vector for GemPy
igrf_field = np.array([
    F * np.cos(np.radians(I)) * np.cos(np.radians(D)),  # Bx (north)
    F * np.cos(np.radians(I)) * np.sin(np.radians(D)),  # By (east)
    F * np.sin(np.radians(I))                            # Bz (down)
])
```

**Temporal variations:**
- **Secular variation**: IGRF changes ~50-100 nT/year
- **Diurnal variation**: ±20-30 nT daily due to ionosphere
- **Magnetic storms**: Can cause variations of 100-1000 nT
- **Best practice**: Use IGRF for survey date; consider diurnal corrections for high-precision work

### Remanent Magnetization

Remanent magnetization (M_r) is the "permanent" magnetization acquired during rock formation. It can dominate over induced magnetization in some rocks:

**Types of remanence:**
- **TRM (Thermoremanent)**: Acquired during cooling through Curie temperature
- **DRM (Detrital)**: Alignment of magnetic grains during sedimentation
- **CRM (Chemical)**: Acquired during chemical alteration

**Koenigsberger ratio (Q):**
```
Q = |M_remanent| / |M_induced| = |M_r| / (χ·F/μ₀)
```

- Q < 1: Induced dominates (sedimentary rocks, most minerals)
- Q > 1: Remanent dominates (basalts, some igneous rocks)
- Q >> 1: Strong remanence (can cause negative anomalies!)

#### Normalized Source Strength (NSS) Approach

For cases with unknown remanence direction, the **Normalized Source Strength (NSS)** method provides a remanence-independent interpretation [[1]](https://www.researchgate.net/publication/258787696_Mitigating_remanent_magnetization_effects_in_magnetic_data_using_the_normalized_source_strength):

```
NSS = √(Bx² + By² + Bz²) / F₀
```

This approach:
- Removes dependence on remanence direction
- Works when Q ratios are moderate (not extreme remanence)
- Simplifies interpretation in complex remanence scenarios
- Can be implemented as an alternative forward calculation mode

### Usage Example (Planned)

```python
from gempy_engine.core.data.centered_grid import CenteredGrid
from gempy_engine.core.data.geophysics_input import GeophysicsInput
from gempy_engine.modules.geophysics.magnetic_gradient import calculate_magnetic_gradient_tensor
from gempy_engine.API.model.model_api import compute_model
import numpy as np

# Define observation points
geophysics_grid = CenteredGrid(
    centers=np.array([[0.25, 0.5, 0.75], [0.75, 0.5, 0.75]]),
    resolution=np.array([10, 10, 15]),
    radius=np.array([1, 1, 1])
)

# IGRF field for survey location and date
# Example: Spain, August 2023
igrf_params = {
    'inclination': 54.2,   # degrees
    'declination': -2.1,   # degrees
    'intensity': 44500     # nT
}

# Calculate magnetic gradient tensor
mag_gradient = calculate_magnetic_gradient_tensor(
    geophysics_grid, 
    igrf_params,
    compute_tmi=True  # Pre-compute TMI kernel
)

# Define physical properties
geophysics_input = GeophysicsInput(
    mag_kernel=mag_gradient['tmi_kernel'],
    susceptibilities=np.array([0.001, 0.05, 0.0001]),  # SI units
    remanence=None,  # Optional: np.array([[Mx, My, Mz], ...]) per unit
    igrf_params=igrf_params
)

# Compute model with magnetics
interpolation_input.grid.geophysics_grid = geophysics_grid
solutions = compute_model(
    interpolation_input, 
    options, 
    structure, 
    geophysics_input=geophysics_input
)

# Access results
tmi_anomaly = solutions.tmi  # Total Magnetic Intensity anomaly in nT
```

### Implementation Roadmap

1. **Phase 1**: Basic TMI forward calculation
   - Implement magnetic gradient tensor for rectangular prisms
   - Pre-compute TMI kernel (similar to gravity tz)
   - Support induced magnetization only (no remanence)
   - IGRF parameters as user input

2. **Phase 2**: Remanent magnetization
   - Add remanence vectors per geological unit
   - Implement Q-ratio calculations
   - Validation against analytical sphere solution

3. **Phase 3**: Advanced features
   - Normalized Source Strength (NSS) calculation
   - Full tensor components (Bx, By, Bz separately)
   - Magnetic gradient components (for gradiometry surveys)
   - IGRF integration via pyIGRF

4. **Phase 4**: Optimization
   - GPU acceleration for large models
   - Adaptive grids for near-source accuracy
   - Caching strategies for parametric studies

### Key References

- Blakely, R.J. (1995). *Potential Theory in Gravity and Magnetic Applications*. Cambridge University Press.
- Reford, M.S. (1980). "Magnetic method." *Geophysics*, 45(11), 1640-1658.
- Li, Y. & Oldenburg, D.W. (2003). "Fast inversion of large-scale magnetic data using wavelet transforms." *Geophysical Journal International*, 152(2), 251-265.
- Shearer, S.E. (2005). "Three-dimensional inversion of magnetic data in the presence of remanent magnetization." PhD thesis, Colorado School of Mines.

## Testing

Comprehensive tests are provided in `test_geophysics.py`, including:
- Gravity forward calculation validation
- Integration with octree interpolation
- Multi-device scenarios
- Visualization with PyVista (optional)

### Analytical Benchmarks

`test_gravity_benchmark.py` provides validation against analytical solutions:

1. **Sphere benchmark**: Validates numerical accuracy against the analytical solution for a homogeneous sphere
   - Tests multiple observation distances
   - Quantifies voxelization errors
   - Validates physical decay with distance

2. **Line profile symmetry**: Tests spatial consistency and physical behavior
   - Verifies symmetry of response
   - Confirms peak location
   - Validates decay away from anomaly

`test_magnetic_benchmark.py` (planned) will include:
- Magnetic dipole analytical solution
- Sphere with induced magnetization
- Remanence effects validation
- Comparison with published test cases

These benchmarks ensure the implementation is both mathematically correct and physically plausible.

## Future Development

### Magnetics Priorities

1. ✅ Gravity implementation complete
2. 🔄 Magnetics TMI forward calculation (in progress)
3. ⏳ Remanent magnetization support
4. ⏳ Joint gravity-magnetic inversion utilities
5. ⏳ Integration with geophysical survey data formats

### Extended Capabilities

- **Tensor gradiometry**: Full gradient tensor for airborne surveys
- **Temporal effects**: Diurnal corrections and storm filtering
- **Topographic effects**: Terrain corrections for both methods
- **Noise models**: Realistic measurement uncertainties
- **Inversion tools**: Direct integration with GemPy's inverse modeling

```