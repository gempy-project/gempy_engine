# Magnetics Implementation Plan (GemPy v3)

This document breaks down the tasks required to implement magnetics in GemPy v3 in the same architectural style and workflow used for gravity. It is intended as a developer roadmap, with explicit milestones, APIs, unit conventions, tests, and acceptance criteria.

## Guiding Principles

- Mirror the gravity pipeline: precompute kernel(s) for a CenteredGrid and combine with mapped properties per voxel.
- Keep BackendTensor abstraction, batching across devices, and grid semantics consistent with gravity.
- Start with the simplest useful feature (induced-only TMI) and iterate.
- Provide clear unit handling (SI vs nT) and numerical stability.

---

## Phase 1 — Induced-only TMI forward modeling (MVP)

Goal: Compute Total Magnetic Intensity (TMI) anomalies for induced magnetization only, using a precomputed voxel kernel analogous to gravity's tz.

### 1. Data structures and API surface

- Extend/confirm GeophysicsInput to carry magnetic inputs:
  - `mag_kernel: BackendTensor.t` (precomputed, per-voxel kernel for TMI projection)
  - `susceptibilities: np.ndarray | BackendTensor.t` (per geologic unit, SI)
  - `igrf_params: dict` with fields: `inclination` (deg), `declination` (deg), `intensity` (nT)
  - Optional placeholders (not used in Phase 1): `remanence` (None)
- Solutions output additions:
  - `Solutions.fw_magnetics` or `Solutions.tmi` (n_devices,) in nT
  - Alias similar to gravity: `solutions.tmi` convenience accessor
- No changes to gravity code paths.

Acceptance criteria:
- The new fields are optional and do not affect existing gravity workflows.
- When `geophysics_input.mag_kernel` and `susceptibilities` are provided, `compute_model` returns `solutions.tmi`.

### 2. Magnetic kernel computation module

Create `gempy_engine/modules/geophysics/magnetic_gradient.py` with:

```python
from gempy_engine.core.data.centered_grid import CenteredGrid
import numpy as np

# public API

def calculate_magnetic_gradient_tensor(centered_grid: CenteredGrid,
                                       igrf_params: dict,
                                       compute_tmi: bool = True,
                                       units_nT: bool = True):
    """
    Compute magnetic kernels for rectangular voxel prisms around each device.

    Returns dict with keys depending on compute_tmi:
      - if True: {
          'tmi_kernel': np.ndarray (n_voxels_per_device,),
          'field_direction': np.ndarray (3,),
          'inclination': float,
          'declination': float,
          'intensity': float
        }
      - if False: {
          'tensor': np.ndarray (n_voxels_per_device, 3, 3),
          'field_direction': np.ndarray (3,),
          'inclination': float,
          'declination': float,
          'intensity': float
        }
    """
    ...
```

Implementation tasks:
- Implement direction cosines from I, D: `l = [cos I cos D, cos I sin D, sin I]` (I positive downward)
- Implement analytical rectangular-prism magnetic field kernel via second derivatives of 1/r (see Blakely 1995). For MVP, directly pre-project into TMI using `B_anom · l` so only one scalar kernel per voxel is stored, analogous to gravity tz.
- Use μ0 = 4π×10⁻⁷ H/m. Handle units: produce output in nT when `units_nT=True`.
- Numerical stability: add small eps in logs/atan2 to avoid singularities at voxel corners.

Acceptance criteria:
- For a single device, returns a 1D kernel with length equal to voxels per device.
- Re-usable across devices by repeating per-device kernel as with gravity tz.
- Unit tests validate sign conventions and decay with distance.

### 3. Forward magnetic calculation

Create `gempy_engine/modules/geophysics/fw_magnetic.py` with:

```python
# Note: illustrative snippet for API shape; actual module will import from gempy_engine.core
import numpy as np
# from gempy_engine.core.data.geophysics_input import GeophysicsInput
# from gempy_engine.core.data.interp_output import InterpOutput
# from gempy_engine.core.backend_tensor import BackendTensor

# map susceptibilities to voxel IDs (1-based like densities)

def map_susceptibilities_to_ids_basic(ids_geophysics_grid, susceptibilities):
    return susceptibilities[ids_geophysics_grid - 1]


def compute_tmi(geophysics_input, root_output):
    """Compute induced-only TMI anomalies (nT)."""
    # precomputed kernel already includes projection along field direction
    mag_kernel = geophysics_input.mag_kernel  # shape (n_voxels_per_device,)

    # property mapping
    chi = map_susceptibilities_to_ids_basic(
        ids_geophysics_grid=root_output.ids_geophysics_grid,
        susceptibilities=np.asarray(geophysics_input.susceptibilities)
    )

    # induced magnetization magnitude factor embedded in kernel:
    # Option A (recommended for MVP): incorporate F/μ0 factor into kernel during precomputation.
    # Then forward model reduces to sum(chi * mag_kernel).

    n_devices = chi.shape[0] // mag_kernel.shape[0]
    mag_kernel = mag_kernel.reshape(1, -1)
    chi = chi.reshape(n_devices, -1)

    tmi = np.sum(chi * mag_kernel, axis=1)
    return tmi
```

Acceptance criteria:
- Mirrors `compute_gravity` structure: mapping, reshape by devices, weighted sum.
- Works with BackendTensor across supported backends.

### 4. Integration into compute_model

Tasks:
- Modify `API/model/model_api.compute_model` to:
  - After interpolation at finest octree level, if `geophysics_input.mag_kernel` exists, call `compute_tmi()` and store into `Solutions.fw_magnetics` and alias `Solutions.tmi`.
- Ensure gravity path remains unchanged.

Acceptance criteria:
- Providing both gravity and magnetics inputs yields both outputs in Solutions.

### 5. Units and constants

Tasks:
- Define constants locally in magnetic modules:
  - μ0 = 4π×10⁻⁷ H/m
- Default to nT output for TMI (common in exploration). Document clearly in docstrings.

Acceptance criteria:
- Numerical values match expected scales (tens to thousands of nT for typical bodies and F=25–65 μT).

### 6. Tests for Phase 1

- Unit tests under `tests/test_common/test_modules/`:
  1) Kernel sanity tests:
     - Increasing distance reduces magnitude.
     - Symmetry with respect to voxel grid for a centered anomaly.
  2) Induced-only sphere benchmark:
     - Compare numerical TMI over a magnetized sphere (induced) against analytical approximation at points above the center.
     - Accept ~10–20% error due to voxelization, akin to gravity benchmark.
  3) API integration test:
     - Similar to `test_geophysics.test_gravity`, confirm `solutions.tmi` shape/values for a synthetic case.

Acceptance criteria:
- Tests pass on CPU NumPy backend; gravity tests remain green.

---

## Phase 2 — Remanent magnetization support

Goal: Include optional remanent magnetization vectors per geologic unit and combine with induced magnetization.

### 1. Data/API additions
- `GeophysicsInput.remanence: Optional[np.ndarray]` with shape (n_units, 3) in A/m or in equivalent using scaling conventions. Provide `remanence_units` flag if needed.
- Option to pass `Q` (Koenigsberger) and remanence direction; compute magnitude from Q and induced magnitude.

### 2. Kernel usage
- For TMI with remanence, pre-projected scalar kernel remains valid if kernel is derived for unit dipole oriented along each axis and then projected onto IGRF direction. Two options:
  - Keep scalar pre-projected TMI kernel and require user to provide TMI-effective susceptibility (induced + projected remanence). Simpler, but less flexible.
  - Preferable: keep 3-component kernels Kx, Ky, Kz or full tensor and project on the fly with M_total vectors.

Implementation choice:
- Implement a 3-vector kernel per voxel: `K` such that `B_anom = K · M_total` (where K is a 3x3 effective mapping or three 3-vectors per voxel). For TMI: `ΔT = (K · M_total) · l`.

### 3. Forward calculation
- Map unit-wise `chi` and `M_r` to voxels.
- Compute `M_ind = chi * B0 / μ0` (vector along l scaled by F/μ0).
- `M_total = M_ind + M_r`.
- Apply K and project to TMI.

### 4. Tests
- Analytical dipole tests with specified remanence.
- Q-ratio scenarios: Q<1, ≈1, >1 produce plausible changes.

Acceptance criteria:
- Numerical stability at extreme inclinations and declinations.

---

## Phase 3 — Advanced magnetic outputs

Goal: Extend beyond TMI to support additional products.

Tasks:
- Output separate Bx, By, Bz components at devices.
- Optionally output full gradient tensor (for gradiometry).
- Implement Normalized Source Strength (NSS): `sqrt(Bx^2 + By^2 + Bz^2) / F0`.
- IGRF integration helper using pyIGRF (optional dependency): function to generate `igrf_params` from lat, lon, alt, date.

Acceptance criteria:
- New outputs gated by options; default behavior unchanged.

---

## Phase 4 — Performance and robustness

Tasks:
- GPU acceleration leveraging BackendTensor (PyTorch/JAX/TF backends) without branching logic.
- Cache kernels per grid geometry and IGRF where possible.
- Adaptive refinement close to sources (optional): increase grid density around devices or anomalies.
- Numerical safeguards: clipping arctan/log inputs, epsilon near singularities, stable atan2 usage.

Acceptance criteria:
- Comparable performance to gravity for equal voxel counts.
- No NaNs/Infs in practical geometries.

---

## Mathematical details — rectangular prism kernels

- Magnetic field from a uniformly magnetized rectangular prism can be written via derivatives of potential with closed-form corner sums using logs and arctans, analogous to gravity but vector/tensor-valued.
- Reference: Blakely (1995) Potential Theory, and many open references for prism magnetics.
- Implementation sketch for scalar TMI kernel per voxel with induced magnetization:
  - Compute tensor T_ij = ∂²(1/r)/∂x_i∂x_j integrated over voxel volume via 8-corner sum with signs.
  - For induced-only, M is parallel to l. Then B_anom = (μ0/4π) V (T · M).
  - If we pre-multiply by F/μ0 and project on l, we can define `tmi_kernel = (V/4π) (lᵀ T l) * (F)` and then simply sum `chi * tmi_kernel`.
  - This pushes physical constants into kernel for fast forward evaluation.

Unit notes:
- Input F in nT; internal convert to Tesla if needed and convert back to nT at the end. MVP can keep kernel in nT per unit susceptibility to simplify.

---

## Integration checklist

- [ ] New file: `magnetic_gradient.py` with kernel computation API.
- [ ] New file: `fw_magnetic.py` with `compute_tmi` and basic mapping.
- [ ] Update `GeophysicsInput` dataclass (if not already general) to include magnetic fields.
- [ ] Update `compute_model` to call magnetics forward when inputs present.
- [ ] Update `Solutions` to store `tmi`.
- [ ] Documentation: README magnetics section link to this plan.

---

## Testing plan

- [ ] Unit: direction cosines from I/D with edge cases (I=±90°, D wrap-around).
- [ ] Unit: kernel symmetry and distance decay.
- [ ] Analytical: induced sphere line test analogous to gravity sphere benchmark (tolerances 10–20%).
- [ ] Integration: compute_model pipeline with both gravity and magnetics.
- [ ] Performance smoke: large grid does not time out; memory within limits.

---

## Acceptance criteria (per phase)

Phase 1
- `calculate_magnetic_gradient_tensor(..., compute_tmi=True)` returns valid kernel.
- `compute_tmi` produces reasonable nT values; passes unit/integration tests.
- No regressions in gravity tests.

Phase 2
- Remanence inputs supported; tests cover Q-ratio scenarios.

Phase 3
- Component outputs and NSS available behind flags; docs updated.

Phase 4
- GPU/BackendTensor parity; caching; no numerical instabilities in CI tests.

---

## Migration notes (GemPy v2 → v3)

- v2 used Theano; v3 uses BackendTensor abstraction (NumPy/PyTorch/etc.). Avoid framework-specific code.
- Property mapping mirrors gravity: 1-based IDs -> property arrays via direct indexing.
- Reuse gravity’s batch-by-device reshaping pattern.
- Keep kernels precomputed and independent of properties to enable fast property sweeps.

---

## References

- Blakely, R.J. (1995). Potential Theory in Gravity and Magnetic Applications. CUP.
- Reford, M.S. (1980). Magnetic method. Geophysics, 45(11), 1640–1658.
- Li, Y., Oldenburg, D.W. (2003). Fast inversion of large-scale magnetic data using wavelet transforms. GJI, 152(2), 251–265.
- Shearer, S.E. (2005). Three-dimensional inversion of magnetic data in the presence of remanent magnetization. PhD thesis, CSM.
