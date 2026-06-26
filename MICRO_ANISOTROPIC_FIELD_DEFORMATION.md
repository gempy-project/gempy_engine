# Micro Anisotropic Field Deformation Notes

This document summarizes the current prototype and the recommended next steps for moving the micro-correction idea into GemPy Engine with minimal disruption to the existing macro interpolation path.

## Goal

Preserve the existing GemPy macro model as the structural/geological hypothesis, then apply an optional local scalar-field correction that improves high-density contact compliance without inserting all contacts into the global cokriging system.

The target architecture is:

```text
macro cokriging solve -> macro scalar field -> optional micro correction -> final scalar field
```

The macro solve should remain unchanged as much as possible:

- Do not add dense borehole contacts to the global covariance system.
- Do not modify `SolverInput` for macro interpolation.
- Do not modify covariance assembly, universal drift, fault drift, or cokriging weights.
- Keep the micro layer as an optional evaluator-side additive correction.

## Current Prototype

The prototype currently exists in:

- `gempy_engine/core/data/options/micro_anisotropic_options.py`
- `gempy_engine/modules/evaluator/micro_anisotropic_evaluator.py`
- `tests/test_common/test_modules/test_evaluator/test_micro_anisotropic_evaluator.py`
- `tests/test_common/test_modules/test_evaluator/test_micro_anisotropic_macro_integration.py`

The prototype supports:

- Pure NumPy micro correction evaluation.
- Symmetric micro covariance solve.
- 2D and 3D anisotropy matrix construction from macro gradients.
- Optional additive correction in both evaluator paths:
  - `symbolic_evaluator.py`
  - `generic_evaluator.py`
- A pytest integration demo with optional plotting via `GEMPY_PLOT_MICRO=1`.

Run the visual test with:

```bash
MPLBACKEND=QtAgg GEMPY_PLOT_MICRO=1 DEFAULT_BACKEND=numpy \
  /home/leguark/.venv/2025/bin/pytest \
  tests/test_common/test_modules/test_evaluator/test_micro_anisotropic_macro_integration.py -s
```

Run the headless tests with:

```bash
MPLBACKEND=Agg DEFAULT_BACKEND=numpy \
  /home/leguark/.venv/2025/bin/pytest \
  tests/test_common/test_modules/test_evaluator/ -v
```

At the time of writing, the evaluator test group passes:

```text
13 passed
```

## Micro Field Formula

The final field is currently:

```text
V_final(x) = V_macro(x) + V_micro(x)
```

where:

```text
V_micro(x) = sum_i w_i * exp(-||A_i (x - p_i)|| / range)
```

Definitions:

- `p_i`: micro constraint point.
- `w_i`: micro weight solved from the micro system.
- `A_i`: local anisotropy transform built from the macro gradient at `p_i`.
- `range`: micro kernel range, intended to be small relative to macro kernel range.

The current prototype uses `micro_kernel_range = 0.5` in the 2D demo.

## Micro Covariance Solve

Weights are solved from:

```text
K w = residuals
```

with a symmetric anisotropic distance:

```text
Dist^2(i, j) = (p_i - p_j)^T * M_ij * (p_i - p_j)
M_ij = (A_i^T A_i + A_j^T A_j) / 2
K_ij = exp(-Dist(i, j) / range)
```

The symmetric distance is important because the micro covariance matrix must stay symmetric and numerically stable.

The passive evaluation uses the cheaper one-sided distance:

```text
||A_i (x - p_i)||
```

This means exact round-trip equality is guaranteed for isotropic/identical transforms, but not necessarily for strongly varying anisotropy. This is expected from the current formulation.

## Target Scalar Logic

The main correction made during prototyping was how targets are computed.

Incorrect prototype logic:

```text
target_scalar = macro scalar at one arbitrary micro contact
residual_i = target_scalar - V_macro(contact_i)
```

Corrected logic:

1. Evaluate macro scalar values at original macro surface points.
2. Split those values by surface using `TensorsStructure.number_of_points_per_surface`.
3. Compute one target scalar per surface/interface.
4. Assign every micro contact to a surface/interface id.
5. Compute per-contact residuals using that contact's assigned surface target.

Current test uses median targets:

```python
target_per_surface = [
    median(V_macro(surface_0_points)),
    median(V_macro(surface_1_points)),
]

target_values_at_contacts = target_per_surface[micro_surface_ids]
residuals = target_values_at_contacts - V_macro(micro_contacts)
```

Median was chosen because it is robust and simple. It is also suitable while the system is still experimental.

## Macro Preservation Constraint

The most useful refinement so far is adding original macro surface points as zero-residual constraints in the micro solve.

Instead of solving only with micro contacts:

```text
points = contacts
residuals = target_surface - V_macro(contact)
```

the current prototype solves with:

```text
points = [contacts, macro_surface_points]
residuals = [target_surface - V_macro(contact), 0]
```

This tells the micro field:

- Correct the field at micro contacts.
- Preserve the already-authored macro field at original macro surface points.

In the 2D prototype this produced:

```text
RMS before: 2.459
RMS after:  0.311
Macro point drift max:  0.699
Macro point drift mean: 0.185
```

This is a good first result: contact compliance improves strongly while macro points move much less than the largest contact residuals.

## Anisotropy Construction

Anisotropy matrices are built from the macro gradient sampled at each micro constraint point.

The local frame is:

- Last local axis: normalized macro gradient, treated as stratigraphic up / vertical.
- Remaining local axes: lateral directions.

The scale matrix is:

```text
2D: S = diag(1 / r_lateral, 1 / r_vertical)
3D: S = diag(1 / r_lateral, 1 / r_lateral, 1 / r_vertical)
```

The transform is:

```text
A_i = S * R_i^T
```

where `R_i` contains the local basis vectors as columns.

Interpretation:

- Smaller `r_vertical` means faster decay across stratigraphy.
- Larger `r_lateral` means wider influence along the layer.

The visual test draws ellipses for contact anisotropy using:

```text
||A_i d|| = micro_kernel_range
```

In 3D, the equivalent visualization would be ellipsoids or principal axes arrows.

## Minimal Production Architecture

Keep the production change centered around an optional evaluator overlay.

### Data Object

Continue with an option object similar to:

```python
class MicroAnisotropicOptions(BaseModel):
    enabled: bool = False
    points: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    anisotropy_matrices: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    kernel_range: float = 1.0
    nugget: float = 0.0
```

Potential additions:

```python
strength: float = 1.0
preserve_macro_points: bool = True
r_vertical: float = 0.5
r_lateral: float = 5.0
```

If `strength` is added, evaluation becomes:

```text
V_final = V_macro + strength * V_micro
```

This is useful as a diagnostic/tuning knob, but it is not a replacement for macro zero constraints.

### Evaluator Hook

Keep the hook after macro scalar evaluation:

```python
scalar_field = scalar_field + evaluate_micro_correction(...)
```

Do this in both:

- `symbolic_evaluator`
- `generic_evaluator`

Reason: PyKeOps currently has known LazyTensor compatibility issues in parts of the evaluator stack. The generic path must remain capable of exercising the micro workflow.

### Avoid Touching

Avoid touching these until necessary:

- `compute_weights()`
- `_solve_interpolation()`
- covariance matrix assembly
- kernel constructor internals
- `SolverInput` semantics
- stack loop / octree loop

## Moving To 3D

The 3D implementation should follow the same steps as the 2D test, but with 3-component coordinates and gradients.

### 3D Pipeline

1. Run macro solve normally.
2. Evaluate macro scalar and gradient at micro contacts.
3. Evaluate macro scalar and gradient at macro surface points.
4. Compute target scalar per surface:

```python
target_per_surface[s] = median(V_macro(points_of_surface_s))
```

5. Assign each micro contact a surface id:

```python
micro_surface_ids: np.ndarray  # shape (N_contacts,)
```

6. Compute contact residuals:

```python
contact_residuals = target_per_surface[micro_surface_ids] - V_macro(micro_contacts)
```

7. Build augmented constraints:

```python
constraint_points = np.vstack([micro_contacts, macro_surface_points])
constraint_residuals = np.concatenate([contact_residuals, zeros_for_macro_points])
constraint_gradients = np.vstack([grad_at_contacts, grad_at_macro_points])
```

8. Build anisotropy matrices:

```python
A = compute_anisotropy_matrices_from_gradients(
    constraint_points,
    constraint_gradients,
    r_vertical=..., 
    r_lateral=...,
)
```

9. Solve micro weights:

```python
micro_weights = solve_micro_weights(
    constraint_points,
    constraint_residuals,
    A,
    kernel_range=...,
    nugget=...,
)
```

10. Store on `options.evaluation_options.micro_anisotropic` and evaluate final field.

### 3D Frame Construction

For each gradient `g`:

```text
z_axis = normalize(g)
ref = [0, 1, 0]
if abs(dot(z_axis, ref)) > 0.99:
    ref = [1, 0, 0]
x_axis = normalize(cross(z_axis, ref))
y_axis = normalize(cross(z_axis, x_axis))
R = [x_axis, y_axis, z_axis]
A = S @ R.T
```

This exists in the prototype and should be kept unless a more geologically meaningful strike direction is available.

### 3D Tests To Add

Add a test using `simple_model` or another lightweight 3D fixture:

- Macro solve.
- Choose synthetic 3D micro contacts assigned to one surface.
- Compute per-surface median target scalar.
- Add macro surface points as zero constraints.
- Solve micro correction.
- Evaluate at contacts and macro points.

Assertions:

```python
contact_rms_after < contact_rms_before
max_macro_point_drift < tolerance
np.all(np.isfinite(final_field))
```

Start with a loose macro drift tolerance and tighten it after visual inspection.

## Octree And Mesh Extraction

The current prototype only corrects scalar values at evaluation locations. For mesh extraction to capture micro contacts reliably, the octree must eventually refine around micro contact locations.

Without this, a micro contact can lie inside a coarse cell that never gets evaluated finely enough for dual contouring to capture the corrected crossing.

Recommended staged approach:

### Stage 1: Scalar Evaluation Only

Current state. Validate math and field behavior.

### Stage 2: Corners / Dense Grid Evaluation

Evaluate micro correction on the grid or octree corners used for mesh extraction.

No octree logic changes yet.

### Stage 3: Forced Refinement Around Contacts

Add an optional octree refinement criterion:

```text
refine cell if it intersects macro isosurface OR contains/near a micro contact
```

Try to implement this as a narrow optional hook in octree refinement, not in the macro interpolation loop.

Potential option fields:

```python
micro.force_octree_refinement: bool = False
micro.refinement_radius: float = ...
```

### Stage 4: Dual Contouring Validation

Once micro correction is evaluated at final corner locations, run dual contouring and verify that extracted surfaces move toward micro contacts.

## PyKeOps / GPU Path

The prototype uses dense NumPy for the micro solve. This is correct for math validation.

Next GPU path should be added behind the same public functions, not by changing evaluator call sites.

Suggested evolution:

```python
solve_micro_weights(..., backend="numpy" | "pykeops")
evaluate_micro_correction(..., backend="numpy" | "pykeops")
```

or use `BackendTensor` dispatch internally.

Be careful: the project currently has known PyKeOps LazyTensor issues around NumPy ufuncs and some matrix operations. Keep the NumPy implementation as the reference path.

Known environment/test pitfalls:

- Some PyKeOps tests try to write to `/home/miguel`, causing permission errors.
- Some LazyTensor expressions fail with standard NumPy ufuncs like `sqrt`/`exp`.
- Run new tests with `MPLBACKEND=Agg` unless plotting intentionally.

## Suggested Helper To Promote Later

Once the 3D test looks good, promote the repeated integration-test logic into a helper, probably under `modules/evaluator` or a small `modules/micro_correction` package.

Candidate function:

```python
def build_micro_anisotropic_constraints(
    macro_solver_input: SolverInput,
    macro_weights: np.ndarray,
    options: InterpolationOptions,
    micro_contacts: np.ndarray,
    micro_surface_ids: np.ndarray,
    macro_surface_points: np.ndarray,
    n_points_per_surface: np.ndarray,
    r_vertical: float,
    r_lateral: float,
    kernel_range: float,
    nugget: float,
) -> MicroAnisotropicOptions:
    ...
```

Responsibilities:

- Evaluate macro scalar/gradient at contacts.
- Evaluate macro scalar/gradient at macro surface points.
- Compute per-surface target scalars.
- Build augmented constraints with zero macro residuals.
- Build anisotropy matrices.
- Solve micro weights.
- Return populated `MicroAnisotropicOptions`.

This helper should not call `compute_model()` and should not mutate macro input data.

## Open Design Questions

1. **Surface assignment source**

   The current prototype uses explicit `micro_surface_ids`. Production code needs this information from borehole/contact metadata.

2. **Macro preservation strength**

   Zero macro constraints help, but there can still be drift. If stricter preservation is required, consider:

   - Smaller kernel range.
   - Larger macro constraint weight via repeated constraints or lower nugget.
   - A weighted solve.
   - Adding nearby orientation constraints later.

3. **Contact vs macro weighting**

   The current system treats contact residuals and zero macro residuals equally. Production may need weights:

   ```text
   high confidence borehole contacts vs expert macro control points
   ```

4. **Per-surface micro solve**

   Solving each surface independently may reduce cross-interface interference. This is attractive for production.

5. **Faults**

   The prototype ignores fault-specific behavior. A later version should evaluate whether micro corrections should be isolated by fault block or respect fault masks.

6. **Gradient quality**

   Anisotropy relies on macro gradients. Need safeguards for near-zero gradients, noisy gradients, and points outside the stable macro field region.

## Recommended Next Steps

1. Add a 3D pytest integration demo analogous to the current 2D test.
2. Add optional visual 3D diagnostics:
   - Slices through macro/micro scalar fields.
   - Micro contact residuals before/after.
   - Principal axes of anisotropy ellipsoids.
3. Promote repeated constraint-building logic into a helper.
4. Add `strength` as a diagnostic/tuning parameter, but keep zero macro constraints as the preferred preservation mechanism.
5. Add per-surface or per-stack solve mode to reduce cross-talk.
6. Add optional contact-driven octree refinement.
7. Only after the NumPy path is stable, add PyKeOps versions of the micro solve/evaluation behind the same function signatures.

## Current Best Mental Model

Treat the macro model as the authored geological hypothesis and the micro field as a local, anisotropic, constrained residual corrector.

The micro solve should answer:

```text
What is the smallest local correction field that moves assigned contacts onto their intended surface scalar values while keeping macro control points unchanged?
```

That framing keeps the macro path intact and makes the micro layer optional, testable, and removable.
