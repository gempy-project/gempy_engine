# Finite Faults Implementation

## Goal

Define finite faults as validated, serializable input data and use that
definition to taper fault displacement on the interpolated fault surface.

The implementation is based on the projection and UV taper prototype in
`gempy_engine/modules/faults/finite_faults.py`. Its declarative input is now
connected to fault-stack interpolation; geometric and backend validation remain
for Phase 4.

## Status

### Phase 1: Serializable input definition

- [x] Introduce a Pydantic dataclass for the finite-fault definition.
- [x] Keep persisted fields JSON-native rather than storing NumPy arrays.
- [x] Serialize taper types as stable string values.
- [x] Validate centers, radii, rotation, and spline control points.
- [x] Add JSON and Python round-trip tests using `pydantic.TypeAdapter`.
- [x] Preserve the prototype import path and numerical API.

### Phase 2: Projection correctness

- [x] Remove the incorrect fixed half-step from plane projection.
- [x] Define explicit behavior for near-zero gradients.
- [x] Test exact projection onto a plane and projection idempotency.
- [x] Decide whether nonlinear fields need iterative re-evaluation.
- [x] Fix the dense-grid gradient accessor before using it for projection.

### Phase 3: Stack wiring

- [x] Replace the callable in `FiniteFaultData` with the declarative definition.
- [x] Associate at most one finite-fault definition with each fault stack.
- [x] Validate that definitions are attached only to `StackRelationType.FAULT` stacks.
- [x] Make scalar gradients available when a finite-fault stack is evaluated.
- [x] Pass the fault scalar field, gradients, and surface isovalue to the taper operation.
- [x] Apply the taper to the fault drift before dependent stacks are interpolated.
- [x] Cover both sequential and flat-stack interpolation paths.

### Phase 4: Integration and backend support

- [ ] Add a numerical integration test for a dependent stratigraphic stack.
- [ ] Verify that displacement reaches zero at the finite-fault tips.
- [ ] Document the approximately planar local-frame limitation.
- [ ] Define and test NumPy and PyTorch backend behavior.
- [ ] Add the finite-fault definition to the server payload when stack data is exposed there.

## Input Contract

`FiniteFault` is a frozen Pydantic dataclass. Its persisted representation
contains only JSON-compatible values:

| Field | Type | Meaning |
| --- | --- | --- |
| `center` | 3-tuple of floats | Point at the center of the finite-fault footprint |
| `strike_radius` | positive float or 2-tuple | Positive and negative strike radii |
| `dip_radius` | positive float or 2-tuple | Positive and negative dip radii |
| `taper` | `cubic`, `quadratic`, or `spline` | Slip taper profile |
| `rotation_deg` | float | In-plane rotation in degrees |
| `spline_control_points` | optional sequence of 2-tuples | Distance-to-slip profile for a spline taper |

For an asymmetric radius, tuple order is `(positive_direction,
negative_direction)`. All radii must be finite and greater than zero.

Spline control points use `(normalized_distance, slip_multiplier)`. Distances
must be strictly increasing from `0` to `1`, and multipliers must remain in the
range `[0, 1]`. Omitting the points for a spline taper selects the engine's
default profile. Supplying spline points for another taper is invalid.

`normal_radius` is intentionally not part of this contract. The UV workflow
projects points onto the fault surface, so normal distance is not part of its
two-dimensional footprint. A volumetric ellipsoid would be a separate model.

Direct GemPy Engine callers provide the center and radii in engine coordinates.
GemPy models persist these values in world coordinates and transform a runtime
copy together with the model inputs before calling the engine. The persisted
finite-fault definition is not modified. The current strike/dip representation
requires an isotropic transform that does not tilt the vertical axis.

## Serialization

Pydantic dataclasses use `TypeAdapter` for serialization and deserialization:

```python
from pydantic import TypeAdapter

from gempy_engine.core.data.finite_fault import FiniteFault, TaperType

adapter = TypeAdapter(FiniteFault)

finite_fault = FiniteFault(
    center=(0.0, 0.0, 0.0),
    strike_radius=(2.0, 1.0),
    dip_radius=0.75,
    taper=TaperType.SPLINE,
    rotation_deg=15.0,
    spline_control_points=(
        (0.0, 1.0),
        (0.5, 0.8),
        (1.0, 0.0),
    ),
)

payload: bytes = adapter.dump_json(finite_fault)
restored: FiniteFault = adapter.validate_json(payload)
```

Equivalent JSON:

```json
{
  "center": [0.0, 0.0, 0.0],
  "strike_radius": [2.0, 1.0],
  "dip_radius": 0.75,
  "taper": "spline",
  "rotation_deg": 15.0,
  "spline_control_points": [
    [0.0, 1.0],
    [0.5, 0.8],
    [1.0, 0.0]
  ]
}
```

## Projection Contract

`project_points_onto_surface` performs one Newton step:

```text
P' = P - (F(P) - target) * grad(F(P)) / ||grad(F(P))||^2
```

This projection is exact for a linear scalar field. The function cannot iterate
for a nonlinear field because its inputs contain scalar and gradient values only
at the original points. Iterative projection, if required by integration tests,
must re-evaluate the interpolator at each set of projected coordinates and will
be implemented at the engine integration layer.

A point whose gradient norm is below `gradient_tolerance` is left unchanged only
when its scalar residual is within `surface_tolerance`. Otherwise projection is
undefined and the function raises `ValueError`. Tolerances are keyword-only and
must be non-negative.

## Stack Attachment

The finite-fault definition belongs to the stack that interpolates the fault.
Attach it through that stack's `FaultsData` entry:

```python
from gempy_engine.core.data.kernel_classes.faults import FaultsData

fault_data = FaultsData.from_user_input(
    thickness=None,
    finite_fault=finite_fault,
)
input_data_descriptor.stack_structure.faults_input_data = [
    fault_data,
    None,
    None,
]
```

`faults_input_data` must contain one entry per stack. A finite-fault definition
is valid only on a cokriging stack with `StackRelationType.FAULT` and exactly one
surface, because that surface supplies the projection isovalue.

The engine privately enables scalar gradients for finite-fault stacks without
changing the caller's global or per-stack options. The fault scalar field,
full-length gradients, and surface isovalue are used to project all evaluation
coordinates. A fixed local frame is taken from the valid projected point nearest
the configured center. The resulting slip multiplier tapers the stored fault
drift before dependent stacks read it.

In flat-stack mode, finite-fault stacks are isolated into singleton chunks and
use the non-stacked symbolic evaluator. Ordinary stacks continue to use the
optimized stacked evaluator. NumPy and PyKeOps reductions can differ slightly
because their floating-point reduction orders are different.

## Known Prototype Issues

- Standalone `FiniteFault.calculate_slip` expects callers to project points separately; engine stack wiring performs that projection.
- The local strike/dip frame is constant and therefore approximates curved faults.
- Existing integration assertions do not verify the projected surface residual or expected geometry.

## Design Decisions

- The finite-fault definition is declarative; serialized callables are not supported.
- NumPy conversion happens at the numerical boundary, not in persisted fields.
- Geometry belongs to the fault stack that defines it, not each destination stack affected by it.
- Projection algorithm settings are evaluation concerns and are not part of geological input data.
