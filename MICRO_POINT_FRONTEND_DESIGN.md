# Micro Point Front-End and Serialization Design

Status: proposed

This document defines how high-density micro contacts should enter GemPy, cross
the GemPy-to-engine boundary, and be serialized. It also defines the local RBF
correction that consumes those contacts.

Where this document conflicts with `MICRO_ANISOTROPIC_FIELD_DEFORMATION.md` or
`PLAN.md`, this document is authoritative. Those documents describe the
prototype and contain assumptions that are not valid for spatially varying
anisotropy.

## 1. Purpose

Micro points are dense observations used to make an existing macro geological
model locally comply with contacts without adding every contact to the global
cokriging system.

The intended pipeline is:

```text
GemPy macro observations
    -> macro cokriging solve
    -> macro scalar field and gradients
    -> per-stack micro residual solve
    -> macro field + local micro correction
    -> activation, octree refinement, and mesh extraction
```

The macro model remains the structural hypothesis. Micro points are a local
compliance layer and do not replace surface points or orientations.

## 2. Goals

- Represent each micro point as a position, local geological frame, and local
  anisotropic support.
- Use the same representation in a Python front end and a 3D editor.
- Associate every micro point with exactly one `StructuralElement`.
- Keep authored observations separate from options and solved runtime state.
- Preserve micro points through `gempy.save_model()` and `gempy.load_model()`.
- Apply a correction only to the stack and interface to which a point belongs.
- Use one mathematically consistent operator for fitting and evaluation.
- Preserve coordinate, dtype, device, and gradient consistency.

## 3. Non-Goals

- Micro points do not participate in the macro cokriging matrix.
- Solved residuals and RBF weights are not durable model input.
- Version 1 does not support one micro point constraining multiple interfaces.
- Version 1 does not define micro corrections across fault blocks.
- Version 1 does not accept perspective transforms or arbitrary projective
  matrices.

## 4. Terminology

- **Macro point:** A standard `SurfacePointsTable` or `OrientationsTable`
  observation used by the main interpolation system.
- **Micro point:** A dense contact observation used by the additive local
  correction.
- **Support transform:** A `4x4` affine transform mapping normalized local
  support coordinates to the model's world/input coordinate system.
- **Support scale:** The correlation lengths encoded in the linear part of a
  support transform. It is not merely a display-gizmo scale.
- **Local RBF:** A radial basis function centered at one micro point and
  evaluated using that point's support transform.

## 5. Front-End Data Model

### 5.1 Ownership

`MicroPointsTable` should be owned by `StructuralElement`, in the same vein as
`SurfacePointsTable` and `OrientationsTable`:

```python
class StructuralElement:
    surface_points: SurfacePointsTable
    orientations: OrientationsTable
    micro_points: MicroPointsTable
```

This establishes the interface association without a separate mutable
model-level relation. Moving an element between groups moves its micro points;
removing an element removes them. Basement elements must have an empty micro
table.

`StructuralFrame` should provide derived aggregate views:

```text
micro_points_copy
number_of_micro_points_per_element
number_of_micro_points_per_group
```

The aggregate table is needed for binary serialization and engine conversion,
but it is not the authoritative mutable owner.

The element key used in flattened binary rows must be stable across renaming.
The current fallback `StructuralElement.id` is derived from the element name
when `_id == -1`; that is not a sufficient durable foreign key. Before micro
points are persisted, the implementation must either materialize and serialize
an explicit element ID or introduce a stable element UUID. Dense surface and
stack indices remain runtime values derived from current structural order.

### 5.2 Canonical Transform

For point `i`, define:

```text
x_world_h = H_world_from_support[i] @ x_support_h
```

with:

```text
H_world_from_support = [ B_i  p_i ]
                       [ 0     1  ]
```

- `p_i` is the micro-point position.
- The columns of `B_i` are local support axes expressed in world coordinates.
- The lengths of those columns are the kernel correlation lengths.
- The third local axis is the interface-normal direction.

For the initial axisymmetric model:

```text
B_i = R_i @ diag(lateral_range_i, lateral_range_i, normal_range_i)
```

where `normal_range_i < lateral_range_i` in the common case. Rotation around
the normal has no effect when both lateral ranges are equal, but retaining a
complete frame is convenient for 3D front ends and allows future triaxial
support.

The position must not also be serialized as independent `X`, `Y`, and `Z`
fields. The translation column is authoritative. A convenience `xyz` property
may return `support_transforms[:, :3, 3]`.

### 5.3 Scale Semantics

The three support scales are physical correlation lengths in model coordinate
units. Applying the inverse transform produces dimensionless local coordinates.

This removes the ambiguous double scaling in the prototype, where
`anisotropy_matrices` contain inverse ranges and the result is divided by a
second `kernel_range`. The durable representation has one source of geometric
range: the support transform.

An optional global support multiplier may exist as an algorithm option, but it
must multiply all support lengths explicitly and must be applied identically
during fitting and evaluation.

### 5.4 Proposed Table

The conceptual public object is:

```python
@dataclass
class MicroPointsTable:
    data: np.ndarray
    name_id_map: dict[str, int] | None = None

    @classmethod
    def from_transforms(
        cls,
        support_transforms: np.ndarray,
        names: Sequence[str] | str,
        nugget: np.ndarray | None = None,
        name_id_map: dict[str, int] | None = None,
    ) -> "MicroPointsTable": ...

    @classmethod
    def initialize_empty(cls) -> "MicroPointsTable": ...

    @property
    def support_transforms(self) -> np.ndarray: ...

    @property
    def xyz(self) -> np.ndarray: ...
```

The proposed version 1 structured dtype is:

```python
np.dtype([
    ("support_transform", "<f8", (4, 4)),
    ("element_id", "<i8"),
    ("nugget", "<f8"),
])
```

This has a packed row size of 144 bytes:

| Field | Bytes | Meaning |
|---|---:|---|
| `support_transform` | 128 | Local-support-to-world affine transform |
| `element_id` | 8 | Durable association used during flatten/load |
| `nugget` | 8 | Diagonal regularization for this observation |

Little-endian encoding is explicit. The wider element ID avoids repeating the
`int32` limitation of the existing point tables.

The per-row ID remains useful even though elements own their tables: the
flattened binary table needs to be redistributed after loading, just as the
existing surface-point and orientation tables are redistributed by ID.

### 5.5 Authored Versus Derived State

Persisted input:

- Support transforms.
- Element association.
- Per-observation nugget.

Algorithm configuration:

- Enabled state.
- Kernel family.
- Correction strength.
- Macro-preservation policy.
- Solver and conditioning policy.
- Contact-driven refinement policy.

Derived runtime state:

- Engine-coordinate support transforms.
- Macro values and gradients at micro points.
- Target interface scalar values.
- Residuals.
- Local RBF weights.
- Solver diagnostics.

Derived state must not be stored in `MicroAnisotropicOptions` or serialized as
model input. It becomes stale when macro observations, transforms, structural
grouping, faults, or interpolation settings change.

## 6. Validation

`MicroPointsTable` construction must validate:

- Shape is exactly `(N, 4, 4)`.
- All matrix and nugget values are finite.
- Every last row is approximately `[0, 0, 0, 1]`.
- The `3x3` linear block is invertible.
- Authored support axes are orthogonal within tolerance for version 1.
- All three authored support scales are strictly positive.
- The authored linear block has positive determinant.
- The condition number remains below a documented limit.
- The number of names, nuggets, and transforms is identical.
- Every element association resolves to a non-basement element.
- Nugget values are nonnegative.

Version 1 deliberately accepts TRS support transforms, not shear. The combined
world-to-engine transform may make the engine-space linear block non-orthogonal;
orthogonality is therefore checked on authored transforms before composition,
not after it.

Normal validation errors must use regular exceptions or Pydantic validators,
not `assert`, because assertions disappear under `python -O`.

## 7. Coordinate Frames

### 7.1 Persisted Frame

Support transforms are persisted in the same world/input coordinate frame as
surface points and orientation positions. Micro points must not influence the
calculation of the model's global input transform; adding local compliance data
must not rescale the macro model.

### 7.2 World-to-Engine Conversion

Let `E` be the complete homogeneous world-to-engine transform, including the
grid transform around its cached pivot followed by `GeoModel.input_transform`:

```text
x_engine_h = E @ x_world_h
```

Then each support transforms exactly by composition:

```text
H_engine_from_support = E @ H_world_from_support
```

Runtime evaluation extracts:

```text
p_i = H_engine_from_support[:3, 3]
B_i = H_engine_from_support[:3, :3]
A_i = inverse(B_i)
```

For an engine-space query point `x`, normalized local coordinates and distance
are:

```text
u_i(x) = A_i @ (x - p_i)
r_i(x) = ||u_i(x)||
```

This guarantees coordinate invariance:

```text
||A_world @ (x_world - p_world)||
    ==
||A_engine @ (x_engine - p_engine)||
```

The conversion must use homogeneous matrix multiplication. It must not use
component-wise `Transform.__add__` or decompose the support through
`Transform.from_matrix()`, because those paths do not preserve general affine
composition under rotation and nonuniform scale.

## 8. Engine Boundary

### 8.1 Data Placement

Numerical micro observations belong in `InterpolationInput`, alongside surface
points and orientations. They do not belong in interpolation options.

Conceptually:

```python
@dataclass
class MicroPoints:
    support_transforms: ArrayLike  # (N, 4, 4), engine coordinates
    surface_indices: ArrayLike     # (N,), dense global interface indices
    nuggets: ArrayLike             # (N,)
```

`InputDataDescriptor` should carry partition metadata, preferably the number of
micro points per surface. That supports validation and deterministic slicing by
stack without pretending micro points are macro cokriging constraints.

`MicroAnisotropicOptions` should contain policy only. The prototype fields
`points`, `residuals`, `anisotropy_matrices`, and `weights` should not form the
public architecture.

### 8.2 Stack Isolation

During stack subset construction, include only micro points associated with
surfaces in the active stack. The resulting correction must not be added to:

- Unrelated stratigraphic stacks.
- Fault scalar fields unless fault micro correction is explicitly supported.
- External or null-space interpolation groups unless explicitly supported.

This is required because the current prototype stores one global correction in
evaluation options and can apply it to every evaluated scalar field.

## 9. Local RBF Correction

### 9.1 Macro Targets and Residuals

For micro point `p_i` associated with interface `s(i)`, evaluate the macro field
and use the same canonical interface scalar value consumed by activation:

```text
y_i = c_macro[s(i)] - V_macro(p_i)
```

The canonical macro interface values must remain immutable during the micro
pass. They must not be recomputed from surface-point samples after applying the
micro correction.

### 9.2 Consistent Basis

For a kernel `k`, basis `i` evaluated at query `x` is:

```text
phi_i(x) = k(r_i(x))
```

where `r_i` is calculated from support transform `i`. Construct the fitting
matrix by evaluating that exact basis at every constraint point:

```text
Phi[j, i] = phi_i(p_j)
```

Solve:

```text
(Phi + diag(nugget)) @ w = y
```

and evaluate:

```text
V_micro(x) = sum_i w_i * phi_i(x)
V_final(x) = V_macro(x) + strength * V_micro(x)
```

`Phi` is generally nonsymmetric because each column uses its source point's
support transform. The system is therefore a local RBF system, not a covariance
matrix. Dense direct solve, GMRES, BiCGSTAB, or least squares are valid solver
families; Conjugate Gradient is not valid for the general nonsymmetric system.

A nonzero nugget deliberately relaxes exact snapping. With zero nugget and a
nonsingular system, evaluating the fitted basis at the constraint points must
reproduce the residual vector.

### 9.3 Prototype Difference

The snapping prototype solves with a symmetric pair distance based on:

```text
(A_i.T @ A_i + A_j.T @ A_j) / 2
```

but evaluates with only the source matrix `A_i`. Those operators differ when
anisotropy varies between points. The identity-matrix round-trip test does not
expose the mismatch.

The production design uses the one-sided source basis in both fitting and
evaluation. It makes no positive-semidefinite covariance claim.

### 9.4 Gradients

When scalar gradients are requested, the returned gradient must include the
micro correction. For `u_i = A_i(x - p_i)` and `r_i = ||u_i||`:

```text
grad(r_i) = A_i.T @ u_i / r_i
grad(phi_i) = k'(r_i) * grad(r_i)
```

The implementation must handle `r_i = 0` using the analytic kernel limit.

```text
grad(V_final) = grad(V_macro)
              + strength * sum_i w_i * grad(phi_i)
```

Returning a corrected scalar with macro-only gradients is invalid for dual
contouring and any gradient-based refinement or diagnostics.

## 10. Model Serialization

### 10.1 Existing `save_model` Format

The user-facing `gempy.save_model()` currently writes a deterministic ZIP-like
`.gempy` container in `gempy/modules/serialization/save_load.py`:

```text
model.gempy
|-- header.json
|-- input.bin
|-- grid.bin
`-- liquid_earth_meta.json
```

Current behavior relevant to this design:

- `header.json` is generated with `GeoModel.model_dump_json()`.
- `SurfacePointsTable.data` and `OrientationsTable.data` are excluded from JSON.
- `input.bin` concatenates all surface-point rows followed by all orientation
  rows.
- `StructuralFrame.binary_meta_data` records the two byte lengths.
- Loading injects binary data through `loading_model_from_binary()` while
  Pydantic reconstructs the model.
- The frame validator decodes global tables and redistributes rows to elements
  by element ID.

This is the correct ownership and reconstruction pattern for micro points, but
the existing binary stream should not simply receive a third unversioned
segment.

### 10.2 Proposed Container Version

Add a serialization manifest to `header.json`:

```json
{
  "serialization": {
    "format": "gempy",
    "version": 2,
    "writer_version": "<gempy-version>",
    "byte_order": "little"
  }
}
```

A missing manifest means legacy version 1.

Version 2 adds a dedicated member:

```text
model.gempy
|-- header.json
|-- input.bin
|-- micro_points.bin
|-- grid.bin
`-- liquid_earth_meta.json
```

A dedicated member is preferable to appending data to `input.bin` because:

- Existing readers currently ignore trailing `input.bin` bytes.
- A distinct member has an independently validated length and schema.
- Legacy surface-point and orientation layout remains unchanged.
- Future micro-table versions can evolve without changing macro table offsets.

The structural-frame metadata should include:

```json
{
  "micro_points": {
    "dtype_version": 1,
    "row_count": 42,
    "byte_length": 6048
  }
}
```

The reader must use the fixed dtype selected by `dtype_version`; it must not
execute or blindly trust an arbitrary dtype supplied by the file.

### 10.3 Save Flow

1. Each `StructuralElement.micro_points.data` remains excluded from JSON.
2. `StructuralFrame.micro_points_copy` concatenates rows in structural order.
3. `model_to_bytes()` writes those bytes to `micro_points.bin`.
4. The ZIP member order and timestamps remain deterministic.
5. Serialization validation compares the original and loaded micro tables
   directly, not through process-local `hash(bytes)` values.

The ZIP writer should set `ZipInfo.compress_type` explicitly if compression is
expected. The current `make_info()` path creates stored members despite the
`ZipFile` compression setting.

### 10.4 Load Flow

1. Read and validate the serialization manifest.
2. Read `micro_points.bin` for format version 2.
3. Verify its exact byte length and row-size divisibility before `np.frombuffer`.
4. Inject it through the binary loading context with `input.bin` and `grid.bin`.
5. Construct `StructuralElement` objects with empty micro tables by default.
6. Decode the global micro table using its fixed little-endian dtype.
7. Reject rows whose element IDs are unknown or duplicated ambiguously.
8. Redistribute rows to elements by `element_id`.
9. Run normal table and affine-transform validation.

Loading a version 1 model, or a transitional archive with no
`micro_points.bin`, produces empty micro tables. Old model files therefore
remain loadable.

### 10.5 What Is Not Serialized

Do not serialize:

- Engine-coordinate transforms.
- Inverse `3x3` anisotropy operators.
- Macro scalar samples or gradients.
- Target scalar values.
- Residuals.
- RBF weights.
- Solver factorizations or condition estimates.

These values depend on the current macro model and are recomputed. If solved
state is cached in the future, it must be a disposable cache keyed by a strong
fingerprint over all inputs and options, not authoritative model data.

## 11. Required Tests

The following tests are required when this design is implemented.

### 11.1 `MicroPointsTable`

- Empty initialization.
- Construction from one and multiple support transforms.
- Exact dtype names, byte order, offsets, and 144-byte row size.
- `xyz` extraction from the translation column.
- Selection by element name and ID.
- Copy and writable-view behavior.
- Rejection of incorrect array shapes.
- Rejection of non-affine last rows.
- Rejection of NaN and infinity.
- Rejection of singular or ill-conditioned support blocks.
- Rejection of zero or negative support scales.
- Rejection of shear in the version 1 authored format.
- Rejection of negative nuggets and mismatched array lengths.

Suggested location:

```text
gempy/test/test_core/test_micro_points.py
```

### 11.2 Structural Ownership

- Each element owns an independent micro table.
- Flattening preserves structural order.
- Per-element and per-group counts are correct.
- Redistributing a flattened table restores exact ownership.
- Moving an element between groups moves its micro points.
- Removing an element cannot leave dangling micro rows.
- Basement contributes zero rows and rejects authored micro points.

### 11.3 Serialization

- Empty micro tables round-trip through `save_model` and `load_model`.
- Multiple elements with different row counts round-trip exactly.
- All 16 transform values, IDs, and nuggets compare exactly after loading.
- Binary rows shuffled before loading are redistributed by element ID.
- Existing version 1 fixtures load with empty micro tables.
- Missing required version 2 members raise a clear error.
- Unsupported major versions raise a clear error.
- Truncated rows and incorrect byte lengths are rejected.
- Unknown element IDs are rejected rather than silently discarded.
- Two saves of the same model produce identical archive bytes.
- Large micro arrays remain in binary and never enter `header.json`.
- Saving after compute does not persist residuals or weights.

Suggested location:

```text
gempy/test/test_modules/test_serialize_model.py
```

### 11.4 Coordinate Conversion

- Identity conversion.
- Translation, rotation, and isotropic model scaling.
- Nonuniform model scaling.
- Grid rotation around a nonzero pivot.
- Combined grid and input transforms.
- Micro center conversion matches the normal point-conversion pipeline.
- Normalized support distance is invariant between world and engine frames.
- Source matrices are not mutated during conversion.

### 11.5 Local RBF

- One-point correction.
- Solve/evaluate round trip with different support transforms at every point.
- All supported kernels use the same type during solve and evaluation.
- Zero nugget reproduces residuals within solver tolerance.
- Nonzero nugget has documented smoothing behavior.
- `strength=0` returns the exact macro field.
- Analytic micro gradients match finite differences.
- Scalar output is identical whether gradients are requested or not.
- Duplicate and nearly duplicate points fail or regularize deterministically.
- NumPy and PyTorch preserve dtype, device, and numerical parity.

### 11.6 Integration

- Micro points only modify their associated stack.
- Points on two interfaces receive the correct target scalar values.
- Fault and unsupported stack types reject micro correction clearly.
- Macro-preservation policy has measurable, documented behavior.
- Corrected gradients reach dual contouring.
- Contact-driven octree refinement resolves isolated contacts.
- Extracted interfaces approach contacts within the configured tolerance.
- Plotting is opt-in and disabled in automated tests.

## 12. Implementation Sequence

The recommended implementation order is:

1. Add and validate `MicroPointsTable` in the user-facing `gempy` package.
2. Attach an empty table to every `StructuralElement` and aggregate it through
   `StructuralFrame`.
3. Introduce serialization version 2 and `micro_points.bin` with compatibility
   tests for existing files.
4. Add GemPy APIs for adding, modifying, and deleting micro points.
5. Add the numerical micro data object to `InterpolationInput` and partition
   metadata to `InputDataDescriptor`.
6. Compose support transforms into engine coordinates in the GemPy engine
   factory.
7. Slice micro data per stack and derive residuals from immutable macro
   interface values.
8. Replace the prototype solve with the consistent local RBF system.
9. Add micro gradients, backend parity, and conditioning diagnostics.
10. Add contact-driven octree refinement and mesh-level compliance tests.

Each stage should leave models with no micro points behaviorally identical to
current models.

## 13. Open Decisions

- Whether macro preservation uses all macro surface points, one reference point
  per interface, or nearby weighted anchors.
- Which nonsymmetric solver is used after the dense reference implementation.
- Whether per-point nugget is a direct diagonal regularizer or derived from a
  separately named uncertainty measurement.
- Whether triaxial supports with unequal lateral scales are exposed in the
  first public API or only accepted through complete support transforms.
- How micro correction is masked across finite-fault domains.
- Whether support transforms are editable through Euler/TRS convenience APIs
  while retaining the raw matrix as the canonical representation.

These decisions do not change the core contract: a micro observation is owned
by one structural element and persisted as a local-support-to-world `4x4`
transform whose scale defines anisotropic kernel support.
