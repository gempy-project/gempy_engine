# Best-effort extent capping

Extent capping is opt-in and operates independently on each exported isosurface:

```python
from gempy_engine.core.data import MeshExtentCapping

options.evaluation_options.mesh_extraction_extent_capping = (
    MeshExtentCapping.SCALAR_LESS_EQUAL
)
```

The equivalent string is `"scalar_less_equal"`; the default is `"none"`.
The inside convention is `scalar <= isovalue`. Nested surfaces therefore produce
overlapping solids, not a partition into geological unit volumes.

## Geometry

The capper constructs three connected pieces:

1. The ordinary interior dual-contouring triangles.
2. Transition triangles from boundary-cell QEF vertices to the extent contour,
   including primal-edge triangles between adjacent boundary cells.
3. Flat, outward-facing triangles covering inside regions of the six box faces.

Boundary squares use a consistent diagonal and piecewise-linear triangle
clipping. Lattice points and edge intersections have shared integer identities,
including intersections on face diagonals and on box edges and corners.

Only cap components connected to a contour with an available, contained QEF
vertex are retained. An enclosed isosurface receives no disconnected box shell.
If no isosurface intersects the box, no box shell is manufactured, even when
the whole box satisfies the inside condition. This is closure of existing
isosurfaces, not unconditional extraction of the complete clipped sublevel set.

`RegularGrid.orthogonal_extent` is the single requested extent used for grid
sampling, refinement, and capping. All octree levels retain that exact extent,
whether capping is enabled or disabled. There is no conditional private root-grid
resampling or extent translation. Caps are generated in engine coordinates,
before downstream output transforms. Original vertex indices are preserved and
cap vertices are appended.

Enabled extraction and cap clipping share strict scalar crossing rules. An
endpoint equal to the isovalue is inside; an entire iso-valued edge is not a
crossing. Interpolation parameters are clamped and snapped within `1e-12` of an
endpoint. Nonfinite scalar inputs are rejected with a diagnostic. QEF mass
points use edge-validity masks globally, including genuine zero coordinates,
regardless of capping mode. Enabled QEF solving avoids PyTorch's additional
origin-centered regularization.

With capping disabled, no boundary evaluation or cap construction is performed.
Exact grid sampling and zero-coordinate handling apply in both modes; uncapped
mesh arrays are not guaranteed to match legacy output. Enabled interior geometry
can differ because its crossing and QEF-solving conventions are stricter.

## Best-Effort Contract

Inspect `mesh.capping_report["closure_success"]` before treating a result as a
closed solid. Unsupported geometry emits `RuntimeWarning` and returns the
available mesh; it is not silently declared successful.

- Fault stacks, fault-affected stacks, and stacks with removed extraction-mask
  cells currently skip capping. Scalar samples alone do not encode enough cut
  ownership to safely prevent bridging intentional fault or relation boundaries.
  These meshes retain their extracted geometry and report `skipped_reason`.
- Missing refinement support is never repaired by extending a cap into the
  interior. Missing boundary QEFs are reported and transitions are omitted.
- Escaped QEF vertices are reported and excluded from transition construction;
  this implementation does not introduce a bounded QEF solver.
- Exact ties and multiple contour components in one cell remain best effort.
  Edge-incidence, directed-winding, and vertex-link audits detect many failures,
  but there is no geometric self-intersection test.
- New duplicate or zero-area triangles are removed. Invalid original geometry
  is preserved and reported unsuccessful rather than silently repaired.
- `GEMPY_SKIP_TRIANGULATION` also skips capping and boundary evaluation.

The full issue's general fault/mask-aware capping and all ambiguous topology
cases are not implemented by this first version.

## Reports and Metadata

Each mesh exposes `stack_index`, `surface_index`, `exported_surface_index`,
`isovalue`, and, when enabled, `inside_convention`.

The capping report includes before/after open-edge counts, inferred non-extent
openings, added cap and transition triangle counts, removed triangle counts,
missing/escaped QEF counts, manifoldness and winding checks, maximum cap-plane
error, evaluated boundary-point count, and the original vertex count.

`watertight` describes topology; `closure_success` additionally checks finite,
nondegenerate geometry and unresolved construction diagnostics. Neither proves
absence of self-intersections. Physical-opening classification for original
QEF edges uses owning-cell membership, not the interior QEF positions; it is
therefore a cell-level inference rather than proof of the opening's cause.
Cap-plane checks use `64 * float64_epsilon * max(1, abs(extent))` tolerance.

`vertices_tensor` remains the pre-overlap differentiable QEF snapshot. Final
triangle indices reference `mesh.vertices`, **not** `vertices_tensor`. Appended
cap geometry and topology are NumPy postprocessing and are not differentiable.

## Cost

One unique finest-level six-face lattice is shared across surfaces. Boundary
scalar values are evaluated once per stack per batch and reused across that
stack's isosurfaces. The API module `API/dual_contouring/extent_capping.py`
orchestrates capping and restores the grid and stack cursor after boundary
evaluation, including failures. `evaluation_chunk_size` bounds the boundary point batch;
the existing evaluator also applies its own kernel-workload chunking.

Boundary storage scales as `O(nx*ny + nx*nz + ny*nz)`, but Python dictionaries,
triangle connectivity, and topology audits have substantial additional memory
cost. This is a correctness-first implementation: there is no adaptive cap
simplification, cross-call cache, or reuse of existing corner samples.
Large-depth GPU and production-scale memory benchmarks remain outstanding.
