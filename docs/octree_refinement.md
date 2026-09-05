# Octree refinement support modes

```python
from gempy_engine.core.data import OctreeRefinementMode

options.evaluation_options.number_octree_levels_surface = 4
options.evaluation_options.octree_refinement_mode = OctreeRefinementMode.PRECISE
```

The mode also accepts the strings `"fast"`, `"balanced"`, and `"precise"`.

| Mode | Support stencil | Isolated interior parent count |
| --- | --- | --- |
| `fast` (default) | No additional support; existing selector | 1 |
| `balanced` | Six face neighbors | 7 |
| `precise` | 26 face, edge, and corner neighbors | 27 |

This setting does not change octree depth, minimum-level refinement, curvature
thresholds, or categorical surface selection. One dilation of their combined
selection is applied per transition, only while generating surface extraction
levels and only when mesh extraction is enabled. Support cells never seed another
dilation in the same transition. Each selected parent generates eight children.

`balanced` is a cost/coverage compromise, **not 2:1 octree balancing**. It does not
provide diagonal support around primal edges. `precise` supplies the full touching
neighborhood among existing sparse cells, clipped to the physical domain. Neither
mode guarantees discovery of unsampled features or watertight surfaces. A surface
newly detected at the outer edge of an earlier support band can still request an
absent branch; this produces a warning rather than silently claiming closure.
Physical extent capping and extraction-mask stitching are separate concerns.

## Diagnostics

With `options.debug = True` (or evaluation `verbose`), generated octree grids expose
`refinement_debug`: primary surface, additional, and support-only parent masks;
their counts; final count; closure multiplier; generated children; and unique
missing in-domain support requests. These masks index the **previous** generation,
like `active_cells`. Counts for surface and additional selection may overlap.

For non-fast modes, or when `options.debug` is true, each extracted mesh exposes
`support_report`. This CPU diagnostic enumerates unique sampled sign-changing
primal edges before extraction masking. Missing incident cells are classified as
physical, mask-removed, or never generated. Records include edge direction and
integer coordinate, missing coordinates, and the last existing ancestor level.
An edge can have more than one boundary classification. Internal refinement
failures emit a warning; tests can assert the corresponding count is zero.

The report checks true sign changes, not the extractor's slightly extrapolated
edge-intersection tolerance. It does not audit later fault/overlap triangle
removal or all final mesh edge incidences. Debug reports and coordinate sets add
CPU memory/time overhead, especially for many surfaces.

Integer extraction coordinates now use signed int64 and theoretical domain
bounds rather than byte packing and maximum active coordinates. Normal fast-mode
selection is unchanged; meshes affected by the old coordinate/bounds defects can
change even in fast mode.

## Performance

For an interior planar one-cell selection, the full halo approaches 3x the parent
count; isolated selections can reach 27x (7x in balanced mode). Each selected
parent contributes eight centers and 64 stored corner rows at the next level.
No corner deduplication or interpolation caching is introduced here.

A NumPy float64 lookup-only smoke benchmark on a dense `32 x 32 x 32` lattice
gave the following counts (not an end-to-end interpolation benchmark):

| Selection | Primary | Balanced final | Precise final |
| --- | ---: | ---: | ---: |
| Plane `ix == 16` | 1,024 | 3,072 (3x) | 3,072 (3x) |
| Sphere band `abs(norm(coord - 15.5) - 9) < 0.5` | 1,032 | 2,696 (2.61x) | 4,024 (3.90x) |
| Isolated seeds `all(coord % 4 == 2)` | 512 | 3,584 (7x) | 13,824 (27x) |

Single-call lookup times on the development machine were 0.36–0.63 ms for
balanced and 1.05–1.68 ms for precise; these are indicative, not performance
thresholds or measurements of interpolation/reporting overhead.

The default remains fast. Representative curved, multi-stack, faulted, and GPU
time/memory benchmarks are still needed before recommending a different default.
