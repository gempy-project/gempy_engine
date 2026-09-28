# Authored micro points

Micro contacts add a local anisotropic scalar-field correction after the ordinary
macro cokriging solve. They do not enter the macro covariance system. The feature
is opt-in: authored points are ignored with a warning while
`options.micro_options.enabled` is `False`.

For a GemPy `GeoModel`, assign contacts to the owning structural element. Each
4-by-4 support transform has its contact position in the last column and its
three support axes in the upper-left 3-by-3 block:

```python
import numpy as np
import gempy as gp

support = np.eye(4)[None].copy()
support[0, :3, 3] = [0.5, 0.5, 0.52]
support[0, :3, :3] = np.diag([0.5, 0.5, 0.5])
element = model.structural_frame.structural_groups[0].elements[0]
element.micro_points = gp.data.MicroPointsTable.from_transforms(
    support, names=element.name
)
model.interpolation_options.micro_options.enabled = True
gp.compute_model(model)
```

Alternatively, engine callers pass `MicroPoints(points, anisotropy_matrices,
nuggets, surface_indices)` to `InterpolationInput(..., micro_points=...)`.
`surface_indices` are zero-based **global** surface indices in structural-frame
order. Inputs have shapes `(N, 3)`, `(N, 3, 3)`, `(N,)`, and `(N,)` respectively.
The matrices transform coordinate differences into each contact's local metric;
GemPy builds them from the inverse support axes after coordinate transforms.

Configure `options.micro_options.kernel_type` (`"exponential"`, `"matern_3_2"`,
or `"matern_5_2"`), `kernel_range`, `nugget`, `strength`, and
`preserve_macro_points` as needed. With macro preservation enabled (the default),
surface points become additional zero-correction constraints. Micro weights are
fitted separately per stack at each interpolation level; neither inputs nor
options store fitted state. Set `enabled = False` to deactivate without removing
authored contacts. PyTorch fits preserve gradients through coordinates, matrices,
and contact nuggets; nearest metric selection for macro constraints is discrete.

Fault-coupled and external-function stacks do not support enabled authored micro
contacts. Flat stacks retain fused macro scalar evaluation when possible; gradient
evaluation uses the non-fused macro evaluator.
