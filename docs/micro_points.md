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
Internal stack subsets rebase surface indices to the owning stack. Model preparation
validates ownership and supported stack types once, then shares contact queries
across stacks; the authored arrays are not converted or mutated in place.
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

Contact targets use the macro field at the first (reference) point of each
surface, not the mean of its macro samples. These target isovalues remain fixed
for segmentation and mesh extraction after correction, including when
`preserve_macro_points = False` allows the macro reference points to move.

The final scalar field is the faulted macro field plus the stack-local micro
deformation. Faults do not mask, displace, or split the micro kernel. Contacts
on fault-affected stratigraphic stacks fit residuals against the macro field
including upstream ordinary or finite faults. When any contact's stack is enabled,
every stack evaluates the same coordinate layout: grid, all macro surface points,
then all authored micro contacts (including contacts on disabled stacks). These
last coordinates are evaluation queries only, not macro interpolation constraints
or public grid cells. The fit reads macro values directly from that evaluation;
there is no separate macro kernel call for the contacts. Upstream fault rows use
the same suffix, including activation and finite-fault projection. Fault-row
minimum and finite-fault normal selection use only the grid and macro surface
point prefix, so appended contacts cannot change the reference frame. Enabled
authored micro points on fault stacks are rejected, whether or not the fault has
upstream dependencies. Disabled authored points on fault stacks are ignored.
The micro fit uses a dense NumPy or PyTorch solve (not PyKeOps); macro PyKeOps
acceleration remains available and micro evaluation is tensor-native (the dense
fit and correction are not PyKeOps-accelerated). Flat stacks
retain fused macro scalar evaluation when possible; gradients use the non-fused
macro evaluator. Nearest-center metric assignment for preserved macro points is
discrete. Finite-fault projection uses NumPy and is not end-to-end differentiable.
Enabled authored micro contacts on external-function stacks remain unsupported.
