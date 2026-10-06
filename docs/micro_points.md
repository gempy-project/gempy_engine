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
nuggets, surface_indices, support_to_engine=None)` to `InterpolationInput(..., micro_points=...)`.
`surface_indices` are zero-based **global** surface indices in structural-frame
order. Inputs have shapes `(N, 3)`, `(N, 3, 3)`, `(N,)`, and `(N,)` respectively.
Internal stack subsets rebase surface indices to the owning stack. Model preparation
validates ownership and supported stack types once, then shares contact queries
across stacks; the authored arrays are not converted or mutated in place.
The matrices transform coordinate differences into each contact's local metric;
GemPy builds them from the inverse support axes after coordinate transforms.
`support_to_engine` is an optional finite invertible 3-by-3 linear map `A` from
the original support/world frame to engine coordinates (`None` means identity).
It is shared by all contacts and carried unchanged through stack subsets.

With `micro_options.align_to_macro = True` (default), each contact's support z
axis is aligned with the **uncorrected** macro field gradient at that contact,
including upstream fault effects. The engine transforms row gradients by
`g_world = g_engine @ A`, recovers the authored world axes
`B_world = inv(A) @ inv(M)` and preserves their three column lengths. A fixed
world reference axis defines the two lateral axes (x except near parallel,
then y); prior authored rotations do not influence the new frame. The fitted
metric is `inv(A @ (frame @ diag(radii)))`. A near-zero gradient leaves the
authored matrix unchanged. Set `align_to_macro = False` to retain authored
orientations and disable alignment writeback.

`Solutions.micro_point_results` is `None` without enabled aligned contacts;
otherwise it is a `MicroPointResults` with `source_indices` (NumPy integer
indices into the root `MicroPoints` rows), `macro_gradients` (engine-frame
uncorrected contact gradients, shape N-by-3), and `anisotropy_matrices`
(fitted contact metrics, shape N-by-3-by-3). Arrays use the active NumPy or
PyTorch backend, preserving PyTorch autograd; only authored contact rows are
included, never macro-preservation centers. Results are taken from the last
octree level, concatenated in stack order. Macro gradients are computed for
alignment even with `compute_scalar_gradient = False`, without adding public
scalar-gradient fields or modifying the caller's options.

When GemPy assigns the solution to `GeoModel.solutions`, it writes the fitted
support axes back to each contact in original model coordinates before reordering
elements. Positions, support-axis lengths, nuggets, and ownership are unchanged.
The updated support transforms are part of the normal `.gempy` serialization;
saving and loading preserves their orientations without serializing `Solutions`.
Recomputation sets an absolute frame rather than accumulating rotations.

The server response ZIP includes `micro_point_support_transforms.npy` (N-by-4-by-4),
`micro_point_element_ids.npy` (N), and `micro_point_row_indices.npy` (N) when
alignment results are present. These contain all contacts' current world-space
supports, including unchanged contacts on disabled stacks. Match each row using
its element ID and zero-based row index within that element, not response order
or an element's position in the structural frame. The arrays are omitted when
no contacts were aligned. Clients can apply these transforms to their dataset.

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
