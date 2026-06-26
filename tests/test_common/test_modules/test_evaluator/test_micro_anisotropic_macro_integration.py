import os

import numpy as np

from gempy_engine.core.data import Orientations
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.API.interp_single._interp_scalar_field import (
    _solve_interpolation,
    _evaluate_sys_eq,
)
from gempy_engine.modules.data_preprocess._input_preparation import (
    surface_points_preprocess,
    orientations_preprocess,
)
from gempy_engine.modules.evaluator.micro_anisotropic_evaluator import (
    compute_anisotropy_matrices_from_gradients,
    solve_micro_weights,
)

PLOT = os.getenv("GEMPY_PLOT_MICRO", "") == "1"

_MICRO_SURFACE_COLORS = {0: "#00bfff", 1: "#ff6b35"}
_MACRO_SURFACE_COLORS = {0: "#0099cc", 1: "#cc5500"}


def _build_grid_2d(x_range, y_range, nx, ny):
    x = np.linspace(*x_range, nx)
    y = np.linspace(*y_range, ny)
    xv, yv = np.meshgrid(x, y)
    return np.column_stack([xv.ravel(), yv.ravel()])


def _eval_at_points(sp_internal, ori_internal, options, weights, xyz):
    eval_in = SolverInput(sp_internal, ori_internal, xyz_to_interpolate=xyz, fault_internal=None)
    options.evaluation_options.compute_scalar_gradient = True
    return _evaluate_sys_eq(eval_in, weights, options)


def test_micro_correction_moves_contacts_closer_to_target(simple_model_2):
    sp, orientations, options, data_descriptor = simple_model_2
    orientations.dip_positions  = np.array([[ 0.,  4.], [ 4., 1.]])
    orientations.dip_gradients = np.array([[ -.2,  .8], [ 0, 1.]])
    options.kernel_options.range = 20
    
    options.evaluation_options.compute_scalar_gradient = True

    sp_internal = surface_points_preprocess(sp, data_descriptor.tensors_structure)
    ori_internal = orientations_preprocess(orientations)

    n_per_surface = data_descriptor.tensors_structure.number_of_points_per_surface
    macro_sp_coords = sp.sp_coords
    macro_ori_positions = orientations.dip_positions
    macro_sp_surface_ids = np.concatenate([
        np.full(n_per_surface[0], 0, dtype=int),
        np.full(n_per_surface[1], 1, dtype=int),
    ])

    solver_input = SolverInput(sp_internal, ori_internal, xyz_to_interpolate=None, fault_internal=None)
    macro_weights = _solve_interpolation(solver_input, options.kernel_options)

    # --- target scalars: median macro scalar at original surface points ---
    exported_macro_sp = _eval_at_points(sp_internal, ori_internal, options, macro_weights, macro_sp_coords)
    macro_at_sp = exported_macro_sp.scalar_field
    macro_sp_gx = exported_macro_sp.gx_field
    macro_sp_gy = exported_macro_sp.gy_field
    macro_sp_gradients = np.column_stack([macro_sp_gx, macro_sp_gy])
    target_per_surface = [
        float(np.median(macro_at_sp[:n_per_surface[0]])),
        float(np.median(macro_at_sp[n_per_surface[0]:])),
    ]
    print(f"target S0 = {target_per_surface[0]:.3f}  |  target S1 = {target_per_surface[1]:.3f}")

    # --- micro contacts ---
    contacts = np.array([
        [1.0, 2.3], [2.0, 2.5], [3.0, 1.5],
        [0.5, 1.8], [1.4, 0.2], [2.5, 1.0],
    ], dtype=np.float64)
    contact_surface_ids = np.array([1, 1, 0, 1, 0, 0], dtype=int)

    exported_contacts = _eval_at_points(sp_internal, ori_internal, options, macro_weights, contacts)
    macro_values_at_contacts = exported_contacts.scalar_field
    contact_gx = exported_contacts.gx_field
    contact_gy = exported_contacts.gy_field
    contact_gradients = np.column_stack([contact_gx, contact_gy])

    target_values_at_contacts = np.array([target_per_surface[sid] for sid in contact_surface_ids])
    contact_residuals = target_values_at_contacts - macro_values_at_contacts

    for i in range(len(contacts)):
        print(f"  contact {i} (S{contact_surface_ids[i]}): target={target_values_at_contacts[i]:.3f}  "
              f"macro={macro_values_at_contacts[i]:.3f}  residual={contact_residuals[i]:.3f}")

    # --- build augmented micro system (Option 3): contacts + macro points as zero constraints ---
    constraint_points = np.vstack([contacts, macro_sp_coords])
    constraint_gradients = np.vstack([contact_gradients, macro_sp_gradients])
    constraint_residuals = np.concatenate([
        contact_residuals,
        np.zeros(len(macro_sp_coords)),
    ])
    n_contacts = len(contacts)
    n_macro = len(macro_sp_coords)

    micro_kernel_range = 0.5
    A = compute_anisotropy_matrices_from_gradients(
        constraint_points, constraint_gradients, r_vertical=.5, r_lateral=5.0,
    )
    all_weights = solve_micro_weights(constraint_points, constraint_residuals, A,
                                      kernel_range=micro_kernel_range, nugget=1e-6)

    print(f"  micro weights: contacts {np.array2string(all_weights[:n_contacts], precision=3)},  "
          f"macro {np.array2string(all_weights[n_contacts:], precision=3)}")

    # --- grid evaluation ---
    grid_xy = _build_grid_2d((-1, 5), (-1, 5), 40, 40)
    options.evaluation_options.compute_scalar_gradient = False
    macro_fields = _eval_at_points(sp_internal, ori_internal, options, macro_weights, grid_xy)

    micro = options.evaluation_options.micro_anisotropic
    micro.enabled = True
    micro.points = constraint_points
    micro.weights = all_weights
    micro.anisotropy_matrices = A
    micro.kernel_range = micro_kernel_range

    micro_fields = _eval_at_points(sp_internal, ori_internal, options, macro_weights, grid_xy)

    macro_field_2d = macro_fields.scalar_field.reshape(40, 40)
    micro_field_2d = micro_fields.scalar_field.reshape(40, 40)
    diff_field = micro_field_2d - macro_field_2d

    assert np.all(np.isfinite(micro_field_2d)), "Micro field is not finite"
    assert np.all(np.isfinite(diff_field)), "Diff field is not finite"
    max_abs_diff = np.max(np.abs(diff_field))
    assert max_abs_diff > 1e-6, f"Micro correction should produce nonzero change, got max abs diff = {max_abs_diff}"

    # --- contact compliance ---
    micro_exported = _eval_at_points(sp_internal, ori_internal, options, macro_weights, contacts)
    options.evaluation_options.micro_anisotropic.enabled = False
    corrected_contacts = micro_exported.scalar_field
    rms_before = np.sqrt(np.mean(contact_residuals ** 2))
    rms_after = np.sqrt(np.mean((target_values_at_contacts - corrected_contacts) ** 2))
    assert rms_after < rms_before, (
        f"Micro correction should reduce contact RMS error. "
        f"Before: {rms_before:.6f}, After: {rms_after:.6f}"
    )

    # --- macro point preservation ---
    options.evaluation_options.compute_scalar_gradient = False
    micro.enabled = True  # re-enable for this eval
    macro_after_exported = _eval_at_points(sp_internal, ori_internal, options, macro_weights, macro_sp_coords)
    micro.enabled = False
    macro_after_sp = macro_after_exported.scalar_field
    macro_drift = np.abs(macro_after_sp - macro_at_sp)
    max_macro_drift = np.max(macro_drift)
    mean_macro_drift = np.mean(macro_drift)
    assert max_macro_drift < 1.0, (
        f"Macro points shifted too much by micro correction. "
        f"Max drift: {max_macro_drift:.4f}, Mean: {mean_macro_drift:.4f}"
    )

    print(f"RMS before: {rms_before:.6f}, RMS after: {rms_after:.6f}")
    print(f"Macro point drift — max: {max_macro_drift:.4f}, mean: {mean_macro_drift:.4f}")

    if PLOT:
        _plot_results(
            grid_xy, macro_field_2d, micro_field_2d, diff_field,
            contacts, contact_surface_ids,
            macro_values_at_contacts, corrected_contacts, target_values_at_contacts,
            macro_sp_coords, macro_sp_surface_ids, n_per_surface, macro_ori_positions,
            micro_kernel_range, A, target_per_surface,
            n_contacts, macro_before=macro_at_sp, macro_after=macro_after_sp,
        )


# ----------------------------------------------------------------
# plotting
# ----------------------------------------------------------------
def _plot_results(grid_xy, macro_field, micro_field, diff,
                  contacts, contact_surface_ids,
                  macro_vals, corrected_vals, target_vals,
                  macro_sp_coords, macro_sp_surface_ids, n_per_surface,
                  macro_ori_positions, micro_kernel_range, A_matrices,
                  target_per_surface, n_contacts,
                  macro_before=None, macro_after=None):
    import matplotlib.pyplot as plt

    x = grid_xy[:, 0].reshape(macro_field.shape)
    y = grid_xy[:, 1].reshape(macro_field.shape)
    xlim = (-1, 6)
    ylim = (-1, 5)

    vmin = min(macro_field.min(), micro_field.min())
    vmax = max(macro_field.max(), micro_field.max())
    levels = np.linspace(vmin, vmax, 25)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # --- Top-left: Macro field ---
    ax = axes[0, 0]
    ax.set_title("Macro scalar field")
    ax.contourf(x, y, macro_field, levels=levels, cmap="viridis", extend="both")
    _draw_macro_input(ax, macro_sp_coords, n_per_surface, macro_ori_positions)

    for sv, label in zip(target_per_surface, ["S0 target", "S1 target"]):
        ax.contour(x, y, macro_field, levels=[sv], colors="white", linewidths=1.5, linestyles="-")
    _draw_contacts(ax, contacts, contact_surface_ids, contact_vals=macro_vals)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper right", fontsize=6)
    ax.set_aspect("equal")

    # --- Top-right: Micro field ---
    ax = axes[0, 1]
    ax.set_title(f"Micro-adjusted scalar field (range={micro_kernel_range})")
    ax.contourf(x, y, micro_field, levels=levels, cmap="viridis", extend="both")
    for sv in target_per_surface:
        ax.contour(x, y, micro_field, levels=[sv], colors="white", linewidths=1.5, linestyles="-")

    _draw_anisotropy_ellipses(ax, contacts, A_matrices[:n_contacts], micro_kernel_range,
                               color="yellow", alpha=0.25)
    _draw_macro_input(ax, macro_sp_coords, n_per_surface, macro_ori_positions)
    _draw_contacts(ax, contacts, contact_surface_ids, contact_vals=corrected_vals)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper right", fontsize=6)
    ax.set_aspect("equal")

    # --- Bottom-left: Difference field ---
    ax = axes[1, 0]
    c = ax.contourf(x, y, diff, cmap="RdBu_r", levels=20, extend="both")
    plt.colorbar(c, ax=ax, shrink=0.9)
    ax.set_title("Micro - Macro difference")
    _draw_macro_input(ax, macro_sp_coords, n_per_surface, macro_ori_positions)
    _draw_contacts(ax, contacts, contact_surface_ids)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")

    # --- Bottom-right: Residual bar chart + macro drift ---
    ax = axes[1, 1]
    before_abs = np.abs(macro_vals - target_vals)
    after_abs = np.abs(corrected_vals - target_vals)
    n = len(contacts)
    x_idx = np.arange(n)
    width = 0.35
    colors_before = [_MICRO_SURFACE_COLORS[sid] for sid in contact_surface_ids]
    colors_after = [_MACRO_SURFACE_COLORS[sid] for sid in contact_surface_ids]
    ax.bar(x_idx - width/2, before_abs, width, color=colors_before, label="|macro - target|")
    ax.bar(x_idx + width/2, after_abs, width, color=colors_after, label="|corrected - target|")
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"c{i}\n(S{contact_surface_ids[i]})" for i in range(n)])
    ax.set_title("Contact residual error (abs)")
    ax.legend(loc="upper left", fontsize=7)

    if macro_before is not None and macro_after is not None:
        drift_text = (f"macro pt drift:\n"
                      f"  max: {np.max(np.abs(macro_after - macro_before)):.4f}\n"
                      f"  mean: {np.mean(np.abs(macro_after - macro_before)):.4f}")
        ax.text(0.95, 0.90, drift_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    plt.tight_layout()
    plt.show()


def _draw_contacts(ax, contacts, contact_surface_ids, contact_vals=None):
    for sid in [0, 1]:
        mask = contact_surface_ids == sid
        if mask.any():
            ax.plot(contacts[mask, 0], contacts[mask, 1], "o",
                    color=_MICRO_SURFACE_COLORS[sid], markersize=8,
                    markeredgecolor="black", label=f"micro contacts S{sid}")
    if contact_vals is not None:
        for i in range(len(contacts)):
            ax.annotate(f"{contact_vals[i]:.2f}", (contacts[i, 0], contacts[i, 1]),
                         textcoords="offset points", xytext=(5, 5), fontsize=7,
                         color=_MICRO_SURFACE_COLORS[contact_surface_ids[i]])


def _draw_anisotropy_ellipses(ax, points, A_matrices, kernel_range, color="yellow", alpha=0.3):
    from matplotlib.patches import Ellipse

    for i, p in enumerate(points):
        A = A_matrices[i]
        ATA = A.T @ A
        eigvals, eigvecs = np.linalg.eigh(ATA)
        eigvals = np.maximum(eigvals, 1e-12)
        semi_axes = kernel_range / np.sqrt(eigvals)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ell = Ellipse(
            xy=(p[0], p[1]),
            width=2 * semi_axes[0],
            height=2 * semi_axes[1],
            angle=angle,
            facecolor=color,
            edgecolor="black",
            alpha=alpha,
            linewidth=0.5,
        )
        ax.add_patch(ell)


def _draw_macro_input(ax, macro_sp_coords, n_per_surface, macro_ori_positions):
    s0 = slice(0, n_per_surface[0])
    s1 = slice(n_per_surface[0], n_per_surface[0] + n_per_surface[1])
    ax.plot(macro_sp_coords[s0, 0], macro_sp_coords[s0, 1], "s",
            color=_MACRO_SURFACE_COLORS[0], markersize=5, markeredgecolor="black",
            label=f"macro S0 pts")
    ax.plot(macro_sp_coords[s1, 0], macro_sp_coords[s1, 1], "s",
            color=_MACRO_SURFACE_COLORS[1], markersize=5, markeredgecolor="black",
            label=f"macro S1 pts")
    ax.plot(macro_ori_positions[:, 0], macro_ori_positions[:, 1], "^",
            color="magenta", markersize=6, markeredgecolor="black", label="macro orientations")


# ----------------------------------------------------------------
# 3D integration test
# ----------------------------------------------------------------
def _build_grid_3d(x_range, y_range, z_range, nx, ny, nz):
    x = np.linspace(*x_range, nx)
    y = np.linspace(*y_range, ny)
    z = np.linspace(*z_range, nz)
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])


def test_micro_correction_moves_3d_contacts_closer_to_target(simple_model):
    """3D analog of the 2D integration test.

    Uses the existing simple_model fixture (7 surface points, 2 orientations,
    3D cubic kernel) and exercises the same pipeline: macro solve -> macro scalar/gradient
    at contacts and surface points -> per-surface median target -> augmented micro solve
    with macro zero constraints -> grid evaluation -> assertions.

    The dense NumPy micro solve here is a reference implementation. Production
    path for 3D is intended to be PyKeOps matvec + CG, not dense K assembly.
    """
    sp, orientations, options, data_descriptor = simple_model

    options.evaluation_options.compute_scalar_gradient = True

    from gempy_engine.core.data.internal_structs import SolverInput
    from gempy_engine.API.interp_single._interp_scalar_field import (
        _solve_interpolation,
        _evaluate_sys_eq,
    )
    from gempy_engine.modules.data_preprocess._input_preparation import (
        surface_points_preprocess,
        orientations_preprocess,
    )

    sp_internal = surface_points_preprocess(sp, data_descriptor.tensors_structure)
    ori_internal = orientations_preprocess(orientations)

    n_per_surface = data_descriptor.tensors_structure.number_of_points_per_surface
    macro_sp_coords = sp.sp_coords

    # --- macro solve ---
    solver_input = SolverInput(sp_internal, ori_internal, xyz_to_interpolate=None, fault_internal=None)
    macro_weights = _solve_interpolation(solver_input, options.kernel_options)

    # --- target scalars: median macro scalar at original surface points ---
    def _eval_3d(xyz):
        proxy = SolverInput(sp_internal, ori_internal, xyz_to_interpolate=xyz, fault_internal=None)
        options.evaluation_options.compute_scalar_gradient = True
        return _evaluate_sys_eq(proxy, macro_weights, options)

    exported_macro_sp = _eval_3d(macro_sp_coords)
    macro_at_sp = exported_macro_sp.scalar_field
    macro_sp_gx = exported_macro_sp.gx_field
    macro_sp_gy = exported_macro_sp.gy_field
    macro_sp_gz = exported_macro_sp.gz_field
    macro_sp_gradients = np.column_stack([macro_sp_gx, macro_sp_gy, macro_sp_gz])

    target_per_surface = [float(np.median(macro_at_sp[:n_per_surface[0]]))]
    print(f"target S0 = {target_per_surface[0]:.3f}")

    # --- micro contacts (3D, placed close to the macro interface) ---
    # Contacts are near existing macro surface points but shifted in z
    # (stratigraphic direction) to produce small, realistic residuals.
    contacts = np.array([
        [0.48, 0.49, 0.38],   # near SP1 [0.50, 0.50, 0.375], above
        [0.68, 0.51, 0.48],   # near SP2 [0.667, 0.50, 0.417], above
        [0.60, 0.50, 0.37],   # near SP5 [0.583, 0.50, 0.392], below
        [0.72, 0.49, 0.55],   # near SP6 [0.733, 0.50, 0.500], above
    ], dtype=np.float64)
    contact_surface_ids = np.zeros(len(contacts), dtype=int)  # all target the single surface

    exported_contacts = _eval_3d(contacts)
    macro_values_at_contacts = exported_contacts.scalar_field
    contact_gx = exported_contacts.gx_field
    contact_gy = exported_contacts.gy_field
    contact_gz = exported_contacts.gz_field
    contact_gradients = np.column_stack([contact_gx, contact_gy, contact_gz])

    target_values_at_contacts = np.array([target_per_surface[sid] for sid in contact_surface_ids])
    contact_residuals = target_values_at_contacts - macro_values_at_contacts

    for i in range(len(contacts)):
        print(f"  contact {i}: target={target_values_at_contacts[i]:.3f}  "
              f"macro={macro_values_at_contacts[i]:.3f}  residual={contact_residuals[i]:.3f}")

    # --- augmented micro system: contacts + macro points as zero constraints ---
    constraint_points = np.vstack([contacts, macro_sp_coords])
    constraint_gradients = np.vstack([contact_gradients, macro_sp_gradients])
    constraint_residuals = np.concatenate([
        contact_residuals,
        np.zeros(len(macro_sp_coords)),
    ])
    n_contacts = len(contacts)
    n_macro = len(macro_sp_coords)

    micro_kernel_range = .5  # larger than 2D case because macro domain is ~0.5
    A = compute_anisotropy_matrices_from_gradients(
        constraint_points, constraint_gradients, r_vertical=.3, r_lateral=1.0,
    )
    all_weights = solve_micro_weights(constraint_points, constraint_residuals, A,
                                      kernel_range=micro_kernel_range, nugget=1e-6)

    print(f"  micro weights: contacts {np.array2string(all_weights[:n_contacts], precision=3)},  "
          f"macro {np.array2string(all_weights[n_contacts:], precision=3)}")

    # --- grid evaluation (small 3D dense grid) ---
    grid_xyz = _build_grid_3d(
        (0.25, 0.75), (0.45, 0.55), (0.3, 0.6), 16, 8, 16,
    )

    options.evaluation_options.compute_scalar_gradient = False
    macro_fields = _eval_3d(grid_xyz)

    micro = options.evaluation_options.micro_anisotropic
    micro.enabled = True
    micro.points = constraint_points
    micro.weights = all_weights
    micro.anisotropy_matrices = A
    micro.kernel_range = micro_kernel_range

    micro_fields = _eval_3d(grid_xyz)

    macro_field = macro_fields.scalar_field
    micro_field = micro_fields.scalar_field
    diff_field = micro_field - macro_field

    assert np.all(np.isfinite(micro_field)), "3D micro field is not finite"
    assert np.all(np.isfinite(diff_field)), "3D diff field is not finite"
    max_abs_diff = np.max(np.abs(diff_field))
    assert max_abs_diff > 1e-8, f"3D micro correction should produce nonzero change, got max abs diff = {max_abs_diff}"

    # --- contact compliance ---
    micro.enabled = True
    micro_exported = _eval_3d(contacts)
    micro.enabled = False
    corrected_contacts = micro_exported.scalar_field
    rms_before = np.sqrt(np.mean(contact_residuals ** 2))
    rms_after = np.sqrt(np.mean((target_values_at_contacts - corrected_contacts) ** 2))
    assert rms_after < rms_before, (
        f"3D micro correction should reduce contact RMS error. "
        f"Before: {rms_before:.6f}, After: {rms_after:.6f}"
    )

    # --- macro point preservation ---
    options.evaluation_options.compute_scalar_gradient = False
    micro.enabled = True
    macro_after_exported = _eval_3d(macro_sp_coords)
    micro.enabled = False
    macro_after_sp = macro_after_exported.scalar_field
    macro_drift = np.abs(macro_after_sp - macro_at_sp)
    max_macro_drift = np.max(macro_drift)
    mean_macro_drift = np.mean(macro_drift)
    assert max_macro_drift < 2.0, (  # loose tolerance for first 3D pass
        f"3D macro points shifted too much by micro correction. "
        f"Max drift: {max_macro_drift:.4f}, Mean: {mean_macro_drift:.4f}"
    )

    print(f"3D RMS before: {rms_before:.6f}, RMS after: {rms_after:.6f}")
    print(f"3D macro point drift — max: {max_macro_drift:.4f}, mean: {mean_macro_drift:.4f}")

    if PLOT:
        _plot_3d_results(
            grid_xyz, macro_field, micro_field, diff_field,
            contacts, macro_values_at_contacts, corrected_contacts, target_values_at_contacts,
            macro_sp_coords, n_per_surface, A, target_per_surface, n_contacts,
            micro_kernel_range=micro_kernel_range,
            macro_before=macro_at_sp, macro_after=macro_after_sp,
        )
        _plot_3d_pyvista(
            grid_xyz, macro_field, micro_field,
            contacts, macro_sp_coords, target_per_surface,
        )


# ----------------------------------------------------------------
# 3D plotting (slice-based)
# ----------------------------------------------------------------
def _plot_3d_results(grid_xyz, macro_field, micro_field, diff,
                     contacts, macro_vals_before, macro_vals_after, target_vals,
                     macro_sp_coords, n_per_surface, A_matrices, target_per_surface,
                     n_contacts, micro_kernel_range=1.0, macro_before=None, macro_after=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    # extract grid shape
    x = grid_xyz[:, 0]
    y = grid_xyz[:, 1]
    z = grid_xyz[:, 2]
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    nz = len(np.unique(z))
    shape = (nx, ny, nz)

    macro_3d = macro_field.reshape(shape)
    micro_3d = micro_field.reshape(shape)
    diff_3d = diff.reshape(shape)

    # take a mid-y slice
    y_idx = ny // 2
    x_grid = np.unique(x)
    z_grid = np.unique(z)

    macro_slice = macro_3d[:, y_idx, :].T  # shape (nz, nx)
    micro_slice = micro_3d[:, y_idx, :].T
    diff_slice = diff_3d[:, y_idx, :].T

    vmin = min(macro_slice.min(), micro_slice.min())
    vmax = max(macro_slice.max(), micro_slice.max())
    levels = np.linspace(vmin, vmax, 20)
    xlim = (x_grid[0], x_grid[-1])
    zlim = (z_grid[0], z_grid[-1])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- macro slice ---
    ax = axes[0, 0]
    ax.set_title("3D Macro scalar field (mid-y slice)")
    ax.contourf(x_grid, z_grid, macro_slice, levels=levels, cmap="viridis", extend="both")
    ax.plot(macro_sp_coords[:, 0], macro_sp_coords[:, 2], "ks",
            markersize=6, markeredgecolor="white", label="macro SP (projected)")
    for tv in target_per_surface:
        ax.contour(x_grid, z_grid, macro_slice, levels=[tv], colors="white",
                    linewidths=1.5, linestyles="-")
    ax.plot(contacts[:, 0], contacts[:, 2], "ro", markersize=6,
            markeredgecolor="black", label="contacts (projected)")
    for i in range(n_contacts):
        ax.annotate(f"{macro_vals_before[i]:.2f}", (contacts[i, 0], contacts[i, 2]),
                     textcoords="offset points", xytext=(4, 4), fontsize=6, color="red")
    _draw_3d_anisotropy_disks(ax, contacts, A_matrices[:n_contacts], micro_kernel_range * 0.15,
                               color="yellow", alpha=0.25)
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal")
    ax.legend(fontsize=6)

    # --- micro slice ---
    ax = axes[0, 1]
    ax.set_title("3D Micro-adjusted scalar field (mid-y slice)")
    ax.contourf(x_grid, z_grid, micro_slice, levels=levels, cmap="viridis", extend="both")
    for tv in target_per_surface:
        ax.contour(x_grid, z_grid, micro_slice, levels=[tv], colors="white",
                    linewidths=1.5, linestyles="-")
    ax.plot(macro_sp_coords[:, 0], macro_sp_coords[:, 2], "ks",
            markersize=6, markeredgecolor="white", label="macro SP")
    ax.plot(contacts[:, 0], contacts[:, 2], "ro", markersize=6,
            markeredgecolor="black", label="contacts")
    for i in range(n_contacts):
        ax.annotate(f"{macro_vals_after[i]:.2f}", (contacts[i, 0], contacts[i, 2]),
                     textcoords="offset points", xytext=(4, 4), fontsize=6, color="red")
    _draw_3d_anisotropy_disks(ax, contacts, A_matrices[:n_contacts], micro_kernel_range * 0.15,
                               color="yellow", alpha=0.3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal")
    ax.legend(fontsize=6)

    # --- diff slice ---
    ax = axes[1, 0]
    c = ax.contourf(x_grid, z_grid, diff_slice, cmap="RdBu_r", levels=16, extend="both")
    plt.colorbar(c, ax=ax, shrink=0.9)
    ax.set_title("3D Micro - Macro difference (mid-y slice)")
    ax.plot(macro_sp_coords[:, 0], macro_sp_coords[:, 2], "ks",
            markersize=6, markeredgecolor="white", label="macro SP")
    ax.plot(contacts[:, 0], contacts[:, 2], "ro",
            markersize=6, markeredgecolor="black", label="contacts")
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal")
    ax.legend(fontsize=6)

    # --- residual bar chart ---
    ax = axes[1, 1]
    before_abs = np.abs(macro_vals_before - target_vals)
    after_abs = np.abs(macro_vals_after - target_vals)
    n = n_contacts
    x_idx = np.arange(n)
    width = 0.35
    ax.bar(x_idx - width / 2, before_abs, width, color="#00bfff", label="|macro - target|")
    ax.bar(x_idx + width / 2, after_abs, width, color="#0099cc", label="|corrected - target|")
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"c{i}" for i in range(n)])
    ax.set_title("3D Contact residual error (abs)")
    ax.legend(loc="upper left", fontsize=7)

    if macro_before is not None and macro_after is not None:
        drift_text = (f"macro pt drift:\n"
                      f"  max: {np.max(np.abs(macro_after - macro_before)):.4f}\n"
                      f"  mean: {np.mean(np.abs(macro_after - macro_before)):.4f}")
        ax.text(0.95, 0.85, drift_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    plt.tight_layout()
    plt.show()


def _draw_3d_anisotropy_disks(ax, points, A_matrices, display_radius, color="yellow", alpha=0.3):
    """Project 3D anisotropy ellipsoids onto the xz plane as ellipses.

    A_i is (3,3). The projected 2x2 metric on the xz plane is:
        M = (A_i[:, [0,2]])^T @ A_i[:, [0,2]]
    The ellipse is {v : v^T M v = display_radius^2}.

    display_radius should be a fraction of kernel_range to show the core
    influence region rather than the e^-1 decay contour.
    """
    from matplotlib.patches import Ellipse

    for i, p in enumerate(points):
        A = A_matrices[i]  # (3, 3)
        B = A[:, [0, 2]]   # (3, 2) — x and z columns
        M = B.T @ B         # (2, 2) projected metric
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-12)
        semi_axes = display_radius / np.sqrt(eigvals)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ell = Ellipse(
            xy=(p[0], p[2]),
            width=2 * semi_axes[0],
            height=2 * semi_axes[1],
            angle=angle,
            facecolor=color,
            edgecolor="black",
            alpha=alpha,
            linewidth=0.5,
        )
        ax.add_patch(ell)


def _plot_3d_pyvista(grid_xyz, macro_field, micro_field,
                     contacts, macro_sp_coords, target_per_surface):
    """3D PyVista visualization of the adjusted scalar field.

    Shows: semi-transparent scalar volume, target isosurfaces,
    macro surface points (black), and micro contacts (red).
    """
    try:
        import pyvista as pv
    except ImportError:
        print("pyvista not installed, skipping 3D plot")
        return

    x = np.unique(grid_xyz[:, 0])
    y = np.unique(grid_xyz[:, 1])
    z = np.unique(grid_xyz[:, 2])
    nx, ny, nz = len(x), len(y), len(z)

    xyz_reshaped = grid_xyz.reshape(nx, ny, nz, 3)
    xv = xyz_reshaped[..., 0]
    yv = xyz_reshaped[..., 1]
    zv = xyz_reshaped[..., 2]

    grid = pv.StructuredGrid(xv, yv, zv)

    macro_3d = macro_field.reshape(nx, ny, nz)
    micro_3d = micro_field.reshape(nx, ny, nz)
    grid["macro_scalar"] = macro_3d.ravel(order="F")
    grid["micro_scalar"] = micro_3d.ravel(order="F")

    p = pv.Plotter()
    p.add_text("Micro-adjusted scalar field — volume + isosurfaces", font_size=10)

    p.add_mesh(grid, scalars="micro_scalar", opacity=0.3, cmap="viridis",
               show_edges=False, show_scalar_bar=False)

    for target in target_per_surface:
        contour = grid.contour(isosurfaces=[target], scalars="micro_scalar")
        p.add_mesh(contour, color="white", opacity=0.85, show_edges=False,
                   label=f"target isosurface ({target:.3f})")

    p.add_mesh(pv.PolyData(macro_sp_coords), color="black",
               point_size=12, render_points_as_spheres=True,
               label="macro SP")

    p.add_mesh(pv.PolyData(contacts), color="red",
               point_size=16, render_points_as_spheres=True,
               label="micro contacts")

    p.add_axes()
    p.add_legend()
    p.show()
