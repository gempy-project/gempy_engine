import os

import numpy as np

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
    options.evaluation_options.compute_scalar_gradient = True

    sp_internal = surface_points_preprocess(sp, data_descriptor.tensors_structure)
    ori_internal = orientations_preprocess(orientations)

    n_per_surface = data_descriptor.tensors_structure.number_of_points_per_surface
    macro_sp_coords = sp.sp_coords
    macro_ori_positions = orientations.dip_positions

    solver_input = SolverInput(sp_internal, ori_internal, xyz_to_interpolate=None, fault_internal=None)
    macro_weights = _solve_interpolation(solver_input, options.kernel_options)

    # --- target scalars: median macro scalar at original surface points ---
    exported_macro_sp = _eval_at_points(sp_internal, ori_internal, options, macro_weights, macro_sp_coords)
    macro_at_sp = exported_macro_sp.scalar_field
    target_per_surface = [
        float(np.median(macro_at_sp[:n_per_surface[0]])),
        float(np.median(macro_at_sp[n_per_surface[0]:])),
    ]
    print(f"target S0 = {target_per_surface[0]:.3f}  |  target S1 = {target_per_surface[1]:.3f}")

    # --- micro contacts ---
    contacts = np.array([
        [1.0, 3.0], [2.0, 2.5], [3.0, 1.5],
        [0.5, 4.0], [4.0, 0.5], [2.5, 1.0],
    ], dtype=np.float64)
    micro_surface_ids = np.array([1, 1, 0, 1, 0, 0], dtype=int)

    exported_contacts = _eval_at_points(sp_internal, ori_internal, options, macro_weights, contacts)
    macro_values_at_contacts = exported_contacts.scalar_field
    gx = exported_contacts.gx_field
    gy = exported_contacts.gy_field
    macro_gradients = np.column_stack([gx, gy])

    target_values_at_contacts = np.array([target_per_surface[sid] for sid in micro_surface_ids])
    residuals = target_values_at_contacts - macro_values_at_contacts

    for i in range(len(contacts)):
        print(f"  contact {i} (S{micro_surface_ids[i]}): target={target_values_at_contacts[i]:.3f}  "
              f"macro={macro_values_at_contacts[i]:.3f}  residual={residuals[i]:.3f}")

    # --- micro solve ---
    A = compute_anisotropy_matrices_from_gradients(
        contacts, macro_gradients, r_vertical=0.5, r_lateral=5.0,
    )
    micro_kernel_range = 0.5
    micro_weights = solve_micro_weights(contacts, residuals, A, kernel_range=micro_kernel_range, nugget=1e-6)

    # --- grid evaluation ---
    grid_xy = _build_grid_2d((-1, 5), (-1, 5), 40, 40)
    options.evaluation_options.compute_scalar_gradient = False
    macro_fields = _eval_at_points(sp_internal, ori_internal, options, macro_weights, grid_xy)

    micro = options.evaluation_options.micro_anisotropic
    micro.enabled = True
    micro.points = contacts
    micro.weights = micro_weights
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

    # --- verify contact compliance ---
    micro_exported = _eval_at_points(sp_internal, ori_internal, options, macro_weights, contacts)
    options.evaluation_options.micro_anisotropic.enabled = False
    corrected_values = micro_exported.scalar_field
    before_err = target_values_at_contacts - macro_values_at_contacts
    after_err = target_values_at_contacts - corrected_values
    rms_before = np.sqrt(np.mean(before_err ** 2))
    rms_after = np.sqrt(np.mean(after_err ** 2))
    assert rms_after < rms_before, (
        f"Micro correction should reduce contact RMS error. "
        f"Before: {rms_before:.6f}, After: {rms_after:.6f}"
    )
    print(f"RMS before: {rms_before:.6f}, RMS after: {rms_after:.6f}")

    if PLOT or True:
        _plot_results(
            grid_xy, macro_field_2d, micro_field_2d, diff_field,
            contacts, micro_surface_ids,
            macro_values_at_contacts, corrected_values, target_values_at_contacts,
            macro_sp_coords, n_per_surface, macro_ori_positions, micro_kernel_range,
            A, target_per_surface,
        )


# ----------------------------------------------------------------
# plotting
# ----------------------------------------------------------------
def _plot_results(grid_xy, macro_field, micro_field, diff,
                  contacts, micro_surface_ids,
                  macro_vals, corrected_vals, target_vals,
                  macro_sp_coords, n_per_surface, macro_ori_positions,
                  micro_kernel_range, A_matrices, target_per_surface):
    import matplotlib.pyplot as plt

    x = grid_xy[:, 0].reshape(macro_field.shape)
    y = grid_xy[:, 1].reshape(macro_field.shape)
    xlim = (-1, 6)
    ylim = (-1, 5)

    vmin = min(macro_field.min(), micro_field.min())
    vmax = max(macro_field.max(), micro_field.max())
    levels = np.linspace(vmin, vmax, 25)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # --- Top-left: Macro field + macro input + micro contacts ---
    ax = axes[0, 0]
    ax.set_title("Macro scalar field")
    ax.contourf(x, y, macro_field, levels=levels, cmap="viridis", extend="both")

    _draw_macro_input(ax, macro_sp_coords, n_per_surface, macro_ori_positions)

    for sv, label in zip(target_per_surface, ["S0 target", "S1 target"]):
        ax.contour(x, y, macro_field, levels=[sv], colors="white", linewidths=1.5, linestyles="-")
        ax.contour(x, y, macro_field, levels=[sv], colors=["#cccccc" if label.startswith("S0") else "#aaaaaa"],
                   linewidths=2.5, linestyles="--")
    for sid in [0, 1]:
        mask = micro_surface_ids == sid
        if mask.any():
            ax.plot(contacts[mask, 0], contacts[mask, 1], "o",
                    color=_MICRO_SURFACE_COLORS[sid], markersize=8,
                    markeredgecolor="black", label=f"micro contacts S{sid}")
    for i in range(len(contacts)):
        ax.annotate(f"{macro_vals[i]:.2f}", (contacts[i, 0], contacts[i, 1]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7,
                     color=_MICRO_SURFACE_COLORS[micro_surface_ids[i]])

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper right", fontsize=6)
    ax.set_aspect("equal")

    # --- Top-right: Micro field + target contours + anisotropy ellipses ---
    ax = axes[0, 1]
    ax.set_title(f"Micro-adjusted scalar field (range={micro_kernel_range})")
    ax.contourf(x, y, micro_field, levels=levels, cmap="viridis", extend="both")
    for sv in target_per_surface:
        ax.contour(x, y, micro_field, levels=[sv], colors="white", linewidths=1.5, linestyles="-")

    _draw_anisotropy_ellipses(ax, contacts, A_matrices, micro_kernel_range, color="yellow", alpha=0.3)
    _draw_macro_input(ax, macro_sp_coords, n_per_surface, macro_ori_positions)

    for sid in [0, 1]:
        mask = micro_surface_ids == sid
        if mask.any():
            ax.plot(contacts[mask, 0], contacts[mask, 1], "o",
                    color=_MICRO_SURFACE_COLORS[sid], markersize=8,
                    markeredgecolor="black", label=f"micro contacts S{sid}")
    for i in range(len(contacts)):
        ax.annotate(f"{corrected_vals[i]:.2f}", (contacts[i, 0], contacts[i, 1]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7,
                     color=_MICRO_SURFACE_COLORS[micro_surface_ids[i]])

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
    for sid in [0, 1]:
        mask = micro_surface_ids == sid
        if mask.any():
            ax.plot(contacts[mask, 0], contacts[mask, 1], "o",
                    color=_MICRO_SURFACE_COLORS[sid], markersize=8,
                    markeredgecolor="black")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")

    # --- Bottom-right: Residual bar chart ---
    ax = axes[1, 1]
    before_abs = np.abs(macro_vals - target_vals)
    after_abs = np.abs(corrected_vals - target_vals)
    n = len(contacts)
    x_idx = np.arange(n)
    width = 0.35
    colors_before = [_MICRO_SURFACE_COLORS[sid] for sid in micro_surface_ids]
    colors_after = [_MACRO_SURFACE_COLORS[sid] for sid in micro_surface_ids]
    ax.bar(x_idx - width/2, before_abs, width, color=colors_before, label="|macro - target|")
    ax.bar(x_idx + width/2, after_abs, width, color=colors_after, label="|corrected - target|")
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"c{i}\n(S{micro_surface_ids[i]})" for i in range(n)])
    ax.set_title("Contact residual error (abs)")
    ax.legend()

    plt.tight_layout()
    plt.show()


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
