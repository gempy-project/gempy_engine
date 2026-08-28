import copy
import os
from pathlib import Path

import numpy as np
import pytest

from gempy_engine.core.data import TaperType
from gempy_engine.modules.faults.finite_faults import get_ellipsoid_distance, get_local_frame


MODEL_PATH = Path(__file__).with_name("last_model.gempy")


def _load_model():
    gempy = pytest.importorskip("gempy", reason="Loading .gempy files currently requires the GemPy package")
    with pytest.warns(UserWarning, match="still in development"):
        return gempy, gempy.load_model(str(MODEL_PATH))


def _directional_radius(radius, positive: bool) -> float:
    if isinstance(radius, tuple):
        return radius[0 if positive else 1]
    return radius


def _plot_finite_fault_diagnostics(
        model,
        no_fault_model,
        finite_fault,
        normal,
        finite_fault_scalar,
        output_dir: Path,
) -> tuple[Path, ...]:
    try:
        import matplotlib
    except ImportError:
        return ()
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    u, v, w = get_local_frame(normal, angle_deg=finite_fault.rotation_deg)
    center = np.asarray(finite_fault.center)

    max_radius = max(
        _directional_radius(finite_fault.strike_radius, True),
        _directional_radius(finite_fault.strike_radius, False),
        _directional_radius(finite_fault.dip_radius, True),
        _directional_radius(finite_fault.dip_radius, False),
    )
    local_range = np.linspace(-1.15 * max_radius, 1.15 * max_radius, 301)
    strike, dip = np.meshgrid(local_range, local_range)
    plane_points = center + strike[..., None] * u + dip[..., None] * v
    distance = get_ellipsoid_distance(
        points=plane_points.reshape(-1, 3),
        center=center,
        u=u,
        v=v,
        a=finite_fault.strike_radius,
        b=finite_fault.dip_radius,
    ).reshape(strike.shape)
    grid_relative = model.grid.values - center
    grid_strike = grid_relative @ u
    grid_dip = grid_relative @ v
    grid_normal = np.abs(grid_relative @ w)
    section = np.argsort(grid_normal)[:max(1024, len(grid_normal) // 32)]
    fault_element = model.structural_frame.get_group_by_name("fault_series").get_element_by_name("fault")
    surface_relative = fault_element.surface_points.xyz - center

    footprint_path = output_dir / "last_model_computed_finite_fault_taper.png"
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image = ax.scatter(
        grid_strike[section],
        grid_dip[section],
        c=finite_fault_scalar[section],
        s=20,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax.contour(strike, dip, distance, levels=[1.0], colors="red", linewidths=2.0)
    ax.scatter(surface_relative @ u, surface_relative @ v, marker="x", s=80, color="cyan", label="Fault points")
    ax.scatter(0.0, 0.0, marker="+", s=120, color="white", label="Finite-fault center")
    ax.set(
        title=f"{model.meta.name}: computed finite-fault taper near the fault plane",
        xlabel="Local strike coordinate u",
        ylabel="Local dip coordinate v",
        aspect="equal",
    )
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="Slip multiplier")
    fig.savefig(footprint_path, dpi=160)
    plt.close(fig)

    finite_layer = model.structural_frame.get_group_by_name("stratigraphic_series").get_element_by_name("layer")
    no_fault_layer = no_fault_model.structural_frame.get_group_by_name("stratigraphic_series").get_element_by_name("layer")
    finite_relative = finite_layer.vertices - center
    no_fault_relative = no_fault_layer.vertices - center

    mesh_path = output_dir / "last_model_layer_mesh_comparison.png"
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.contour(strike, dip, distance, levels=[1.0], colors="red", linewidths=2.0)
    ax.scatter(
        no_fault_relative @ u,
        no_fault_relative @ v,
        s=8,
        color="black",
        alpha=0.35,
        label="No-fault layer mesh",
    )
    ax.scatter(
        finite_relative @ u,
        finite_relative @ v,
        s=8,
        color="tab:orange",
        alpha=0.5,
        label="Finite-fault layer mesh",
    )
    ax.set(
        title="Layer meshes projected onto the finite-fault plane",
        xlabel="Local strike coordinate u",
        ylabel="Local dip coordinate v",
        aspect="equal",
    )
    ax.legend()
    fig.savefig(mesh_path, dpi=160)
    plt.close(fig)
    return footprint_path, mesh_path


def test_last_model_finite_fault_description_and_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("SET_RAW_SCALAR_FIELDS_IN_SOLUTION", "True")
    gempy, model = _load_model()
    no_fault_model = copy.deepcopy(model)
    fault_group = model.structural_frame.get_group_by_name("fault_series")

    assert fault_group.fault_type is gempy.data.FaultType.FINITE
    assert fault_group.finite_fault_draft is None
    finite_fault = fault_group.faults_input_data.finite_fault
    assert finite_fault.center == pytest.approx((8.73, -4.2, 3.5515034198760986))
    assert finite_fault.strike_radius == pytest.approx((16.22, 6.94))
    assert finite_fault.dip_radius == pytest.approx((13.93, 16.66))
    assert finite_fault.taper is TaperType.QUADRATIC
    assert finite_fault.rotation_deg == 0.0

    from gempy.modules.data_manipulation import input_data_descriptor_from_geo_model

    engine_descriptor = input_data_descriptor_from_geo_model(model)
    engine_finite_fault = engine_descriptor.stack_structure.faults_input_data[0].finite_fault
    assert engine_finite_fault.center == pytest.approx((0.19135310, -0.55017848, -0.11561399))
    assert engine_finite_fault.strike_radius == pytest.approx((0.97543458, 0.41735610))
    assert engine_finite_fault.dip_radius == pytest.approx((0.83771909, 1.00189520))
    assert fault_group.faults_input_data.finite_fault is finite_fault

    fault_element = fault_group.get_element_by_name("fault")
    normal = fault_element.orientations.grads[0]
    u, v, w = get_local_frame(normal, angle_deg=finite_fault.rotation_deg)
    assert np.allclose(np.stack((u, v, w)) @ np.stack((u, v, w)).T, np.eye(3))

    center = np.asarray(finite_fault.center)
    directions = (
        (u, _directional_radius(finite_fault.strike_radius, True)),
        (-u, _directional_radius(finite_fault.strike_radius, False)),
        (v, _directional_radius(finite_fault.dip_radius, True)),
        (-v, _directional_radius(finite_fault.dip_radius, False)),
    )
    for direction, radius in directions:
        points = center + np.array([0.5, 1.0, 1.01])[:, None] * radius * direction
        slip = finite_fault.calculate_slip(points, normal)
        assert slip[0] == pytest.approx((1.0 - 0.5 ** 2) ** 2)
        assert slip[1] == pytest.approx(0.0, abs=1e-28)
        assert slip[2] == 0.0

    grid_distance = get_ellipsoid_distance(
        points=model.grid.values,
        center=center,
        u=u,
        v=v,
        a=finite_fault.strike_radius,
        b=finite_fault.dip_radius,
    )
    grid_slip = finite_fault.calculate_slip(model.grid.values, normal)
    assert np.any(grid_distance < 1.0)
    assert np.any(grid_distance >= 1.0)
    assert np.all(grid_slip[grid_distance >= 1.0] == 0.0)

    no_fault_model.structural_frame.get_group_by_name("fault_series").fault_relations = (
        gempy.data.FaultsRelationSpecialCase.OFFSET_NONE
    )
    gempy.compute_model(no_fault_model)
    gempy.compute_model(model)
    computed_finite_fault_scalar = model.solutions.raw_arrays.finite_fault_scalar_field_matrix[0]
    assert computed_finite_fault_scalar.shape == (len(model.grid.values),)
    assert np.any(computed_finite_fault_scalar > 0.0)
    assert np.any(computed_finite_fault_scalar == 0.0)
    assert np.all(computed_finite_fault_scalar >= 0.0)

    output_dir = Path(os.environ.get("FINITE_FAULT_PLOT_DIR", tmp_path))
    plot_paths = _plot_finite_fault_diagnostics(
        model=model,
        no_fault_model=no_fault_model,
        finite_fault=finite_fault,
        normal=normal,
        finite_fault_scalar=computed_finite_fault_scalar,
        output_dir=output_dir,
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in plot_paths)
