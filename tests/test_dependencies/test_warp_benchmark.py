"""
GPU benchmark: Warp SDF computation scaling with mesh triangle count and
evaluation point count, across all three mesh_query_point_sign_* variants.

Run with:
    pytest tests/test_dependencies/test_warp_benchmark.py --benchmark-only \
        --benchmark-min-rounds=3 --benchmark-warmup=off --benchmark-sort=fullname
"""

import numpy as np
import pytest

pytest.importorskip("warp")
import warp as wp

wp.init()

requires_cuda = pytest.mark.skipif(
    not wp.is_cuda_available(),
    reason="CUDA device required for GPU benchmark",
)
requires_matplotlib = pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# region: Shared kernels (self-contained copies from test_warp.py)
# ---------------------------------------------------------------------------

@wp.kernel
def _sdf_normal(
    mesh_id: wp.uint64,
    mesh_vertices: wp.array(dtype=wp.vec3),
    mesh_indices: wp.array(dtype=wp.int32),
    query_points: wp.array(dtype=wp.vec3),
    out_distances: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    p = query_points[tid]
    hit = wp.mesh_query_point_sign_normal(mesh_id, p, 1.0e6, 1.0e-6)

    i0 = mesh_indices[hit.face * 3]
    i1 = mesh_indices[hit.face * 3 + 1]
    i2 = mesh_indices[hit.face * 3 + 2]
    v0 = mesh_vertices[i0]
    v1 = mesh_vertices[i1]
    v2 = mesh_vertices[i2]

    w = 1.0 - hit.u - hit.v
    closest = v0 * hit.u + v1 * hit.v + v2 * w

    diff = p - closest
    dist = wp.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])

    out_distances[tid] = wp.where(hit.sign < 0.0, -dist, dist)


@wp.kernel
def _sdf_parity(
    mesh_id: wp.uint64,
    mesh_vertices: wp.array(dtype=wp.vec3),
    mesh_indices: wp.array(dtype=wp.int32),
    query_points: wp.array(dtype=wp.vec3),
    out_distances: wp.array(dtype=wp.float32),
    n_sample: wp.int32,
    perturbation_scale: wp.float32,
):
    tid = wp.tid()
    p = query_points[tid]
    hit = wp.mesh_query_point_sign_parity(mesh_id, p, 1.0e6, n_sample, perturbation_scale)

    i0 = mesh_indices[hit.face * 3]
    i1 = mesh_indices[hit.face * 3 + 1]
    i2 = mesh_indices[hit.face * 3 + 2]
    v0 = mesh_vertices[i0]
    v1 = mesh_vertices[i1]
    v2 = mesh_vertices[i2]

    w = 1.0 - hit.u - hit.v
    closest = v0 * hit.u + v1 * hit.v + v2 * w

    diff = p - closest
    dist = wp.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])

    out_distances[tid] = wp.where(hit.sign < 0.0, -dist, dist)


@wp.kernel
def _sdf_winding(
    mesh_id: wp.uint64,
    mesh_vertices: wp.array(dtype=wp.vec3),
    mesh_indices: wp.array(dtype=wp.int32),
    query_points: wp.array(dtype=wp.vec3),
    out_distances: wp.array(dtype=wp.float32),
    accuracy: wp.float32,
    threshold: wp.float32,
):
    tid = wp.tid()
    p = query_points[tid]
    hit = wp.mesh_query_point_sign_winding_number(mesh_id, p, 1.0e6, accuracy, threshold)

    i0 = mesh_indices[hit.face * 3]
    i1 = mesh_indices[hit.face * 3 + 1]
    i2 = mesh_indices[hit.face * 3 + 2]
    v0 = mesh_vertices[i0]
    v1 = mesh_vertices[i1]
    v2 = mesh_vertices[i2]

    w = 1.0 - hit.u - hit.v
    closest = v0 * hit.u + v1 * hit.v + v2 * w

    diff = p - closest
    dist = wp.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])

    out_distances[tid] = wp.where(hit.sign < 0.0, -dist, dist)

# endregion


# ---------------------------------------------------------------------------
# region: Launchers
# ---------------------------------------------------------------------------

def _launch_normal(mesh_id, vertices, indices, pts_host, out=None):
    n = len(pts_host)
    query_wp = wp.array(pts_host.astype(np.float32), dtype=wp.vec3)
    if out is None:
        out = wp.zeros(n, dtype=wp.float32)
    wp.launch(_sdf_normal, dim=n,
              inputs=[mesh_id, vertices, indices, query_wp, out])
    wp.synchronize()
    return out.numpy()


def _launch_parity(mesh_id, vertices, indices, pts_host, n_sample=3, perturbation_scale=0.1, out=None):
    n = len(pts_host)
    query_wp = wp.array(pts_host.astype(np.float32), dtype=wp.vec3)
    if out is None:
        out = wp.zeros(n, dtype=wp.float32)
    wp.launch(_sdf_parity, dim=n,
              inputs=[mesh_id, vertices, indices, query_wp, out, n_sample, perturbation_scale])
    wp.synchronize()
    return out.numpy()


def _launch_winding(mesh_id, vertices, indices, pts_host, accuracy=2.0, threshold=0.5, out=None):
    n = len(pts_host)
    query_wp = wp.array(pts_host.astype(np.float32), dtype=wp.vec3)
    if out is None:
        out = wp.zeros(n, dtype=wp.float32)
    wp.launch(_sdf_winding, dim=n,
              inputs=[mesh_id, vertices, indices, query_wp, out, accuracy, threshold])
    wp.synchronize()
    return out.numpy()


_LAUNCHERS = {"normal": _launch_normal, "parity": _launch_parity, "winding": _launch_winding}

# endregion


# ---------------------------------------------------------------------------
# region: Mesh builder — scalable flat grid
# ---------------------------------------------------------------------------

def _make_grid_mesh(nx: int, ny: int, extent=(-1.0, 1.0, -1.0, 1.0), z=0.0,
                    support_winding_number=False):
    """Flat XY grid mesh at given z, with (nx-1)*(ny-1)*2 triangles."""
    x_min, x_max, y_min, y_max = extent
    x_vals = np.linspace(x_min, x_max, nx, dtype=np.float32)
    y_vals = np.linspace(y_min, y_max, ny, dtype=np.float32)

    vertices = np.empty((nx * ny, 3), dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            vertices[j * nx + i] = (x_vals[i], y_vals[j], z)

    indices = np.empty(((nx - 1) * (ny - 1) * 2, 3), dtype=np.int32)
    k = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = j * nx + i
            v10 = j * nx + (i + 1)
            v01 = (j + 1) * nx + i
            v11 = (j + 1) * nx + (i + 1)
            indices[k] = (v00, v10, v11)
            k += 1
            indices[k] = (v00, v11, v01)
            k += 1

    ntri = k
    verts_wp = wp.array(vertices, dtype=wp.vec3)
    idxs_wp = wp.array(indices.ravel(), dtype=wp.int32)
    mesh = wp.Mesh(points=verts_wp, indices=idxs_wp,
                   support_winding_number=support_winding_number)
    return mesh, verts_wp, idxs_wp, ntri

# endregion


# ---------------------------------------------------------------------------
# region: Eval-point builder — XZ cross-section
# ---------------------------------------------------------------------------

def _make_xz_cross_section_points(n: int, y_slice=0.0,
                                   x_range=(-1.5, 1.5), z_range=(-0.6, 1.2)):
    """n × n evaluation grid in the XZ plane at a fixed y slice."""
    x = np.linspace(x_range[0], x_range[1], n, dtype=np.float32)
    z = np.linspace(z_range[0], z_range[1], n, dtype=np.float32)
    xx, zz = np.meshgrid(x, z)
    return np.column_stack([xx.ravel(), np.full(xx.size, y_slice, dtype=np.float32), zz.ravel()])

# endregion


# ---------------------------------------------------------------------------
# region: Benchmarks
# ---------------------------------------------------------------------------

MESH_SIZES = [
    pytest.param(17, 512, id="0.5Ktri"),
    pytest.param(65, 8192, id="8Ktri"),
    pytest.param(129, 32768, id="32Ktri"),
    pytest.param(257, 131072, id="130Ktri"),
    pytest.param(513, 524288, id="500Ktri"),
]
EVAL_SIZES = [
    pytest.param(16, id="256pt"),
    pytest.param(32, id="1Kpt"),
    pytest.param(64, id="4Kpt"),
    pytest.param(128, id="16Kpt"),
    pytest.param(256, id="65Kpt"),
]
METHODS = [
    pytest.param("normal", id="normal"),
    pytest.param("parity", id="parity"),
    pytest.param("winding", id="winding"),
]


@requires_cuda
@pytest.mark.benchmark(min_rounds=3, disable_gc=True, warmup=False)
@pytest.mark.parametrize("grid_size,tri_count", MESH_SIZES)
@pytest.mark.parametrize("sdf_method", METHODS)
def test_benchmark_sdf_by_triangle_count(benchmark, grid_size, tri_count, sdf_method, tmp_path):
    """Scale mesh triangle count with a fixed 64×64 (4096 pt) evaluation grid."""
    support_wn = sdf_method == "winding"
    mesh, verts, idxs, ntri = _make_grid_mesh(grid_size, grid_size,
                                              support_winding_number=support_wn)
    assert ntri == tri_count, f"Expected {tri_count} triangles, got {ntri}"

    n_eval = 64
    pts = _make_xz_cross_section_points(n_eval, y_slice=0.0,
                                         x_range=(-1.5, 1.5), z_range=(-0.6, 1.2))
    launcher = _LAUNCHERS[sdf_method]

    _ = launcher(mesh.id, verts, idxs, pts)  # warmup — compile kernels

    def _run():
        return launcher(mesh.id, verts, idxs, pts)

    field = benchmark(_run)

    benchmark.extra_info["method"] = sdf_method
    benchmark.extra_info["ntri"] = ntri
    benchmark.extra_info["n_eval"] = len(pts)

    assert np.all(np.isfinite(field))
    _save_verification_plot(pts, field, n_eval, n_eval, sdf_method, tri_count,
                            tmp_path, suffix="tri")


@requires_cuda
@pytest.mark.benchmark(min_rounds=3, disable_gc=True, warmup=False)
@pytest.mark.parametrize("n_eval", EVAL_SIZES)
@pytest.mark.parametrize("sdf_method", METHODS)
def test_benchmark_sdf_by_eval_count(benchmark, n_eval, sdf_method, tmp_path):
    """Scale evaluation point count with a fixed 257×257 (130K tri) mesh."""
    grid_size = 257
    tri_count = (grid_size - 1) ** 2 * 2

    support_wn = sdf_method == "winding"
    mesh, verts, idxs, ntri = _make_grid_mesh(grid_size, grid_size,
                                              support_winding_number=support_wn)
    assert ntri == tri_count

    pts = _make_xz_cross_section_points(n_eval, y_slice=0.0,
                                         x_range=(-1.5, 1.5), z_range=(-0.6, 1.2))
    launcher = _LAUNCHERS[sdf_method]

    _ = launcher(mesh.id, verts, idxs, pts)  # warmup — compile kernels

    def _run():
        return launcher(mesh.id, verts, idxs, pts)

    field = benchmark(_run)

    benchmark.extra_info["method"] = sdf_method
    benchmark.extra_info["ntri"] = ntri
    benchmark.extra_info["n_eval"] = len(pts)

    assert np.all(np.isfinite(field))
    _save_verification_plot(pts, field, n_eval, n_eval, sdf_method, tri_count,
                            tmp_path, suffix="eval")


@requires_cuda
@pytest.mark.benchmark(min_rounds=3, disable_gc=True, warmup=False)
def test_benchmark_sdf_stress_500Ktri_1Mpt(benchmark, tmp_path):
    """Stress test: 500K‑triangle mesh × 1 M evaluation points, normal method only."""
    grid_size = 501
    tri_count = (grid_size - 1) ** 2 * 2  # 500,000
    n_eval = 1000  # 1,000,000 points

    mesh, verts, idxs, ntri = _make_grid_mesh(grid_size, grid_size, support_winding_number=False)
    assert ntri == tri_count

    pts = _make_xz_cross_section_points(n_eval, y_slice=0.0,
                                         x_range=(-1.5, 1.5), z_range=(-0.6, 1.2))
    launcher = _LAUNCHERS["normal"]

    _ = launcher(mesh.id, verts, idxs, pts)  # warmup / compile

    def _run():
        return launcher(mesh.id, verts, idxs, pts)

    field = benchmark(_run)

    benchmark.extra_info["method"] = "normal"
    benchmark.extra_info["ntri"] = ntri
    benchmark.extra_info["n_eval"] = n_eval * n_eval

    assert np.all(np.isfinite(field))
    _save_verification_plot(pts, field, n_eval, n_eval, "normal", tri_count,
                            tmp_path, suffix="stress")


# endregion


# ---------------------------------------------------------------------------
# region: Verification plot helper
# ---------------------------------------------------------------------------

def _save_verification_plot(pts, field, nx, nz, method, ntri, tmp_path, suffix):
    """Save a quick matplotlib contour to verify the zero-isosurface is correct."""
    import matplotlib.pyplot as plt

    field_2d = field.reshape(nx, nz)
    x = pts[:, 0].reshape(nx, nz)[:, 0]
    z = pts[:, 2].reshape(nx, nz)[0, :]

    fig, ax = plt.subplots(figsize=(6, 4))
    c = ax.contourf(x, z, field_2d.T, levels=30, cmap="RdBu_r")
    ax.contour(x, z, field_2d.T, levels=[0.0], colors="k", linewidths=2.0)
    fig.colorbar(c)

    ax.set_title(f"{method} | {ntri} tri | {len(pts)} pts")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal")

    fname = tmp_path / f"bench_{method}_{ntri}tri_{len(pts)}pt_{suffix}.png"
    fig.savefig(fname, dpi=80)
    plt.close(fig)

# endregion
