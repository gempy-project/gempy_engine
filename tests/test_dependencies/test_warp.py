import numpy as np
import pytest

pytest.importorskip("warp")
import warp as wp

wp.init()


def _make_flat_topography_mesh(extent=(-1.0, 1.0, -1.0, 1.0), z=0.0):
    x_min, x_max, y_min, y_max = extent
    vertices = wp.array(np.array([
        [x_min, y_min, z],
        [x_max, y_min, z],
        [x_max, y_max, z],
        [x_min, y_max, z],
    ], dtype=np.float32), dtype=wp.vec3)
    indices = wp.array(np.array(
        [0, 1, 2, 0, 2, 3], dtype=np.int32
    ), dtype=wp.int32)
    return wp.Mesh(points=vertices, indices=indices), vertices, indices


def _make_tent_topography_mesh():
    vertices = wp.array(np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [ 0.0,  0.0, 0.8],
    ], dtype=np.float32), dtype=wp.vec3)
    indices = wp.array(np.array(
        [0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4], dtype=np.int32
    ), dtype=wp.int32)
    return wp.Mesh(points=vertices, indices=indices), vertices, indices


@wp.kernel
def _compute_signed_scalar_field(
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


def _launch_sdf(mesh_id, vertices, indices, pts_host, out=None):
    n = len(pts_host)
    query_wp = wp.array(pts_host.astype(np.float32), dtype=wp.vec3)
    if out is None:
        out = wp.zeros(n, dtype=wp.float32)
    wp.launch(_compute_signed_scalar_field, dim=n,
              inputs=[mesh_id, vertices, indices, query_wp, out])
    return out.numpy()


def test_warp_mesh_query_creates_signed_scalar_field_from_topography():
    mesh, verts, idxs = _make_flat_topography_mesh()

    query_pts = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.5, -0.5, 0.5],
        [-0.3, 0.7, -0.8],
    ], dtype=np.float32)

    d = _launch_sdf(mesh.id, verts, idxs, query_pts)

    assert np.all(np.isfinite(d))
    assert np.abs(d[1]) < 1e-4
    assert d[0] > 0
    assert d[2] < 0
    assert np.abs(d[0] - 1.0) < 1e-4
    assert np.abs(d[2] + 1.0) < 1e-4


def test_warp_topography_scalar_field_matplotlib_example(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mesh, verts, idxs = _make_flat_topography_mesh()

    n = 80
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n)

    xx, yy = np.meshgrid(x, y)
    zz = 0.6 * yy

    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    field = _launch_sdf(mesh.id, verts, idxs, pts).reshape(n, n)

    fig, ax = plt.subplots()
    c = ax.contourf(x, y, field, levels=20)
    ax.contour(x, y, field, levels=[0.0], colors="k", linewidths=1.5)
    fig.colorbar(c)

    rect = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    ax.plot(rect[:, 0], rect[:, 1], "r-", linewidth=2, label="topography")

    ax.set_title("Signed Scalar Field from Topography Mesh")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    out_path = tmp_path / "warp_topography_scalar_field.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_warp_sdf_perpendicular_to_tent_topography(tmp_path):
    """SDF cross-section perpendicular to a peaked tent topography mesh."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    mesh, verts, idxs = _make_tent_topography_mesh()

    y_slice = 0.2
    n = 120
    x = np.linspace(-1.5, 1.5, n)
    z = np.linspace(-0.6, 1.2, n)

    xx, zz = np.meshgrid(x, z)
    pts = np.column_stack([xx.ravel(), np.full(xx.size, y_slice, dtype=np.float32), zz.ravel()])

    field = _launch_sdf(mesh.id, verts, idxs, pts).reshape(n, n)

    fig, ax = plt.subplots(figsize=(9, 6))
    c = ax.contourf(x, z, field, levels=30, cmap="RdBu_r")
    ax.contour(x, z, field, levels=[0.0], colors="k", linewidths=2.0)
    fig.colorbar(c, label="Signed Distance")

    ax.set_title(f"SDF perpendicular to tent topography  (y = {y_slice})")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal")

    out_path = tmp_path / "warp_tent_sdf_perpendicular.png"
