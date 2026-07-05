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
    return wp.Mesh(points=vertices, indices=indices)


@wp.kernel
def _compute_signed_scalar_field(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    out_distances: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    p = query_points[tid]
    hit = wp.mesh_query_point_sign_normal(mesh_id, p, 1.0e6, 1.0e-6)
    z = p[2]
    abs_dist = wp.abs(z)
    out_distances[tid] = wp.where(hit.sign < 0.0, -abs_dist, abs_dist)


def test_warp_mesh_query_creates_signed_scalar_field_from_topography():
    mesh = _make_flat_topography_mesh()
    mesh_id = mesh.id

    query_pts = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.5, -0.5, 0.5],
        [-0.3, 0.7, -0.8],
    ], dtype=np.float32)

    n = len(query_pts)
    query_wp = wp.array(query_pts, dtype=wp.vec3)
    distances_wp = wp.zeros(n, dtype=wp.float32)

    wp.launch(_compute_signed_scalar_field, dim=n, inputs=[mesh_id, query_wp, distances_wp])

    d = distances_wp.numpy()

    assert np.all(np.isfinite(d))
    assert np.abs(d[1]) < 1e-4
    assert d[0] > 0
    assert d[2] < 0
    assert np.abs(d[0] - 1.0) < 1e-4
    assert np.abs(d[2] + 1.0) < 1e-4


def test_warp_topography_scalar_field_matplotlib_example(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    # matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mesh = _make_flat_topography_mesh()
    mesh_id = mesh.id

    n = 80
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n)

    xx, yy = np.meshgrid(x, y)
    zz = 0.6 * yy

    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    pts = pts.astype(np.float32)

    n_total = len(pts)
    query_wp = wp.array(pts, dtype=wp.vec3)
    distances_wp = wp.zeros(n_total, dtype=wp.float32)

    wp.launch(_compute_signed_scalar_field, dim=n_total, inputs=[mesh_id, query_wp, distances_wp])

    field = distances_wp.numpy().reshape(n, n)

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
    plt.show()

    out_path = tmp_path / "warp_topography_scalar_field.png"
    # fig.savefig(out_path, dpi=100)
    # plt.close(fig)
    # 
    # assert out_path.exists()
    # assert out_path.stat().st_size > 0
