import numpy as np

from gempy_engine.modules.evaluator.micro_anisotropic_evaluator import (
    evaluate_micro_correction,
    build_micro_covariance,
    solve_micro_weights,
    compute_anisotropy_matrices_from_gradients,
)


def _make_identity_anisotropy(N: int) -> np.ndarray:
    return np.tile(np.eye(3, dtype=np.float64), (N, 1, 1))


# ----------------------------------------------------------------
# evaluate_micro_correction
# ----------------------------------------------------------------

def test_evaluate_single_point_identity():
    """One micro point at origin, identity anisotropy, weight=1.0, range=1.0."""
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    weights = np.array([1.0], dtype=np.float64)
    A = _make_identity_anisotropy(1)

    correction = evaluate_micro_correction(xyz, points, weights, A, kernel_range=1.0)

    np.testing.assert_allclose(correction[0], 1.0, rtol=1e-10)
    np.testing.assert_allclose(correction[1], np.exp(-1.0), rtol=1e-6)
    np.testing.assert_allclose(correction[2], np.exp(-2.0), rtol=1e-6)


def test_evaluate_monotonic_decay():
    """Correction magnitude decreases monotonically with distance."""
    xyz = np.linspace(0, 5, 100)[:, np.newaxis] * np.array([1.0, 0.0, 0.0])
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    weights = np.array([1.0], dtype=np.float64)
    A = _make_identity_anisotropy(1)

    correction = evaluate_micro_correction(xyz, points, weights, A, kernel_range=1.0)

    diffs = np.diff(correction)
    assert np.all(diffs <= 0), "Correction should decrease monotonically with distance"


def test_evaluate_vertical_anisotropy():
    """Vertical anisotropy: decay should be faster in Z than in XY."""
    range_val = 1.0
    r_vertical = 0.5
    r_lateral = 5.0

    xyz = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    weights = np.array([1.0], dtype=np.float64)

    grad = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    A = compute_anisotropy_matrices_from_gradients(points, grad, r_vertical, r_lateral)

    correction = evaluate_micro_correction(xyz, points, weights, A, kernel_range=range_val)

    xy_val = correction[0]
    z_val = correction[1]
    assert xy_val > z_val, (
        f"Lateral correction ({xy_val}) should be larger than vertical ({z_val}) "
        f"since r_lateral > r_vertical"
    )


# ----------------------------------------------------------------
# build_micro_covariance
# ----------------------------------------------------------------

def test_build_covariance_identity_symmetric_psd():
    """Covariance matrix with identity anisotropy is symmetric and PSD."""
    N = 5
    rng = np.random.default_rng(42)
    points = rng.uniform(-5, 5, (N, 3)).astype(np.float64)
    A = _make_identity_anisotropy(N)

    K = build_micro_covariance(points, A, kernel_range=2.0)

    np.testing.assert_allclose(K, K.T, atol=1e-14)
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-10), f"K is not PSD: min eigenvalue = {eigvals.min()}"


def test_build_covariance_diagonal_max():
    """Diagonal entries are the largest in each row (kernel is maximum at zero distance)."""
    N = 10
    rng = np.random.default_rng(123)
    points = rng.uniform(-5, 5, (N, 3)).astype(np.float64)
    A = _make_identity_anisotropy(N)

    K = build_micro_covariance(points, A, kernel_range=2.0)

    for i in range(N):
        assert K[i, i] >= np.max(K[i, :]) - 1e-14, f"Row {i}: diag {K[i,i]:.6f} < max {np.max(K[i,:]):.6f}"


def test_build_covariance_nugget():
    """Nugget increases diagonal by exactly the nugget value."""
    N = 5
    rng = np.random.default_rng(99)
    points = rng.uniform(-5, 5, (N, 3)).astype(np.float64)
    A = _make_identity_anisotropy(N)

    K_no_nugget = build_micro_covariance(points, A, kernel_range=2.0, nugget=0.0)
    K_with_nugget = build_micro_covariance(points, A, kernel_range=2.0, nugget=0.1)

    np.testing.assert_allclose(
        np.diag(K_with_nugget) - np.diag(K_no_nugget),
        0.1,
        atol=1e-14
    )


# ----------------------------------------------------------------
# solve_micro_weights
# ----------------------------------------------------------------

def test_solve_and_evaluate_roundtrip_identity():
    """Solve K@w = residuals, then evaluate back at micro points -> should match residuals."""
    N = 4
    rng = np.random.default_rng(42)
    points = rng.uniform(-3, 3, (N, 3)).astype(np.float64)
    A = _make_identity_anisotropy(N)
    residuals = np.array([0.5, -0.3, 1.2, -0.8], dtype=np.float64)

    weights = solve_micro_weights(points, residuals, A, kernel_range=2.0, nugget=1e-6)
    correction = evaluate_micro_correction(points, points, weights, A, kernel_range=2.0)

    np.testing.assert_allclose(correction, residuals, rtol=1e-5)


def test_solve_micro_weights_produces_finite_weights():
    """Solving with anisotropic matrices should produce finite, non-NaN weights."""
    N = 4
    rng = np.random.default_rng(99)
    points = rng.uniform(-3, 3, (N, 3)).astype(np.float64)
    residuals = np.array([0.5, -0.3, 1.2, -0.8], dtype=np.float64)

    grad = rng.normal(0, 1, (N, 3)).astype(np.float64)
    grad = grad / np.linalg.norm(grad, axis=1, keepdims=True)
    A = compute_anisotropy_matrices_from_gradients(points, grad, r_vertical=0.5, r_lateral=5.0)

    weights = solve_micro_weights(points, residuals, A, kernel_range=2.0, nugget=1e-4)
    assert np.all(np.isfinite(weights)), "Weights should be finite"
    assert np.all(weights != 0), "Weights should be non-zero"

    correction = evaluate_micro_correction(points, points, weights, A, kernel_range=2.0)
    assert np.all(np.isfinite(correction)), "Correction evaluation should be finite"


def test_solve_micro_weights_far_apart_points():
    """When points are far apart, weights should approximate residuals (diagonal-dominant K)."""
    N = 3
    points = np.array([
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [0.0, 100.0, 0.0],
    ], dtype=np.float64)
    A = _make_identity_anisotropy(N)
    residuals = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    weights = solve_micro_weights(points, residuals, A, kernel_range=1.0)
    np.testing.assert_allclose(weights, residuals, rtol=1e-3)


# ----------------------------------------------------------------
# compute_anisotropy_matrices_from_gradients
# ----------------------------------------------------------------

def test_anisotropy_matrices_shape():
    N = 3
    points = np.random.default_rng(1).uniform(0, 1, (N, 3)).astype(np.float64)
    grad = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    A = compute_anisotropy_matrices_from_gradients(points, grad)

    assert A.shape == (N, 3, 3)


def test_anisotropy_vertical_gradient_produces_expected_scaling():
    """With vertical gradient, X and Y axes get lateral scale, Z gets vertical scale."""
    r_v, r_l = 0.5, 5.0
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    grad = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

    A = compute_anisotropy_matrices_from_gradients(points, grad, r_v, r_l)

    x_transformed = A[0] @ np.array([1.0, 0.0, 0.0])
    z_transformed = A[0] @ np.array([0.0, 0.0, 1.0])

    np.testing.assert_allclose(np.linalg.norm(x_transformed), 1.0 / r_l, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(z_transformed), 1.0 / r_v, rtol=1e-6)


def test_anisotropy_matrix_is_minimum_stretch():
    """Anisotropy matrices produce ovals not flat lines (determinant > 0)."""
    N = 5
    rng = np.random.default_rng(42)
    points = rng.uniform(0, 1, (N, 3)).astype(np.float64)
    grad = rng.normal(0, 1, (N, 3)).astype(np.float64)

    A = compute_anisotropy_matrices_from_gradients(points, grad, r_vertical=0.3, r_lateral=5.0)

    for i in range(N):
        det = np.linalg.det(A[i])
        assert det > 0, f"Matrix {i} has non-positive determinant: {det}"
