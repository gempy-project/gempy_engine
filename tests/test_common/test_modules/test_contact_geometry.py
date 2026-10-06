"""Independent triangle-patch oracles for affine contact reconciliation."""

import numpy as np
import pytest

from gempy_engine.modules.dual_contouring.contact_geometry import reconcile_planar_contact


TOL = 1e-8


def planar_patches(tilt=0.0):
    # Controller strips cross the seam at y=.3; the target diagonal at y=.47.
    cv = np.array([[x, y, .47 + tilt * (x - .43)]
                   for y in (0., .3, 1.) for x in (0., 1.)])
    cf = np.array([[0, 1, 3], [0, 3, 2], [2, 3, 5], [2, 5, 4]])
    tv = np.array([[.43, 0., 0.], [.43, 1., 0.],
                   [.43, 1., 1.], [.43, 0., 1.]])
    tf = np.array([[0, 1, 2], [0, 2, 3]])
    normal = np.array([-tilt, 0., 1.]) / np.sqrt(1 + tilt ** 2)
    return cv, cf, tv, tf, (normal, (.47 - tilt * .43) / np.sqrt(1 + tilt ** 2)), (np.array([1., 0., 0.]), .43)


def assert_contact_meshes(cv, cf, tv, tf, controller_plane, truncated_plane,
                          retained_sign, controller_area, target_area, winding=1,
                          controller_incidence=2, required_breakpoints=(.3, .47)):
    seam_edges = []
    for vertices, faces, plane, area, incidence in (
            (cv, cf, controller_plane, controller_area, controller_incidence),
            (tv, tf, truncated_plane, target_area, 1)):
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3
        assert len(faces) > 0
        assert np.isfinite(vertices).all()
        assert faces.dtype.kind in "iu"
        assert faces.min() >= 0 and faces.max() < len(vertices)
        assert len(np.unique(np.sort(faces, axis=1), axis=0)) == len(faces)
        np.testing.assert_allclose(vertices @ plane[0], plane[1], atol=TOL, rtol=0)
        triangles = vertices[faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        assert np.all(winding * (cross @ plane[0]) > 1e-12)
        np.testing.assert_allclose(np.linalg.norm(cross, axis=1).sum() / 2, area, atol=TOL, rtol=0)
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        assert np.all(counts <= 2)
        segments = vertices[edges]
        on_seam = (np.all(np.abs(segments @ controller_plane[0] - controller_plane[1]) <= TOL, axis=1)
                   & np.all(np.abs(segments @ truncated_plane[0] - truncated_plane[1]) <= TOL, axis=1))
        assert np.any(on_seam)
        assert np.all(counts[on_seam] == incidence)
        intervals = np.sort(segments[on_seam, :, 1], axis=1)
        intervals = intervals[np.argsort(intervals[:, 0])]
        assert np.all(intervals[:, 1] - intervals[:, 0] > TOL)
        np.testing.assert_allclose(intervals[0, 0], 0., atol=TOL, rtol=0)
        np.testing.assert_allclose(intervals[-1, 1], 1., atol=TOL, rtol=0)
        np.testing.assert_allclose(intervals[:-1, 1], intervals[1:, 0], atol=TOL, rtol=0)
        seam_edges.append(intervals)
        # A seam vertex inside any unsplit edge is a T-junction, even when both
        # meshes contain the same set of contact points.
        used = np.unique(faces)
        seam_vertices = vertices[used]
        seam_vertices = seam_vertices[
            (np.abs(seam_vertices @ controller_plane[0] - controller_plane[1]) <= TOL)
            & (np.abs(seam_vertices @ truncated_plane[0] - truncated_plane[1]) <= TOL)]
        for a, b in segments:
            direction = b - a
            length_squared = direction @ direction
            assert length_squared > 1e-24
            parameters = (seam_vertices - a) @ direction / length_squared
            distances = np.linalg.norm(seam_vertices - a - parameters[:, None] * direction, axis=1)
            assert not np.any((parameters > TOL) & (parameters < 1 - TOL) & (distances < TOL))
    np.testing.assert_allclose(seam_edges[0], seam_edges[1], atol=TOL, rtol=0)
    # Both independently introduced breakpoints must survive as real edges.
    endpoints = np.unique(seam_edges[0])
    for breakpoint in required_breakpoints:
        assert np.any(np.abs(endpoints - breakpoint) < TOL)
    assert np.all(retained_sign * (tv[np.unique(tf)] @ controller_plane[0] - controller_plane[1]) >= -TOL)


@pytest.mark.parametrize("retained_sign", [-1, 1], ids=["erosion_below", "onlap_above"])
@pytest.mark.parametrize("tilt", [0., .2], ids=["perpendicular", "oblique"])
@pytest.mark.parametrize("winding", [1, -1], ids=["forward_winding", "reverse_winding"])
def test_planar_contact_matches_edge_segments_and_preserves_area(retained_sign, tilt, winding):
    cv, cf, tv, tf, cp, tp = planar_patches(tilt)
    if winding == -1:
        cf, tf = cf[:, ::-1].copy(), tf[:, ::-1].copy()
    originals = [array.copy() for array in (cv, cf, tv, tf, cp[0], tp[0])]
    result = reconcile_planar_contact(cv, cf, tv, tf, cp, tp, retained_sign, tolerance=TOL)
    out_cv, out_cf, out_tv, out_tf, report = result
    assert isinstance(report, dict) and report
    assert report["status"] == "reconciled"
    assert report["controller_boundary_seam_segment_count"] == 0
    assert_contact_meshes(out_cv, out_cf, out_tv, out_tf, cp, tp, retained_sign,
                          np.sqrt(1 + tilt ** 2), .47 if retained_sign == -1 else .53, winding)
    for actual, original in zip((cv, cf, tv, tf, cp[0], tp[0]), originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("normal_sign", [1, -1])
def test_parallel_no_join_leaves_retained_target_untouched(normal_sign):
    cv, cf, _, tf, cp, _ = planar_patches()
    tv = np.array([[0., 0., .46], [1., 0., .46], [1., 1., .46], [0., 1., .46]])
    tp = (np.array([0., 0., float(normal_sign)]), normal_sign * .46)
    originals = [array.copy() for array in (cv, cf, tv, tf)]
    *outputs, report = reconcile_planar_contact(cv, cf, tv, tf, cp, tp, -1, tolerance=TOL)
    assert report["status"] == "no_contact"
    assert report["reason"] == "parallel_planes"
    assert report["seam_segment_count"] == report["seam_vertex_count"] == 0
    for actual, original, output in zip((cv, cf, tv, tf), originals, outputs):
        np.testing.assert_array_equal(actual, original)
        np.testing.assert_array_equal(output, original)


@pytest.mark.parametrize("case", ["coincident", "partial_support", "disjoint_support"])
def test_unsupported_contact_has_explicit_diagnostic_without_input_mutation(case):
    cv, cf, tv, tf, cp, tp = planar_patches()
    if case == "coincident":
        tv, tf, tp = cv.copy(), cf.copy(), (cp[0].copy(), cp[1])
    elif case == "partial_support":
        # The target seam extends beyond the finite controller patch.
        cv[:, 1] *= .6
    else:
        cv[:, 1] += 2.
    originals = [array.copy() for array in (cv, cf, tv, tf)]
    diagnostic = "Coincident" if case == "coincident" else "Insufficient controller support"
    with pytest.raises(ValueError, match=diagnostic):
        reconcile_planar_contact(cv, cf, tv, tf, cp, tp, -1, tolerance=TOL)
    for actual, original in zip((cv, cf, tv, tf), originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("empty_role", ["truncated", "both"])
def test_empty_patches_return_valid_arrays_without_join(empty_role):
    cv, cf, tv, tf, cp, tp = planar_patches()
    if empty_role in ("controller", "both"):
        cv, cf = np.empty((0, 3)), np.empty((0, 3), dtype=int)
    if empty_role in ("truncated", "both"):
        tv, tf = np.empty((0, 3)), np.empty((0, 3), dtype=int)
    originals = [array.copy() for array in (cv, cf, tv, tf)]
    *outputs, report = reconcile_planar_contact(cv, cf, tv, tf, cp, tp, -1, tolerance=TOL)
    assert isinstance(report, dict) and report
    assert report["status"] == "no_contact"
    assert report["reason"] == "empty_patch"
    for actual, original, output in zip((cv, cf, tv, tf), originals, outputs):
        np.testing.assert_array_equal(actual, original)
        np.testing.assert_array_equal(output, original)
        assert output.shape == original.shape


@pytest.mark.parametrize("bottom", [.47, .6], ids=["discarded_boundary_touch", "wholly_wrong_side"])
@pytest.mark.parametrize("retained_sign", [-1, 1])
@pytest.mark.parametrize("empty_controller", [False, True])
def test_fully_discarded_target_has_no_faces_or_join(bottom, retained_sign, empty_controller):
    cv, cf, tv, tf, cp, tp = planar_patches()
    if empty_controller:
        cv, cf = np.empty((0, 3)), np.empty((0, 3), dtype=int)
    tv[:, 2] = np.where(tv[:, 2] == 0, bottom, .9)
    if retained_sign == 1:
        tv[:, 2] = .94 - tv[:, 2]
    originals = [array.copy() for array in (cv, cf, tv, tf)]
    out_cv, out_cf, out_tv, out_tf, report = reconcile_planar_contact(
        cv, cf, tv, tf, cp, tp, retained_sign)
    assert out_tf.shape == (0, 3) and out_tf.dtype.kind in "iu"
    assert report["status"] == "no_contact"
    assert report["reason"] == "target_fully_discarded"
    assert report["truncated_faces_after"] == 0
    assert report["seam_segment_count"] == report["seam_vertex_count"] == 0
    assert report["controller_boundary_seam_segment_count"] == 0
    for output, original in zip((out_cv, out_cf, out_tv), originals):
        np.testing.assert_array_equal(output, original)
    for actual, original in zip((cv, cf, tv, tf), originals):
        np.testing.assert_array_equal(actual, original)


def test_disjoint_fully_retained_target_remains_unchanged():
    cv, cf, tv, tf, cp, tp = planar_patches()
    cv[:, 1] += 2.
    tv[:, 2] = np.where(tv[:, 2] == 0, .6, .9)
    *outputs, report = reconcile_planar_contact(cv, cf, tv, tf, cp, tp, 1)
    assert report["status"] == "no_contact"
    for output, original in zip(outputs, (cv, cf, tv, tf)):
        np.testing.assert_array_equal(output, original)


def test_crossing_target_with_empty_controller_has_insufficient_support():
    _, _, tv, tf, cp, tp = planar_patches()
    with pytest.raises(ValueError, match="Insufficient controller support"):
        reconcile_planar_contact(np.empty((0, 3)), np.empty((0, 3), dtype=int),
                                 tv, tf, cp, tp, -1)


def test_disconnected_retained_and_discarded_components_apply_ownership_without_join():
    cv, cf, tv, tf, cp, tp = planar_patches()
    retained = tv.copy()
    retained[:, 2] = np.where(tv[:, 2] == 0, .6, .9)
    discarded = retained.copy()
    discarded[:, 2] -= .6
    tv = np.concatenate((retained, discarded))
    tf = np.concatenate((tf, tf + 4))
    out_cv, out_cf, out_tv, out_tf, report = reconcile_planar_contact(cv, cf, tv, tf, cp, tp, 1)
    assert report["status"] == "no_contact"
    assert report["truncated_faces_after"] == 2
    assert report["seam_segment_count"] == 0
    for output, original in zip((out_cv, out_cf, out_tv, out_tf), (cv, cf, tv, tf[:2])):
        np.testing.assert_array_equal(output, original)


@pytest.mark.parametrize("winding", [1, -1])
def test_boundary_to_boundary_seam_reports_controller_boundary_incidence(winding):
    cv, cf, tv, tf, cp, tp = planar_patches()
    cv[:, 0] = np.where(cv[:, 0] == 0, .43, 1.)
    tv[:, 2] = np.where(tv[:, 2] == 0, .47, .9)
    if winding == -1:
        cf, tf = cf[:, ::-1].copy(), tf[:, ::-1].copy()
    out_cv, out_cf, out_tv, out_tf, report = reconcile_planar_contact(
        cv, cf, tv, tf, cp, tp, 1)
    assert report["status"] == "reconciled"
    assert report["controller_boundary_seam_segment_count"] == report["seam_segment_count"] == 2
    assert_contact_meshes(out_cv, out_cf, out_tv, out_tf, cp, tp, 1, .57, .43,
                          winding, controller_incidence=1, required_breakpoints=(.3,))


@pytest.mark.parametrize("role", ["controller", "target"])
def test_invalid_seam_face_incidence_is_rejected(role):
    cv, cf, tv, tf, cp, tp = planar_patches()
    if role == "controller":
        cf = np.concatenate((cf, cf))
    else:
        tf = np.concatenate((tf, tf))
    with pytest.raises(ValueError, match=role + " seam incidence"):
        reconcile_planar_contact(cv, cf, tv, tf, cp, tp, -1)


def test_tolerance_below_world_coordinate_precision_is_rejected():
    cv, cf, tv, tf, cp, tp = planar_patches(.2)
    shift = np.array([1e12, 1e12, 1e12])
    cv, tv = cv + shift, tv + shift
    cp = (cp[0], cp[1] + shift @ cp[0])
    tp = (tp[0], tp[1] + shift @ tp[0])
    with pytest.raises(ValueError, match="Tolerance.*representable coordinate precision"):
        reconcile_planar_contact(cv, cf, tv, tf, cp, tp, -1, tolerance=TOL)
