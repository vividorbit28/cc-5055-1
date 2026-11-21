import numpy as np
import pytest

from screentrack import geometry


def test_homography_roundtrip():
    rng = np.random.default_rng(3)
    quad = np.array([[40, 30], [250, 55], [230, 190], [55, 205]], np.float64)
    H = geometry.h_from_quad(quad, 320, 180)
    corners = np.array([[0, 0], [320, 0], [320, 180], [0, 180]], np.float64)
    assert np.allclose(geometry.apply_h(H, corners), quad, atol=1e-6)
    truth = rng.uniform(0, 300, (20, 2))
    Hback = np.linalg.inv(H)
    assert np.allclose(geometry.apply_h(Hback, geometry.apply_h(H, truth)), truth, atol=1e-6)


def test_corner_order_fixes_permutations():
    quad = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], np.float64)
    for perm in ([2, 0, 3, 1], [1, 2, 0, 3], [3, 1, 0, 2]):
        assert np.allclose(geometry.corner_order(quad[perm]), quad)


def test_corner_order_rejects_clockwise_flip():
    quad = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], np.float64)
    flipped = quad[[0, 3, 2, 1]]
    assert np.allclose(geometry.corner_order(flipped), quad)


def test_overlap_ratio():
    a = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float64)
    b = np.array([[50, 0], [150, 0], [150, 100], [50, 100]], np.float64)
    assert abs(geometry.overlap_ratio(a, b) - 0.5) < 1e-9
    assert abs(geometry.overlap_ratio(a, a) - 1.0) < 1e-9
    c = np.array([[200, 200], [300, 200], [300, 300], [200, 300]], np.float64)
    assert geometry.overlap_ratio(a, c) == 0.0


def test_homography_distance_zero_for_same_quad():
    quad = np.array([[10, 20], [300, 30], [290, 200], [20, 210]], np.float64)
    assert geometry.homography_distance(quad, quad, 320, 180) < 1e-9


def test_homography_distance_detects_slip():
    a = np.array([[10, 20], [300, 30], [290, 200], [20, 210]], np.float64)
    b = a + [2.0, 0.0]
    d = geometry.homography_distance(a, b, 320, 180)
    assert 1.5 < d < 2.5


def test_decompose_scales():
    H = geometry.h_from_quad(np.array([[0, 0], [200, 0], [200, 100], [0, 100]], np.float64),
                             320, 180)
    d = geometry.decompose_homography_scales(H)
    assert abs(d["scale_u"] - 200 / 320) < 0.02
    assert abs(d["scale_v"] - 100 / 180) < 0.02
    assert abs(d["rotation_deg"]) < 0.01 or abs(abs(d["rotation_deg"]) - 360) < 0.01
