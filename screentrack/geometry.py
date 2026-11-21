"""Planar homography helpers.

All images use pixel coordinates with the origin at the top-left corner and
y pointing down. A homography H maps screen (texture) coordinates to frame
coordinates: p_frame ~ H @ p_screen.
"""
from __future__ import annotations

import numpy as np


def normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pts, dtype=np.float64)
    mean = pts.mean(axis=0)
    d = pts - mean
    scale = np.sqrt(2.0) / max(np.mean(np.linalg.norm(d, axis=1)), 1e-9)
    T = np.array(
        [[scale, 0.0, -scale * mean[0]], [0.0, scale, -scale * mean[1]], [0.0, 0.0, 1.0]]
    )
    return T, (pts - mean) * scale


def homography_dlt(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Normalized DLT. `src` -> `dst`. Returns 3x3 with h33 = 1 when possible."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] < 4:
        raise ValueError("need at least 4 point correspondences")
    if weights is None:
        weights = np.ones(src.shape[0])
    weights = np.asarray(weights, dtype=np.float64)
    Ts, sn = normalize_points(src)
    Td, dn = normalize_points(dst)
    n = sn.shape[0]
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = sn[i]
        u, v = dn[i]
        w = weights[i]
        A[2 * i] = w * np.array([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A[2 * i + 1] = w * np.array([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(A)
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))]) @ H.T
    w = hom[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, np.where(w < 0, -1e-12, 1e-12), w)
    return hom[:, :2] / w


def quad_from_h(H: np.ndarray, width: float, height: float) -> np.ndarray:
    """Corners in order TL, TR, BR, BL of the screen rectangle."""
    return apply_h(H, np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float64))


def h_from_quad(quad: np.ndarray, width: float, height: float) -> np.ndarray:
    return homography_dlt(
        np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float64), quad
    )


def project(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    return apply_h(H, points)


def inverse(H: np.ndarray) -> np.ndarray:
    return np.linalg.inv(H)


def corner_order(quad: np.ndarray) -> np.ndarray:
    """Reorder 4 points to TL, TR, BR, BL using angle around the centroid."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    order = np.argsort(ang)
    q = q[order]
    # argsort gives increasing angle: starts at -pi, so rotate to start at TL-ish point
    start = int(np.argmin(q.sum(axis=1)))
    q = np.roll(q, -start, axis=0)
    # ensure clockwise (in image coords y-down, TL->TR->BR->BL is clockwise)
    area = 0.0
    for i in range(4):
        x1, y1 = q[i]
        x2, y2 = q[(i + 1) % 4]
        area += x1 * y2 - x2 * y1
    if area < 0:
        q = q[[0, 3, 2, 1]]
    return q


def polygon_area(quad: np.ndarray) -> float:
    q = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping of `subject` polygon by convex `clip` polygon."""
    out = [tuple(p) for p in subject]

    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def intersect(p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    clips = [tuple(p) for p in clip]
    for i in range(len(clips)):
        a = clips[i]
        b = clips[(i + 1) % len(clips)]
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if inside(e, a, b):
                if not inside(s, a, b):
                    out.append(intersect(s, e, a, b))
                out.append(e)
            elif inside(s, a, b):
                out.append(intersect(s, e, a, b))
            s = e
    return np.array(out, dtype=np.float64) if out else np.empty((0, 2))


def overlap_ratio(quad_a: np.ndarray, quad_b: np.ndarray) -> float:
    """Fraction of quad_a covered by quad_b."""
    area_a = polygon_area(quad_a)
    if area_a <= 1e-9:
        return 0.0
    inter = clip_polygon(np.asarray(quad_a, np.float64), np.asarray(quad_b, np.float64))
    if len(inter) < 3:
        return 0.0
    return float(min(1.0, polygon_area(inter) / area_a))


def quad_side_lengths(quad: np.ndarray) -> np.ndarray:
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    return np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)


def homography_distance(quad_a: np.ndarray, quad_b: np.ndarray, width: float, height: float,
                        samples: int = 12) -> float:
    """Mean pixel displacement between two homographies over a sampled screen grid."""
    H1 = h_from_quad(quad_a, width, height)
    H2 = h_from_quad(quad_b, width, height)
    us = np.linspace(0, width, samples)
    vs = np.linspace(0, height, samples)
    grid = np.array([[u, v] for u in us for v in vs], dtype=np.float64)
    p1 = apply_h(H1, grid)
    p2 = apply_h(H2, grid)
    return float(np.mean(np.linalg.norm(p1 - p2, axis=1)))


def decompose_homography_scales(H: np.ndarray) -> dict:
    """Approximate in-plane scale, rotation, and perspective from a homography.

    Uses the Jacobian at the image centre, so it describes what the viewer sees
    for a screen-centred point rather than the raw H entries.
    """
    H = np.asarray(H, dtype=np.float64)
    inv = np.linalg.inv(H)
    screen_c = apply_h(inv, np.array([[320.0, 240.0]]))[0]  # normalised-ish centre
    eps = 1.0
    base = apply_h(H, screen_c.reshape(1, 2))[0]
    du = apply_h(H, (screen_c + [eps, 0]).reshape(1, 2))[0] - base
    dv = apply_h(H, (screen_c + [0, eps]).reshape(1, 2))[0] - base
    su = float(np.linalg.norm(du))
    sv = float(np.linalg.norm(dv))
    rot = float(np.degrees(np.arctan2(du[1], du[0])))
    persp = float(np.linalg.det(H[:2, :2]))
    return {"scale_u": su, "scale_v": sv, "scale": (su + sv) / 2.0, "rotation_deg": rot,
            "perspective_det": persp}
