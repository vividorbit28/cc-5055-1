"""Planar screen tracker.

Per frame:
  1. predict the quad from the last accepted observations,
  2. detect SIFT features in the frame and match them against a full-resolution
     keyframe whose features were detected inside the screen polygon,
  3. gate matches with the prediction, run RANSAC, then verify each survivor
     photometrically against the keyframe (this is what rejects features on a
     foreground occluder instead of absorbing them into the homography),
  4. refine the four corners by template matching (retained-corner recheck),
  5. measure occlusion from keyframe-vs-frame consistency over a texture grid,
  6. propagate homography covariance to the corners and score confidence from
     inlier count, spatial coverage and reprojection error.

A new keyframe is promoted only from a frame with full-quality support, which
keeps the reference view close to the current view without accumulating drift.
The original anchor view is kept for relocalisation after long occlusions.

States:
  anchor            manually specified reference
  tracking          full-quality observation
  weak              observation accepted, but coverage/precision below full
  tracking_partial  observation retained while the screen is over half occluded
  predicted         no usable observation; planar extrapolation from history
  paused            prediction horizon exhausted; overlay withheld
  reacquired        corners re-verified after a gap, tracking resumed
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .geometry import (apply_h, corner_order, h_from_quad, homography_dlt,
                       polygon_area, quad_from_h)


@dataclass
class TrackConfig:
    sift_features: int = 1200
    sift_contrast: float = 0.004
    sift_edge: float = 12.0
    ratio: float = 0.76
    ransac_px: float = 3.0
    min_matches: int = 10
    min_inliers: int = 10
    photometric_ncc: float = 0.55
    max_reproj: float = 3.5
    min_coverage_full: float = 0.42
    max_corner_sigma_frac: float = 0.03
    occlusion_hidden: float = 0.50
    occlusion_clear: float = 0.34
    max_predict: int = 5
    gate_frac: float = 0.09
    gate_frac_slope: float = 0.02
    gate_frac_max: float = 0.4
    corner_score: float = 0.55
    corner_search_frac: float = 0.05
    occlusion_grid: tuple = (20, 12)
    coverage_grid: int = 4
    max_feature_points: int = 240
    max_rejected_points: int = 90
    rekey_every: int = 14
    rekey_min_inliers: int = 16
    rekey_max_occlusion: float = 0.03

    def to_dict(self):
        d = asdict(self)
        d["occlusion_grid"] = list(self.occlusion_grid)
        return d


@dataclass
class ReferenceModel:
    """A full-resolution view of the screen plus its keypoint descriptors."""
    frame_idx: int
    quad: np.ndarray            # frame coords, TL TR BR BL
    tex_size: tuple[int, int]   # texture pixels (w, h), constant across keyframes
    H_tex: np.ndarray           # texture -> frame
    gray: np.ndarray            # full frame grayscale
    pts_tex: np.ndarray         # keypoint positions in texture coords
    desc: np.ndarray
    corner_patches: list        # texture-space grayscale patches (anchor only)
    corner_pts_tex: np.ndarray  # texture coords of the four panel corners
    is_anchor: bool = False

    def save(self, directory: str, name: str) -> None:
        os.makedirs(directory, exist_ok=True)
        cv2.imwrite(os.path.join(directory, f"{name}_gray.png"), self.gray)


def _texture_corner_patches(tex_gray: np.ndarray, inset_frac: float = 0.085,
                            patch_frac: float = 0.17) -> tuple[list, np.ndarray]:
    """Grab a textured patch near each panel corner, moving inward if the corner
    sits on a flat bezel. A flat patch makes TM_CCOEFF_NORMED return a meaningless
    1.0 everywhere, which would turn corner verification into a rubber stamp.

    Returns the patches and the texture coordinates of each patch's centre. Those
    centres are the probe points; a homography solved from them maps texture to
    frame, and the outer quad is read back from that homography.
    """
    rh, rw = tex_gray.shape
    inset = int(np.clip(inset_frac * min(rw, rh), 12, 52))
    patch = int(max(23, min(2 * inset - 3, patch_frac * min(rw, rh))))
    if patch % 2 == 0:
        patch += 1
    half = patch // 2
    nominal = np.array([[inset, inset], [rw - 1 - inset, inset],
                        [rw - 1 - inset, rh - 1 - inset], [inset, rh - 1 - inset]],
                       np.float64)
    centre = nominal.mean(axis=0)
    patches, probe = [], []
    for cx, cy in nominal:
        d = centre - np.array([cx, cy])
        d /= max(1e-6, np.linalg.norm(d))
        best_std, best = -1.0, None
        for shift in np.arange(0.0, 0.42 * min(rw, rh), max(3.0, patch / 2.0)):
            x0 = int(np.clip(round(cx + d[0] * shift) - half, 0, rw - patch))
            y0 = int(np.clip(round(cy + d[1] * shift) - half, 0, rh - patch))
            win = tex_gray[y0:y0 + patch, x0:x0 + patch]
            std = float(win.std())
            if std > best_std:
                best_std = std
                best = (win.copy(), np.array([x0 + half, y0 + half], np.float64))
            if std >= 12.0:
                break
        patches.append(best[0])
        probe.append(best[1])
    return patches, np.array(probe)


def _rectified_texture(frame: np.ndarray, quad: np.ndarray, tex_size: tuple[int, int]) -> np.ndarray:
    H = h_from_quad(corner_order(quad), *tex_size)   # texture -> frame
    return cv2.warpPerspective(frame, np.linalg.inv(H), tex_size, flags=cv2.INTER_LINEAR)


def _detect_in_quad(gray: np.ndarray, quad: np.ndarray, cfg: TrackConfig, H_tex: np.ndarray,
                    tex_size: tuple[int, int]):
    mask = np.zeros(gray.shape[:2], np.uint8)
    q = np.round(quad).astype(np.int32)
    cv2.fillConvexPoly(mask, q, 255)
    shrink = max(2, int(0.012 * np.linalg.norm(quad[2] - quad[0])))
    cv2.polylines(mask, [q], True, 0, shrink)
    sift = cv2.SIFT_create(nfeatures=cfg.sift_features, contrastThreshold=cfg.sift_contrast,
                           edgeThreshold=cfg.sift_edge)
    kp, des = sift.detectAndCompute(gray, mask)
    if des is None or len(kp) < 8:
        return None
    pts_tex = apply_h(np.linalg.inv(H_tex), np.array([k.pt for k in kp], np.float64))
    tw, th = tex_size
    inside = ((pts_tex[:, 0] > 2) & (pts_tex[:, 0] < tw - 3) &
              (pts_tex[:, 1] > 2) & (pts_tex[:, 1] < th - 3))
    if inside.sum() < 8:
        return None
    return pts_tex[inside], des[inside]


def build_reference(frame: np.ndarray, quad: np.ndarray, cfg: TrackConfig,
                    frame_idx: int = 0, tex_size: tuple[int, int] | None = None,
                    anchor: ReferenceModel | None = None) -> ReferenceModel:
    quad = corner_order(quad)
    if tex_size is None:
        sides_x = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
        sides_y = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
        tex_size = (int(np.clip(round(sides_x), 160, 1280)), int(np.clip(round(sides_y), 96, 720)))
    H_tex = h_from_quad(quad, *tex_size)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    det = _detect_in_quad(gray, quad, cfg, H_tex, tex_size)
    if det is None:
        raise ValueError("the selected region has too few trackable features; pick a more textured area")
    pts_tex, desc = det
    if anchor is None:
        tex_gray = _rectified_texture(gray, quad, tex_size)
        patches, corner_pts = _texture_corner_patches(tex_gray)
        return ReferenceModel(frame_idx, quad, tex_size, H_tex, gray, pts_tex, desc,
                              patches, corner_pts, is_anchor=True)
    return ReferenceModel(frame_idx, quad, tex_size, H_tex, gray, pts_tex, desc,
                          anchor.corner_patches, anchor.corner_pts_tex, is_anchor=False)


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None):
    if mask is None:
        va = a.astype(np.float32).ravel()
        vb = b.astype(np.float32).ravel()
    else:
        va = a[mask].astype(np.float32)
        vb = b[mask].astype(np.float32)
    if va.size < 16:
        return None
    va = va - va.mean()
    vb = vb - vb.mean()
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-4 or nb < 1e-4:
        return None
    return float(np.dot(va, vb) / (na * nb))


def _local_ncc(a: np.ndarray, b: np.ndarray, x: int, y: int, half: int = 6):
    h, w = a.shape
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    if x1 - x0 < 7 or y1 - y0 < 7:
        return None
    return _ncc(a[y0:y1, x0:x1], b[y0:y1, x0:x1])


def _quad_valid(quad: np.ndarray, frame_shape: tuple[int, int], ref_quad: np.ndarray,
                margins: float = 0.45) -> bool:
    if not np.isfinite(quad).all():
        return False
    ordered = corner_order(quad)
    scale = max(1.0, np.linalg.norm(ordered[2] - ordered[0]))
    if np.max(np.linalg.norm(ordered - quad, axis=1)) > 0.02 * scale:
        return False
    area = polygon_area(quad)
    if area < 0.02 * polygon_area(ref_quad) or area > 25.0 * polygon_area(ref_quad):
        return False
    if np.min(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)) < 14:
        return False
    h, w = frame_shape
    mx, my = margins * w, margins * h
    if quad[:, 0].min() < -mx or quad[:, 0].max() > w + mx:
        return False
    if quad[:, 1].min() < -my or quad[:, 1].max() > h + my:
        return False
    return True


def homography_covariance(H: np.ndarray, pts_tex: np.ndarray, residuals: np.ndarray,
                          corners_tex: np.ndarray):
    """Propagate homography estimation uncertainty to corner positions in pixels."""
    h = np.array([H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2], H[2, 0], H[2, 1]],
                 np.float64)

    def project(pts, params):
        Hm = np.array([[params[0], params[1], params[2]],
                       [params[3], params[4], params[5]],
                       [params[6], params[7], 1.0]])
        hom = np.hstack([pts, np.ones((len(pts), 1))]) @ Hm.T
        return hom[:, :2] / hom[:, 2:3]

    n = len(pts_tex)
    J = np.zeros((2 * n, 8))
    eps = 1e-6
    base = project(pts_tex, h).reshape(-1)
    for k in range(8):
        hp = h.copy()
        hp[k] += eps
        J[:, k] = (project(pts_tex, hp).reshape(-1) - base) / eps
    sigma = max(1e-3, 1.4826 * float(np.median(np.abs(residuals))))
    try:
        JTJ = J.T @ J
        ridge = 1e-9 * (np.trace(JTJ) / 8.0 + 1e-9) * np.eye(8)
        cov = sigma ** 2 * np.linalg.inv(JTJ + ridge)
    except np.linalg.LinAlgError:
        cov = sigma ** 2 * np.linalg.pinv(J.T @ J)
    sigmas = []
    for cp in corners_tex:
        Jc = np.zeros((8, 2))
        base8 = project(cp.reshape(1, 2), h).reshape(-1)
        for k in range(8):
            hp = h.copy()
            hp[k] += eps
            Jc[k] = (project(cp.reshape(1, 2), hp).reshape(-1) - base8) / eps
        cc = Jc.T @ cov @ Jc
        eig = np.linalg.eigvalsh(cc)
        sigmas.append(float(np.sqrt(max(eig[-1], 0.0))))
    return sigmas, float(sigma)


def _grid_coverage(pts_tex: np.ndarray, tex_size: tuple[int, int], n: int = 4) -> float:
    if len(pts_tex) == 0:
        return 0.0
    tw, th = tex_size
    gx = np.clip((pts_tex[:, 0] / tw * n).astype(int), 0, n - 1)
    gy = np.clip((pts_tex[:, 1] / th * n).astype(int), 0, n - 1)
    return len(set(zip(gx.tolist(), gy.tolist()))) / float(n * n)


def _occlusion_map(frame_gray: np.ndarray, ref: ReferenceModel, H_tex_cur: np.ndarray,
                   cfg: TrackConfig):
    """Photometric disagreement between the keyframe and the current frame.

    Returns (ratio, per-cell good flags, warped reference image).
    Cells on a foreground object disagree with the reference texture, so they
    are excluded from the transform rather than modelled as screen motion.
    """
    tw, th = ref.tex_size
    H_kf_cur = H_tex_cur @ np.linalg.inv(ref.H_tex)
    fh, fw = frame_gray.shape
    warped = cv2.warpPerspective(ref.gray, H_kf_cur, (fw, fh), flags=cv2.INTER_LINEAR)
    labels = np.zeros((th, tw), np.uint8)
    gx, gy = cfg.occlusion_grid
    cw = max(1, tw // gx)
    ch = max(1, th // gy)
    idx = 1
    for j in range(gy):
        for i in range(gx):
            labels[j * ch:(j + 1) * ch, i * cw:(i + 1) * cw] = idx
            idx += 1
    wlabels = cv2.warpPerspective(labels, H_tex_cur, (fw, fh), flags=cv2.INTER_NEAREST)
    wmask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), H_tex_cur, (fw, fh),
                                flags=cv2.INTER_NEAREST)
    wgray = cv2.GaussianBlur(warped, (5, 5), 0)
    fgray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    total = 0.0
    occ = 0.0
    good = []
    for k in range(1, idx):
        mask = (wlabels == k) & (wmask > 0)
        area = int(mask.sum())
        if area < 30:
            good.append(True)
            continue
        vals = warped[mask].astype(np.float32)
        if float(vals.std()) < 7.0:
            # uninformative (flat) cell: it cannot testify either way
            good.append(True)
            continue
        # brightness-invariant residual: local contrast of the reference cell is
        # the natural floor, so thin edge-on slivers are not called occluded
        # merely because resampling destroyed their normalised correlation
        ref_c = float(vals.std())
        diff = np.abs(wgray[mask].astype(np.float32) - fgray[mask].astype(np.float32))
        floor = 6.0 + 0.45 * ref_c
        occluded = float(np.mean(diff > floor)) > 0.35
        total += area
        if occluded:
            occ += area
            good.append(False)
        else:
            good.append(True)
    return (float(min(1.0, occ / total)) if total > 0 else 0.0), good, warped


def _warp_corner_patch(patch: np.ndarray, H_tex_cur: np.ndarray, corner_tex: np.ndarray):
    ph, pw = patch.shape[:2]
    off = np.array([[1.0, 0.0, corner_tex[0] - pw / 2.0],
                    [0.0, 1.0, corner_tex[1] - ph / 2.0],
                    [0.0, 0.0, 1.0]])
    M = H_tex_cur @ off
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float64)
    dst = apply_h(M, src)
    x0, y0 = dst.min(axis=0)
    x1, y1 = dst.max(axis=0)
    bw = int(round(x1 - x0))
    bh = int(round(y1 - y0))
    if bw < 10 or bh < 10 or bw > 420 or bh > 420:
        return None, None
    T = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
    warped = cv2.warpPerspective(patch, T @ M, (bw, bh), flags=cv2.INTER_LINEAR)
    return warped, np.array([x0, y0, x1, y1])


def _refine_corners(frame_gray: np.ndarray, ref: ReferenceModel, H_tex_cur: np.ndarray,
                    cfg: TrackConfig, search_px: float):
    fh, fw = frame_gray.shape
    scores, visible, refined = [], [], []
    for patch, cp in zip(ref.corner_patches, ref.corner_pts_tex):
        warped_patch, box = _warp_corner_patch(patch, H_tex_cur, cp)
        pred = apply_h(H_tex_cur, cp.reshape(1, 2))[0]
        if warped_patch is None:
            scores.append(0.0)
            visible.append(False)
            refined.append(pred)
            continue
        bw, bh = warped_patch.shape[1], warped_patch.shape[0]
        pad = int(round(search_px)) + 4
        cx0, cy0 = max(0, int(np.floor(box[0])) - pad), max(0, int(np.floor(box[1])) - pad)
        cx1 = min(fw, int(np.ceil(box[2])) + pad)
        cy1 = min(fh, int(np.ceil(box[3])) + pad)
        if cx1 - cx0 < bw + 4 or cy1 - cy0 < bh + 4:
            scores.append(0.0)
            visible.append(False)
            refined.append(pred)
            continue
        # A corner-template match only verifies visibility. It is never used to
        # move a corner independently: per-corner offsets are exactly the
        # "translate the four corners" shortcut that lets a foreground object
        # drag a single corner, so the homography stays feature-driven.
        crop = frame_gray[cy0:cy1, cx0:cx1]
        res = cv2.matchTemplate(crop, warped_patch, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, _ = cv2.minMaxLoc(res)
        scores.append(float(maxv))
        visible.append(bool(maxv >= cfg.corner_score))
        refined.append(pred)
    return scores, visible, np.array(refined, np.float64)


def _corner_search(frame_gray: np.ndarray, ref: ReferenceModel, cfg: TrackConfig,
                   center_hint: np.ndarray, search_px: float, scale_hint: float = 1.0):
    """Search each corner template over a wide area with a few scale hypotheses.

    Used when no feature match is available (occlusion / relocalisation). The
    corners are measured individually; the resulting quad is only accepted if
    all four templates agree and the whole screen passes a photometric check.
    """
    fh, fw = frame_gray.shape
    scores, found = [], []
    scales = [scale_hint * s for s in (0.55, 0.7, 0.85, 1.0, 1.2, 1.45, 1.75)]
    for k, (patch, cp) in enumerate(zip(ref.corner_patches, ref.corner_pts_tex)):
        hint = center_hint[k]
        best = (0.0, None, None)  # score, center, scale
        for sc in scales:
            ps = max(9, int(round(patch.shape[1] * sc)))
            if ps < 11 or ps > 260:
                continue
            tmpl = cv2.resize(patch, (ps, ps), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
            r = int(round(search_px))
            x0 = max(0, int(hint[0]) - r)
            y0 = max(0, int(hint[1]) - r)
            x1 = min(fw, int(hint[0]) + r + 1)
            y1 = min(fh, int(hint[1]) + r + 1)
            if x1 - x0 < ps + 4 or y1 - y0 < ps + 4:
                continue
            res = cv2.matchTemplate(frame_gray[y0:y1, x0:x1], tmpl, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)
            if maxv > best[0]:
                best = (float(maxv), np.array([x0 + maxloc[0] + ps / 2.0,
                                               y0 + maxloc[1] + ps / 2.0]), sc)
        scores.append(best[0])
        found.append(best[1])
    return scores, found


def _solve_corner_quad(frame_gray: np.ndarray, ref: ReferenceModel, cfg: TrackConfig,
                       corners_frame: np.ndarray, H_pred: np.ndarray | None):
    """Build a homography from four verified probe points and validate it photometrically.

    `corners_frame` are the frame positions of the four corner *probes* (textured
    patches near each panel corner, at `ref.corner_pts_tex`). The outer quad comes
    from mapping the true texture corners through the solved homography.
    """
    H = cv2.getPerspectiveTransform(ref.corner_pts_tex.astype(np.float32),
                                    np.asarray(corners_frame, np.float32))
    tw, th = ref.tex_size
    outer_tex = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float64)
    q = corner_order(apply_h(H, outer_tex))
    if not _quad_valid(q, frame_gray.shape[:2], ref.quad, margins=0.9):
        return None
    side_a = np.linalg.norm(q[1] - q[0])
    side_b = np.linalg.norm(q[3] - q[0])
    if side_a < 24 or side_b < 18:
        return None
    aspect = side_a / max(1e-6, side_b)
    tex_aspect = ref.tex_size[0] / float(ref.tex_size[1])
    if not (0.35 * tex_aspect <= aspect <= 2.8 * tex_aspect):
        return None
    # convexity / angle sanity
    angles = []
    for i in range(4):
        a = q[(i - 1) % 4] - q[i]
        b = q[(i + 1) % 4] - q[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return None
        cos = float(np.clip(np.dot(a, b) / (na * nb), -1, 1))
        angles.append(np.degrees(np.arccos(cos)))
    if min(angles) < 22 or max(angles) > 158:
        return None
    if H_pred is not None:
        pred_center = apply_h(H_pred, np.array([[ref.tex_size[0] / 2, ref.tex_size[1] / 2]]))[0]
        shift = np.linalg.norm(q.mean(axis=0) - pred_center)
        diag = np.linalg.norm(q[2] - q[0])
        if shift > 0.75 * diag:
            return None
    H_kf_cur = H @ np.linalg.inv(ref.H_tex)
    fh, fw = frame_gray.shape
    warped = cv2.warpPerspective(ref.gray, H_kf_cur, (fw, fh), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((ref.tex_size[1], ref.tex_size[0]), 255, np.uint8),
                               H, (fw, fh), flags=cv2.INTER_NEAREST)
    if int(mask.sum()) < 0.02 * fh * fw:
        return None
    # A foreground occluder legitimately disagrees with the reference, so only the
    # unoccluded cells are allowed to veto the re-verification.
    _, grid_good, _ = _occlusion_map(frame_gray, ref, H, cfg)
    occ_mask = _grid_good_mask(grid_good, (fh, fw), ref.tex_size, H, cfg)
    agree = (mask > 0)
    if occ_mask is not None:
        agree &= occ_mask
    if int(agree.sum()) < 0.15 * int((mask > 0).sum()):
        return None
    ncc = _ncc(cv2.GaussianBlur(warped, (5, 5), 0), cv2.GaussianBlur(frame_gray, (5, 5), 0),
               agree)
    if ncc is None or ncc < 0.45:
        return None
    return H, ncc


def _grid_good_mask(grid_good: list, frame_shape, tex_size, H_tex_frame, cfg) -> np.ndarray | None:
    """Rasterise the occlusion grid's 'good' cells back onto the frame."""
    if not grid_good:
        return None
    fh, fw = frame_shape
    tw, th = tex_size
    gx, gy = cfg.occlusion_grid
    cw, ch = tw / gx, th / gy
    out = np.zeros((fh, fw), np.uint8)
    for j in range(gy):
        for i in range(gx):
            k = j * gx + i
            if k >= len(grid_good) or not grid_good[k]:
                continue
            cell = np.array([[i * cw, j * ch], [(i + 1) * cw, j * ch],
                             [(i + 1) * cw, (j + 1) * ch], [i * cw, (j + 1) * ch]], np.float64)
            poly = np.round(apply_h(H_tex_frame, cell)).astype(np.int32)
            cv2.fillConvexPoly(out, poly, 255)
    return out > 0


@dataclass
class Observation:
    H: np.ndarray
    quad: np.ndarray
    pts_tex: np.ndarray
    pts_img: np.ndarray
    ref: ReferenceModel
    n_inliers: int
    n_ratio: int
    n_gated: int
    coverage: float
    reproj: float
    corner_scores: list
    corner_visible: list
    corner_sigma: list
    occlusion: float
    grid_good: list
    rejected: list
    reject_counts: dict
    photometric_ncc: float = 0.0
    expected_coverage: float = 1.0


def _match_candidates(ref: ReferenceModel, gray: np.ndarray, H_pred, cfg: TrackConfig,
                      gate_px, roi_mask):
    sift = cv2.SIFT_create(nfeatures=cfg.sift_features, contrastThreshold=cfg.sift_contrast,
                           edgeThreshold=cfg.sift_edge)
    kp, des = sift.detectAndCompute(gray, roi_mask)
    if des is None or len(kp) < 6:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(ref.desc, des, k=2)
    ref_sel, img_sel, rejected = [], [], []
    counts = {"ratio": 0, "gate": 0}
    for pair in knn:
        if len(pair) < 2:
            m = pair[0]
            ip = np.array(kp[m.trainIdx].pt)
        else:
            m, n2 = pair
            ip = np.array(kp[m.trainIdx].pt)
            if m.distance >= cfg.ratio * n2.distance:
                counts["ratio"] += 1
                if len(rejected) < cfg.max_rejected_points:
                    rejected.append({"p": [round(float(ip[0]), 1), round(float(ip[1]), 1)],
                                     "reason": "ratio"})
                continue
        if gate_px is not None and H_pred is not None:
            pred = apply_h(H_pred, ref.pts_tex[m.queryIdx].reshape(1, 2))[0]
            if float(np.linalg.norm(pred - ip)) > gate_px:
                counts["gate"] += 1
                if len(rejected) < cfg.max_rejected_points:
                    rejected.append({"p": [round(float(ip[0]), 1), round(float(ip[1]), 1)],
                                     "reason": "gate"})
                continue
        ref_sel.append(ref.pts_tex[m.queryIdx])
        img_sel.append(ip)
    if len(ref_sel) < cfg.min_matches:
        return None
    return np.array(ref_sel, np.float64), np.array(img_sel, np.float64), rejected, counts


def _solve_observation(frame: np.ndarray, ref: ReferenceModel, cfg: TrackConfig,
                       H_pred, gate_px, roi_mask, search_px: float,
                       occ_ref: ReferenceModel | None = None) -> Observation | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    matched = _match_candidates(ref, gray, H_pred, cfg, gate_px, roi_mask)
    if matched is None:
        return None
    pts_tex, pts_img, rejected, counts = matched
    H, mask = cv2.findHomography(pts_tex, pts_img, cv2.RANSAC, cfg.ransac_px,
                                 maxIters=3000, confidence=0.995)
    if H is None or mask is None or mask.sum() < 4:
        return None
    mask = mask.ravel().astype(bool)
    if not _quad_valid(quad_from_h(H, *ref.tex_size), frame.shape[:2], ref.quad):
        return None

    H_kf_cur = H @ np.linalg.inv(ref.H_tex)
    fh, fw = gray.shape
    warped = cv2.warpPerspective(ref.gray, H_kf_cur, (fw, fh), flags=cv2.INTER_LINEAR)
    wgray = cv2.GaussianBlur(warped, (3, 3), 0)
    fgray = cv2.GaussianBlur(gray, (3, 3), 0)
    keep = np.zeros(len(pts_tex), bool)
    ncc_vals = []
    for i in range(len(pts_tex)):
        if not mask[i]:
            continue
        ip = pts_img[i]
        ncc = _local_ncc(wgray, fgray, int(round(ip[0])), int(round(ip[1])), half=6)
        if ncc is not None and ncc >= cfg.photometric_ncc:
            keep[i] = True
            ncc_vals.append(ncc)
        else:
            counts["photometric"] = counts.get("photometric", 0) + 1
            if len(rejected) < cfg.max_rejected_points:
                rejected.append({"p": [round(float(ip[0]), 1), round(float(ip[1]), 1)],
                                 "reason": "photometric"})
    if keep.sum() < 4:
        return None
    rp, ip = pts_tex[keep], pts_img[keep]
    H = homography_dlt(rp, ip)
    if not np.isfinite(H).all() or not _quad_valid(quad_from_h(H, *ref.tex_size),
                                                   frame.shape[:2], ref.quad):
        return None

    search = float(np.clip(cfg.corner_search_frac * np.sqrt(polygon_area(ref.quad)) * 2, 8.0, 48.0))
    if H_pred is not None:
        shift = np.linalg.norm(apply_h(H, ref.corner_pts_tex.mean(axis=0).reshape(1, 2)) -
                               apply_h(H_pred, ref.corner_pts_tex.mean(axis=0).reshape(1, 2)))
        search = float(np.clip(search + shift, 8.0, max(60.0, search_px)))
    corner_scores, corner_visible, _ = _refine_corners(gray, ref, H, cfg, search)

    proj = apply_h(H, rp)
    resid = np.linalg.norm(proj - ip, axis=1)
    coverage = _grid_coverage(rp, ref.tex_size, cfg.coverage_grid)
    corner_sigma, _ = homography_covariance(H, rp, resid, ref.corner_pts_tex)
    # Occlusion is estimated against the anchor (frame 0) whenever possible: it is
    # always clean, whereas a promote keyframe can contain a passed occluder.
    occlusion, grid_good, _ = _occlusion_map(gray, occ_ref or ref, H, cfg)
    # visible area of the screen in this frame (handles panels leaving the image)
    tw, th = ref.tex_size
    wmask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), H, (fw, fh),
                                flags=cv2.INTER_NEAREST)
    area_frac = float((wmask > 0).sum()) / float(tw * th) if tw * th else 1.0
    expected = float(np.clip(area_frac * (1.0 - occlusion), 0.04, 1.0))
    n_store = min(len(rp), cfg.max_feature_points)
    idx = np.linspace(0, len(rp) - 1, n_store).astype(int) if n_store else []
    return Observation(
        H=H, quad=corner_order(quad_from_h(H, *ref.tex_size)), pts_tex=rp[idx], pts_img=ip[idx],
        ref=ref, n_inliers=int(len(rp)), n_ratio=len(pts_tex), n_gated=counts.get("gate", 0),
        coverage=float(coverage), reproj=float(np.median(resid)),
        corner_scores=corner_scores, corner_visible=corner_visible, corner_sigma=corner_sigma,
        occlusion=float(occlusion), grid_good=grid_good, rejected=rejected,
        reject_counts=counts, photometric_ncc=float(np.mean(ncc_vals)) if ncc_vals else 0.0,
        expected_coverage=expected)


def _need_more_than_score(obs: Observation) -> float:
    """Coverage score that does not punish a screen that is genuinely partly hidden."""
    return float(np.clip(obs.coverage / max(0.18, 0.9 * obs.expected_coverage), 0.0, 1.0))


def _foreground_polys(record: dict, anchor: ReferenceModel, tex_size, grid) -> list[list]:
    """Polygons of occluded screen regions, used to keep foreground in front."""
    if record.get("quad") is None or not record.get("grid_good"):
        return []
    occ_ratio = record.get("occlusion")
    if occ_ratio is not None and occ_ratio < 0.04:
        return []
    gx, gy = grid
    good = record["grid_good"]
    tw, th = tex_size
    cw, ch = tw / gx, th / gy
    H = np.array(record["H"]) if record.get("H") is not None else h_from_quad(
        corner_order(np.array(record["quad"])), tw, th)
    polys = []
    for j in range(gy):
        for i in range(gx):
            idx = j * gx + i
            if idx < len(good) and good[idx]:
                continue
            cell = np.array([[i * cw, j * ch], [(i + 1) * cw, j * ch],
                             [(i + 1) * cw, (j + 1) * ch], [i * cw, (j + 1) * ch]], np.float64)
            poly = apply_h(H, cell)
            if np.isfinite(poly).all():
                polys.append(np.round(poly, 1).tolist())
    return polys


def _predict_quad(history: deque, frame_idx: int) -> np.ndarray | None:
    if not history:
        return None
    t_last, q_last = history[-1]
    age = frame_idx - t_last
    if age == 0:
        return q_last.copy()
    if len(history) == 1:
        return q_last.copy()
    t_prev, q_prev = history[-2]
    dt = max(1, t_last - t_prev)
    vel = (q_last - q_prev) / dt
    quad = q_last + vel * age
    if len(history) >= 3:
        t_p2, q_p2 = history[-3]
        dt2 = max(1, t_prev - t_p2)
        acc = (vel - (q_prev - q_p2) / dt2) / max(1.0, (t_last - t_p2) / 2.0)
        quad = quad + 0.30 * acc * (age ** 2)
    return quad


def _precision(obs: Observation, cfg: TrackConfig, ref: ReferenceModel):
    diag = max(1e-6, float(np.sqrt(polygon_area(ref.quad))))
    max_sigma_frac = max(obs.corner_sigma) / diag
    n_score = float(np.clip((obs.n_inliers - 6) / 34.0, 0.0, 1.0))
    cov_score = _need_more_than_score(obs)
    err_score = float(np.exp(-obs.reproj / 2.6))
    unc_score = float(np.exp(-max_sigma_frac / 0.030))
    conf = 100.0 * (0.22 * n_score + 0.38 * cov_score + 0.20 * err_score + 0.20 * unc_score)
    precise = (obs.coverage >= cfg.min_coverage_full and obs.reproj <= cfg.max_reproj and
               max_sigma_frac <= cfg.max_corner_sigma_frac and obs.n_inliers >= 12)
    if precise:
        label = "full"
    elif obs.coverage >= 0.25 and obs.reproj <= cfg.max_reproj * 1.6:
        label = "partial"
    else:
        label = "weak"
    return precise, float(np.clip(conf, 0.0, 100.0)), label, max_sigma_frac


def _roi_mask(shape, quad, pad):
    if quad is None:
        return None
    h, w = shape
    c = quad.mean(axis=0)
    poly = np.round((quad - c) * (1 + pad) + c).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    return mask


def _record_stub(frame_idx: int, pred_quad):
    return {
        "frame": frame_idx, "state": "paused", "precision": "none", "quad": None,
        "confidence": 0.0, "occlusion": None, "inliers": [], "inlier_count": 0,
        "rejected": [], "reject_counts": {}, "coverage": 0.0, "reproj_median": 0.0,
        "corner_sigma": [0.0, 0.0, 0.0, 0.0], "corner_visible": [False] * 4,
        "corner_scores": [0.0] * 4, "decision": None, "note": "",
        "grid_good": [], "H": None,
        "predicted_quad": pred_quad.tolist() if pred_quad is not None else None,
        "keyframe": None, "corners_found": 0, "photometric_ncc": None,
        "sigma_frac": None,
    }


def _fill_observation(record: dict, obs: Observation, conf: float, label: str, sigma_frac: float):
    record.update({
        "inliers": [[round(float(x), 1), round(float(y), 1)] for x, y in obs.pts_img],
        "inlier_count": obs.n_inliers, "rejected": obs.rejected,
        "reject_counts": obs.reject_counts, "coverage": round(obs.coverage, 3),
        "reproj_median": round(obs.reproj, 3),
        "corner_sigma": [round(s, 2) for s in obs.corner_sigma],
        "corner_visible": [bool(v) for v in obs.corner_visible],
        "corner_scores": [round(s, 3) for s in obs.corner_scores],
        "occlusion": round(obs.occlusion, 3), "grid_good": obs.grid_good,
        "H": obs.H.tolist(), "quad": obs.quad.tolist(), "precision": label,
        "confidence": round(conf, 1), "sigma_frac": round(sigma_frac, 4),
        "photometric_ncc": round(obs.photometric_ncc, 3), "corners_found": int(sum(obs.corner_visible)),
        "keyframe": obs.ref.frame_idx,
    })


def track_path(source: str, start_frame: int, quad: np.ndarray, cfg: TrackConfig,
               out_path: str | None = None, progress_cb=None,
               max_frames: int | None = None) -> dict:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {source}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end_frame = total if max_frames is None else min(total, start_frame + max_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, frame = cap.read()
    if not ok:
        raise ValueError("cannot read the anchor frame")
    quad0 = corner_order(np.asarray(quad, np.float64))
    anchor = build_reference(frame, quad0, cfg, start_frame)
    key_ref = anchor
    tex_size = anchor.tex_size
    gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frames_out = []
    anc = _record_stub(start_frame, None)
    anc.update({
        "state": "anchor", "precision": "full", "quad": quad0.tolist(), "confidence": 100.0,
        "occlusion": 0.0, "coverage": 1.0, "corner_visible": [True] * 4,
        "corner_scores": [1.0] * 4, "decision": "observe", "corners_found": 4,
        "note": "manual anchor / reference", "inlier_count": len(anchor.pts_tex),
        "grid_good": [True] * (cfg.occlusion_grid[0] * cfg.occlusion_grid[1]),
        "H": anchor.H_tex.tolist(),
        "inliers": [[round(float(x), 1), round(float(y), 1)] for x, y in
                    apply_h(anchor.H_tex, anchor.pts_tex)[:cfg.max_feature_points]],
        "keyframe": start_frame,
    })
    frames_out.append(anc)

    history = deque(maxlen=4)
    history.append((start_frame, quad0.copy()))
    last_obs_frame = start_frame
    last_precise = True
    age = 0
    prev_state = "anchor"
    events = []
    occluded_since = None
    paused_since = None
    stats = {"tracking": 0, "weak": 0, "tracking_partial": 0, "predicted": 0, "paused": 0,
             "reacquired": 0, "total": 0}
    last_key_idx = start_frame
    last_quad = quad0.copy()

    i = start_frame + 1
    while i < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pred_quad = _predict_quad(history, i)
        pred_H = h_from_quad(pred_quad, *tex_size) if pred_quad is not None else None
        diag = float(np.linalg.norm(quad0[2] - quad0[0]))
        gate_frac = min(cfg.gate_frac_max, cfg.gate_frac + cfg.gate_frac_slope * age)
        gate_px = gate_frac * diag if age <= cfg.max_predict + 2 else None
        search_hint = _roi_mask((height, width), pred_quad, 0.06 + 0.02 * min(age, 10)) \
            if pred_quad is not None and age <= 2 else None
        corner_search = float(np.clip((cfg.corner_search_frac * (1 + 0.12 * age)) * diag, 10, 150))

        obs = None
        refs_to_try = [key_ref] if key_ref is anchor else [key_ref, anchor]
        for ref in refs_to_try:
            obs = _solve_observation(frame, ref, cfg, pred_H if ref is key_ref else None,
                                     gate_px if ref is key_ref else None,
                                     search_hint if ref is key_ref else None, corner_search,
                                     occ_ref=anchor)
            if obs is not None:
                key_ref = ref
                break

        record = _record_stub(i, pred_quad)
        accepted = False
        if obs is not None:
            precise, conf, label, sigma_frac = _precision(obs, cfg, anchor)
            _fill_observation(record, obs, conf, label, sigma_frac)
            strong_support = (obs.n_inliers >= cfg.min_inliers and
                              obs.reproj <= cfg.max_reproj * 1.6)
            occ_cov = _need_more_than_score(obs)
            n_vis = int(sum(obs.corner_visible))
            # Concentrated evidence (all inliers in one part of the screen) is not the
            # same as missing evidence. If the screen is visible and the propagated corner
            # uncertainty stays small, keep observing but label the plane as weakly
            # supported instead of pausing. Occlusion still drives predict/pause below.
            visible_ok = (obs.occlusion < cfg.occlusion_hidden and
                          obs.reproj <= cfg.max_reproj and
                          sigma_frac <= cfg.max_corner_sigma_frac)
            # A handful of inliers in a thin visible sliver can fit any homography,
            # so the corner uncertainty has to stay bounded for the plane to count.
            well_conditioned = sigma_frac <= cfg.max_corner_sigma_frac
            observable = (strong_support and n_vis >= 2 and well_conditioned and
                          (occ_cov >= 0.55 or obs.coverage >= 0.5 or visible_ok))
            if observable:
                accepted = True
                if obs.occlusion >= cfg.occlusion_hidden:
                    record.update(state="tracking_partial", precision="partial",
                                  decision="observe",
                                  note=f"screen over half occluded ({obs.occlusion:.0%}); "
                                       f"{n_vis}/4 corners verified, using the partial observation")
                    last_precise = False
                elif precise:
                    record.update(state="tracking", precision="full", decision="observe")
                    last_precise = True
                else:
                    record.update(state="weak", precision="partial", decision="observe",
                                  note="observed, but the inliers are concentrated in one part of "
                                       f"the screen (coverage {obs.coverage:.0%}, "
                                       f"{obs.n_inliers} inliers); the far corners are not "
                                       "independently supported")
                    last_precise = False
                if prev_state in ("paused", "predicted") and (i - last_obs_frame) >= 3:
                    events.append({"frame": i, "event": "reacquired",
                                   "gap": i - last_obs_frame, "corners_verified": n_vis})
                    record["state"] = "reacquired"
                    record["note"] = f"re-verified after {i - last_obs_frame} frames"
                    stats["reacquired"] += 1
                if occluded_since is not None and obs.occlusion < cfg.occlusion_clear:
                    events.append({"frame": i, "event": "occlusion_cleared",
                                   "hidden_frames": i - occluded_since})
                    occluded_since = None
            else:
                if occ_cov < 0.4 or n_vis <= 1:
                    if occluded_since is None:
                        occluded_since = i
                if last_precise and age < cfg.max_predict:
                    record.update(state="predicted", precision="predicted",
                                  confidence=round(float(np.clip(conf * 0.4, 5, 40)), 1),
                                  decision="predict", quad=record["predicted_quad"],
                                  note="too little clean evidence; using the planar prediction")
                    age += 1
                    prev_state = "predicted"
                else:
                    record.update(state="paused", precision="none", confidence=0.0,
                                  decision="pause", quad=None,
                                  note="too little clean evidence; overlay paused")
                    age += 1
                    prev_state = "paused"
            if accepted:
                last_obs_frame = i
                last_quad = obs.quad.copy()
                history.append((i, obs.quad.copy()))
                age = 0
                prev_state = record["state"]
        else:
            # no feature match: try the retained corner templates
            hint = pred_quad if pred_quad is not None else last_quad
            scores, found = _corner_search(gray, anchor, cfg, hint, corner_search, scale_hint=1.0)
            record["corner_scores"] = [round(s, 3) for s in scores]
            record["corner_visible"] = [bool(s >= cfg.corner_score) for s in scores]
            n_found = int(sum(record["corner_visible"]))
            record["corners_found"] = n_found
            solved = None
            if n_found == 4 and all(f is not None for f in found):
                solved = _solve_corner_quad(gray, anchor, cfg, np.array(found), pred_H)
            if solved is not None:
                H, ncc = solved
                quad = corner_order(quad_from_h(H, *tex_size))
                record.update({
                    "quad": quad.tolist(), "H": H.tolist(), "state": "reacquired",
                    "precision": "partial", "decision": "corner_reverify",
                    "confidence": round(float(np.clip(45 + 40 * ncc, 20, 85)), 1),
                    "photometric_ncc": round(float(ncc), 3),
                    "note": "fourth corner visible; re-verified with the retained corner "
                            "templates and a full-screen photometric check",
                })
                record.update(_corner_solve_metrics(frame, gray, anchor, cfg, H, ncc))
                events.append({"frame": i, "event": "reacquired",
                               "gap": i - last_obs_frame, "corners_verified": 4})
                stats["reacquired"] += 1
                accepted = True
                last_obs_frame = i
                last_quad = quad.copy()
                history.append((i, quad.copy()))
                age = 0
                last_precise = False
                prev_state = "reacquired"
                if occluded_since is not None:
                    events.append({"frame": i, "event": "occlusion_cleared",
                                   "hidden_frames": i - occluded_since})
                    occluded_since = None
            else:
                if occluded_since is None:
                    occluded_since = i
                if n_found == 3:
                    pending_note = ("screen occluded; 3/4 corners visible, waiting for the fourth "
                                    "corner before re-verification")
                else:
                    pending_note = f"no reliable match; {n_found}/4 corners visible"
                if last_precise and age < cfg.max_predict:
                    record.update(state="predicted", precision="predicted",
                                  confidence=round(float(np.clip(42 - 4.0 * age, 5, 42)), 1),
                                  decision="await_corner" if n_found == 3 else "predict",
                                  quad=record["predicted_quad"],
                                  note=f"{pending_note}; planar prediction age {age + 1}/"
                                       f"{cfg.max_predict}")
                    age += 1
                    prev_state = "predicted"
                else:
                    record.update(state="paused", precision="none", confidence=0.0,
                                  decision="pause", quad=None,
                                  note=f"{pending_note}; prediction horizon exhausted, overlay paused")
                    age += 1
                    prev_state = "paused"

        if record["state"] in ("tracking", "weak", "tracking_partial", "reacquired"):
            stats[record["state"]] = stats.get(record["state"], 0) + 1
        else:
            stats[record["state"]] = stats.get(record["state"], 0) + 1
        stats["total"] += 1
        record["foreground"] = _foreground_polys(record, anchor, tex_size, cfg.occlusion_grid)
        frames_out.append(record)

        # promote a keyframe from a confidently tracked frame, but only when the
        # screen is essentially unoccluded: otherwise the occluder is baked into
        # the reference texture and later frames read it as occlusion.
        if (isinstance(record["state"], str) and record["state"] in ("tracking", "reacquired") and
                record.get("inlier_count", 0) >= cfg.rekey_min_inliers and
                record.get("coverage", 0) >= 0.5 and
                (record.get("occlusion") or 0.0) <= cfg.rekey_max_occlusion and
                (i - last_key_idx) >= cfg.rekey_every):
            try:
                key_ref = build_reference(frame, np.array(record["quad"]), cfg, i,
                                          tex_size=tex_size, anchor=anchor)
                last_key_idx = i
            except ValueError:
                pass

        i += 1
        if progress_cb and (i - start_frame) % 5 == 0:
            progress_cb(i - start_frame, end_frame - start_frame, record["state"])

    cap.release()

    result = {
        "version": 1,
        "source": os.path.basename(source),
        "fps": fps,
        "frame_size": [width, height],
        "total_frames_in_source": total,
        "start_frame": start_frame,
        "end_frame": i,
        "reference": {"frame": start_frame, "quad": quad0.tolist(),
                      "texture_size": list(tex_size), "n_features": int(len(anchor.pts_tex))},
        "config": cfg.to_dict(),
        "frames": frames_out,
        "events": events,
        "stats": stats,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f)
        os.replace(tmp, out_path)
    if progress_cb:
        progress_cb(end_frame - start_frame, end_frame - start_frame, "done")
    return result


def _corner_solve_metrics(frame, gray, anchor, cfg, H, ncc):
    quad = corner_order(quad_from_h(H, *anchor.tex_size))
    pts_tex, pts_img, rejected, counts = [], [], [], {}
    H_kf_cur = H @ np.linalg.inv(anchor.H_tex)
    fh, fw = gray.shape
    warped = cv2.warpPerspective(anchor.gray, H_kf_cur, (fw, fh))
    wg = cv2.GaussianBlur(warped, (3, 3), 0)
    fg = cv2.GaussianBlur(gray, (3, 3), 0)
    nccs = []
    for cp in anchor.corner_pts_tex:
        ip = apply_h(H, cp.reshape(1, 2))[0]
        pts_tex.append(cp)
        pts_img.append(ip)
        v = _local_ncc(wg, fg, int(round(ip[0])), int(round(ip[1])), 6)
        if v is not None:
            nccs.append(v)
    occ, good, _ = _occlusion_map(gray, anchor, H, cfg)
    sigmas, _ = homography_covariance(H, np.array(pts_tex), np.zeros(4), anchor.corner_pts_tex)
    return {
        "inliers": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in pts_img],
        "inlier_count": 4, "rejected": [], "reject_counts": {"corner_only": 1},
        "coverage": round(float(_grid_coverage(anchor.corner_pts_tex, anchor.tex_size,
                                               cfg.coverage_grid)), 3),
        "reproj_median": 0.0, "corner_sigma": [round(s, 2) for s in sigmas],
        "grid_good": good, "occlusion": round(occ, 3),
        "sigma_frac": round(float(max(sigmas) / max(1e-6, np.sqrt(polygon_area(anchor.quad)))), 4),
        "keyframe": anchor.frame_idx, "photometric_ncc": round(float(ncc), 3),
    }


def quads_from_track(track: dict) -> dict[int, np.ndarray | None]:
    out = {}
    for fr in track.get("frames", []):
        q = fr.get("quad")
        out[fr["frame"]] = np.array(q, np.float64) if q is not None else None
    return out
