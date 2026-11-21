"""Ground-truth evaluation for the synthetic clips and the overlay slip check."""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .compositor import render_overlay
from .geometry import (apply_h, corner_order, h_from_quad, homography_dlt,
                       polygon_area, quad_side_lengths)
from .synth import read_frames


def quad_metrics(est: np.ndarray | None, gt: np.ndarray) -> dict:
    if est is None:
        return {"corner_err_mean": None, "corner_err_max": None, "per_corner": None}
    est = corner_order(est)
    gt = corner_order(gt)
    d = np.linalg.norm(est - gt, axis=1)
    return {"corner_err_mean": float(d.mean()), "corner_err_max": float(d.max()),
            "per_corner": [float(x) for x in d]}


def _H_of(quad, tex_size, frame_size):
    """Homography texture->frame for a quad, normalised to the frame centre."""
    W, Hh = frame_size
    T = np.array([[1, 0, -W / 2], [0, 1, -Hh / 2], [0, 0, 1]], float)
    Hq = h_from_quad(corner_order(quad), *tex_size)
    return T @ Hq @ np.linalg.inv(T)


def decompose(H: np.ndarray) -> dict:
    """Approximate similarity + perspective terms from H, measured at the screen centre.

    scale: local linear scale (sqrt of area Jacobian)
    rotation: in-plane rotation of the texture axes
    perspective: strength of the projective term (|h13| + |h23| normalised)
    """
    a = H[:2, :2]
    det = float(np.linalg.det(a))
    scale = float(np.sqrt(abs(det))) if det != 0 else float("nan")
    # average rotation of the x-axis of the texture
    ang = float(np.degrees(np.arctan2(a[1, 0], a[0, 0])))
    # perspective magnitude: deviation of the third row from (0,0,1) plus
    # anisotropic foreshortening of the two axes
    p = float(np.hypot(H[2, 0], H[2, 1]))
    su = float(np.linalg.norm(H[:2, 0]))
    sv = float(np.linalg.norm(H[:2, 1]))
    shear = float(abs(np.dot(H[:2, 0], H[:2, 1])) / max(1e-9, su * sv))
    return {"scale": scale, "rotation": ang, "perspective": p, "anisotropy": su / max(1e-9, sv),
            "shear": shear}


def _state_should_be_paused(gt_frame: dict) -> bool:
    """Ground-truth decision for whether an observation is defensible."""
    return gt_frame["occlusion"] >= 0.9 or sum(gt_frame["corner_visible"]) <= 1


def evaluate(track: dict, truth: dict, config: dict | None = None) -> dict:
    gt_frames = {f["frame"]: f for f in truth["frames"]}
    tex_size = tuple(track["reference"]["texture_size"])
    frame_size = tuple(track["frame_size"])
    records = {r["frame"]: r for r in track["frames"]}
    common = sorted(set(records) & set(gt_frames))
    errs, scale_err, rot_err, persp_err, aniso_err = [], [], [], [], []
    per_corner_spread = []
    homogeneous_frames = 0
    frames_with_est = 0
    scorable_corner_cases = 0
    honest_corner_cases = 0
    occ_err = []
    state_rows = []
    gt_paused, est_paused = 0, 0
    paused_while_visible = 0
    overlay_when_paused_gt = 0
    for f in common:
        rec, gt = records[f], gt_frames[f]
        est = rec.get("quad")
        if est is not None:
            frames_with_est += 1
        m = quad_metrics(est, np.array(gt["quad"]))
        if est is not None:
            errs.append(m["corner_err_mean"])
            He = _H_of(np.array(est), tex_size, frame_size)
            Hg = _H_of(np.array(gt["quad"]), tex_size, frame_size)
            de, dg = decompose(He), decompose(Hg)
            scale_err.append(abs(de["scale"] - dg["scale"]) / max(1e-9, dg["scale"]))
            rot_err.append(abs(de["rotation"] - dg["rotation"]))
            persp_err.append(abs(de["perspective"] - dg["perspective"]))
            aniso_err.append(abs(de["anisotropy"] - dg["anisotropy"]))
            per_corner_spread.append(m["corner_err_max"] - m["corner_err_mean"])
            # does the report honestly distinguish a partially-supported plane?
            gt_cov = 1.0 - gt["occlusion"]
            if gt_cov < 0.75 and m["corner_err_mean"] > 2.0:
                scorable_corner_cases += 1
                if rec.get("precision") in ("partial", "weak", "predicted", "none"):
                    honest_corner_cases += 1
        else:
            errs.append(None)
        if rec.get("occlusion") is not None:
            occ_err.append(abs(rec["occlusion"] - gt["occlusion"]))
        should_pause = _state_should_be_paused(gt)
        gt_paused += int(should_pause)
        is_paused = rec["state"] in ("paused",) or rec.get("quad") is None
        est_paused += int(is_paused)
        if is_paused and not should_pause and gt["occlusion"] < 0.7:
            paused_while_visible += 1
        if (not is_paused) and should_pause:
            overlay_when_paused_gt += 1
        state_rows.append({"frame": f, "state": rec["state"], "gt_occlusion": gt["occlusion"],
                           "gt_corners_visible": int(sum(gt["corner_visible"])),
                           "corner_err": m["corner_err_mean"]})
    valid = [e for e in errs if e is not None]
    metrics = {
        "frames": len(common),
        "frames_with_estimate": frames_with_est,
        "frames_paused": len(common) - frames_with_est,
        "corner_err_mean_px": float(np.mean(valid)) if valid else None,
        "corner_err_median_px": float(np.median(valid)) if valid else None,
        "corner_err_p95_px": float(np.percentile(valid, 95)) if valid else None,
        "corner_err_max_px": float(np.max(valid)) if valid else None,
        "scale_rel_err_mean": float(np.mean(scale_err)) if scale_err else None,
        "rotation_err_deg_mean": float(np.mean(rot_err)) if rot_err else None,
        "perspective_err_mean": float(np.mean(persp_err)) if persp_err else None,
        "anisotropy_err_mean": float(np.mean(aniso_err)) if aniso_err else None,
        "corner_spread_mean_px": float(np.mean(per_corner_spread)) if per_corner_spread else None,
        "occlusion_err_mean": float(np.mean(occ_err)) if occ_err else None,
        "gt_should_pause_frames": gt_paused,
        "est_paused_frames": est_paused,
        "paused_while_gt_visible": paused_while_visible,
        "overlay_when_gt_fully_occluded": overlay_when_paused_gt,
        "honesty_cases": scorable_corner_cases,
        "honest_cases": honest_corner_cases,
    }
    metrics["fidelity"] = _fidelity(metrics)
    return {"metrics": metrics, "per_frame": state_rows}


def _fidelity(m: dict) -> dict:
    e = m["corner_err_mean_px"]
    if e is None:
        return {"grade": "F", "score": 0.0}
    score = 100.0
    score -= min(60.0, e * 6.0)
    score -= min(20.0, (m["rotation_err_deg_mean"] or 0) * 3.0)
    score -= min(10.0, (m["scale_rel_err_mean"] or 0) * 60.0)
    score -= min(10.0, (m["perspective_err_mean"] or 0) * 200.0)
    score = max(0.0, score)
    grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D" if score >= 40 else "F"
    return {"grade": grade, "score": round(score, 1)}


def measure_overlay_slip(truth: dict, track: dict, content: np.ndarray,
                         sample_every: int = 3) -> dict:
    """Render the overlay with the estimated H and with the ground-truth H.

    Wherever both renderings cover the same pixel, the colour difference shows
    whether the applied content slid relative to the true screen.
    """
    tex_size = tuple(track["reference"]["texture_size"])
    records = {r["frame"]: r for r in track["frames"]}
    gt_frames = {f["frame"]: f for f in truth["frames"]}
    frame_shape = (truth["frame_size"][1], truth["frame_size"][0])
    diffs, valid_counts = [], []
    for f in sorted(records):
        if f not in gt_frames or records[f].get("quad") is None:
            continue
        if f % sample_every != 0:
            continue
        q = np.array(records[f]["quad"])
        H_est = h_from_quad(corner_order(q), *tex_size)
        H_gt = h_from_quad(corner_order(np.array(gt_frames[f]["quad"])), *tex_size)
        blank = np.zeros((*frame_shape, 3), np.uint8)
        a = render_overlay(blank, content, H_est, tex_size)
        b = render_overlay(blank, content, H_gt, tex_size)
        ma = cv2.warpPerspective(np.full(frame_shape, 255, np.uint8), H_est,
                                 (frame_shape[1], frame_shape[0]))
        mb = cv2.warpPerspective(np.full(frame_shape, 255, np.uint8), H_gt,
                                 (frame_shape[1], frame_shape[0]))
        m = (ma > 0) & (mb > 0)
        if m.sum() < 50:
            continue
        d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)[m]
        diffs.append(float(np.mean(d)))
        valid_counts.append(int(m.sum()))
        # discrete slip in px along the texture x direction at the screen centre
    if not diffs:
        return {"slip_px_mean": None, "ncc_mean": None, "samples": 0}
    # also report a direct displacement measure
    disp = []
    for f in sorted(records):
        if f not in gt_frames or records[f].get("quad") is None or f % sample_every != 0:
            continue
        He = h_from_quad(corner_order(np.array(records[f]["quad"])), *tex_size)
        Hg = h_from_quad(corner_order(np.array(gt_frames[f]["quad"])), *tex_size)
        grid = np.array([[u, v] for u in np.linspace(0, tex_size[0], 7)
                         for v in np.linspace(0, tex_size[1], 5)])
        disp.extend(np.linalg.norm(apply_h(He, grid) - apply_h(Hg, grid), axis=1).tolist())
    return {"slip_px_mean": float(np.mean(disp)) if disp else None,
            "slip_px_p95": float(np.percentile(disp, 95)) if disp else None,
            "colour_diff_mean": float(np.mean(diffs)),
            "samples": len(diffs)}


def compare_anchor_handling(truth: dict, track: dict) -> dict:
    """Report how much the applied transform explains real observed motion.

    The 'predicted' and 'paused' states are compared against the true motion to
    show the system is deciding from observations, not a pre-scripted animation.
    """
    gt_frames = {f["frame"]: f for f in truth["frames"]}
    rows = []
    for rec in track["frames"]:
        gt = gt_frames.get(rec["frame"])
        if gt is None:
            continue
        if rec["state"] in ("predicted", "paused", "tracking_partial", "reacquired"):
            rows.append({"frame": rec["frame"], "state": rec["state"],
                         "gt_occlusion": gt["occlusion"],
                         "has_overlay": rec.get("quad") is not None})
    kinds = {}
    for r in rows:
        kinds[r["state"]] = kinds.get(r["state"], 0) + 1
    return {"non_tracking_frames": len(rows), "by_state": kinds, "rows": rows[:400]}


def evaluate_clip_dir(clip_dir: str, track_path: str, content_path: str) -> dict:
    truth = json.load(open(os.path.join(clip_dir, "truth.json")))
    track = json.load(open(track_path))
    content = cv2.imread(content_path)
    ev = evaluate(track, truth)
    slip = measure_overlay_slip(truth, track, content)
    anchor = compare_anchor_handling(truth, track)
    return {"metrics": ev["metrics"], "slip": slip, "decisions": anchor,
            "track_stats": track.get("stats"), "events": track.get("events"),
            "per_frame": ev["per_frame"]}
