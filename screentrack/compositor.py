"""Render the replacement content into the tracked screen and produce diagnostics.

The compositor only consumes the saved track JSON. It never re-runs the
tracker, so a failed frame can be reviewed without touching the source video.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .geometry import (apply_h, corner_order, h_from_quad, polygon_area,
                       quad_from_h)
from .synth import encode_frames_ffmpeg, read_frames

STATE_COLORS = {
    "anchor": (0, 200, 0),
    "tracking": (0, 220, 0),
    "tracking_partial": (0, 190, 255),
    "weak": (0, 160, 255),
    "predicted": (255, 160, 0),
    "paused": (0, 0, 255),
    "reacquired": (255, 0, 220),
}


def smooth_quads(quads: dict[int, np.ndarray | None],
                 states: dict[int, str]) -> dict[int, np.ndarray | None]:
    """Small temporal median filter over consecutive observations.

    Only applied to frames backed by real observations; predicted frames are
    left as they are so the diagnostics show exactly what the model produced.
    """
    frames = sorted(quads)
    out = dict(quads)
    observed = {f for f in frames if states.get(f) in ("tracking", "weak", "tracking_partial",
                                                       "reacquired")}
    for f in frames:
        if f not in observed:
            continue
        window = [quads[g] for g in (f - 1, f, f + 1) if g in observed and quads[g] is not None]
        if len(window) >= 3:
            arr = np.stack(window)
            out[f] = np.median(arr, axis=0)
    return out


def render_overlay(frame: np.ndarray, content: np.ndarray, H_tex: np.ndarray,
                   tex_size: tuple[int, int], opacity: float = 1.0,
                   fit: str = "stretch") -> np.ndarray:
    """Warp `content` into the screen area defined by H_tex (texture -> frame)."""
    h, w = frame.shape[:2]
    ch, cw = content.shape[:2]
    tw, th = tex_size
    Tw, Th = tw / float(cw), th / float(ch)
    content_H = np.array([[Tw, 0, 0], [0, Th, 0], [0, 0, 1.0]])
    if fit == "cover":
        s = max(tw / cw, th / ch)
        content_H = np.array([[s, 0, -(cw * s - tw) / 2], [0, s, -(ch * s - th) / 2], [0, 0, 1]])
    elif fit == "contain":
        s = min(tw / cw, th / ch)
        content_H = np.array([[s, 0, (tw - cw * s) / 2], [0, s, (th - ch * s) / 2], [0, 0, 1]])
    Hc = H_tex @ content_H
    warped = cv2.warpPerspective(content, Hc, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    alpha = cv2.warpPerspective(np.full((ch, cw), 255, np.uint8), Hc, (w, h),
                                flags=cv2.INTER_LINEAR) * float(opacity)
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    return np.clip(frame.astype(np.float32) * (1 - a) + warped.astype(np.float32) * a,
                   0, 255).astype(np.uint8)


def _foreground_mask(shape, polys: list, dilate: int = 3) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    for poly in polys or []:
        pts = np.round(np.array(poly)).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillConvexPoly(mask, pts, 255)
    if mask.any() and dilate > 0:
        mask = cv2.dilate(mask, np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8))
    return mask


def _draw_hud(frame: np.ndarray, rec: dict, gt: dict | None, show_features: bool,
              show_rejected: bool) -> np.ndarray:
    out = frame.copy()
    state = rec["state"]
    color = STATE_COLORS.get(state, (200, 200, 200))
    quad = rec.get("quad")
    pred = rec.get("predicted_quad")
    if pred is not None and state in ("predicted", "paused", "weak", "tracking_partial", "reacquired"):
        qp = np.round(np.array(pred, np.int32))
        cv2.polylines(out, [qp], True, (255, 140, 0), 1, cv2.LINE_AA)
    if quad is not None:
        q = np.round(np.array(quad, np.int32))
        cv2.polylines(out, [q], True, color, 2, cv2.LINE_AA)
        for k, (x, y) in enumerate(quad):
            vis = rec.get("corner_visible", [True] * 4)[k]
            cv2.circle(out, (int(x), int(y)), 4, color if vis else (0, 0, 255), -1, cv2.LINE_AA)
    if show_features:
        for x, y in rec.get("inliers", []):
            cv2.circle(out, (int(x), int(y)), 2, (0, 255, 0), -1, cv2.LINE_AA)
    if show_rejected:
        for r in rec.get("rejected", []):
            x, y = r["p"]
            cv2.drawMarker(out, (int(x), int(y)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 7, 1,
                           cv2.LINE_AA)
    if gt is not None:
        qg = np.round(np.array(gt["quad"], np.int32))
        cv2.polylines(out, [qg], True, (255, 255, 0), 1, cv2.LINE_AA)

    lines = [
        f"frame {rec['frame']:03d}  state {state.upper()}  decision {rec.get('decision') or '-'}",
        f"inliers {rec.get('inlier_count', 0)}  rejected {len(rec.get('rejected', []))}  "
        f"coverage {rec.get('coverage', 0):.0%}  reproj {rec.get('reproj_median', 0):.2f}px",
        f"occlusion {rec['occlusion']:.0%}  confidence {rec.get('confidence', 0):.0f}/100  "
        f"corners {rec.get('corners_found', sum(rec.get('corner_visible', [])))}/4"
        if rec.get("occlusion") is not None else
        f"occlusion n/a  confidence {rec.get('confidence', 0):.0f}/100  "
        f"corners {rec.get('corners_found', 0)}/4",
    ]
    if gt is not None:
        err = float(np.mean(np.linalg.norm(np.array(gt["quad"]) - np.array(quad), axis=1))) \
            if quad is not None else float("nan")
        lines.append(f"GT occlusion {gt['occlusion']:.0%}   manual-check corner error "
                     f"{'n/a' if np.isnan(err) else f'{err:.1f}px'}")
    y0 = 18
    cv2.rectangle(out, (6, 4), (min(out.shape[1] - 6, 560), 18 * len(lines) + 10), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (12, y0 + i * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240),
                    1, cv2.LINE_AA)
    if rec.get("note"):
        cv2.putText(out, rec["note"][:86], (12, 18 * len(lines) + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def composite_track(source: str, track: dict, content: np.ndarray, outdir: str,
                    truth: dict | None = None, opacity: float = 1.0,
                    fit: str = "stretch", draw_features: bool = True,
                    draw_rejected: bool = True, smooth: bool = True) -> dict:
    os.makedirs(outdir, exist_ok=True)
    frames = read_frames(source)
    records = {r["frame"]: r for r in track["frames"]}
    tex_size = tuple(track["reference"]["texture_size"])
    quads = {f: (np.array(r["quad"]) if r.get("quad") is not None else None) for f, r in records.items()}
    states = {f: r["state"] for f, r in records.items()}
    if smooth:
        quads = smooth_quads(quads, states)
    gt_frames = {r["frame"]: r for r in truth["frames"]} if truth else {}

    out_frames, diag_frames, content_frames = [], [], []
    used = 0
    for idx, frame in enumerate(frames):
        rec = records.get(idx)
        if rec is None:
            continue
        q = quads.get(idx)
        if q is not None:
            H = h_from_quad(corner_order(q), *tex_size)
            composed = render_overlay(frame, content, H, tex_size, opacity=opacity, fit=fit)
            fg = _foreground_mask(frame.shape, rec.get("foreground", []))
            if fg.any():
                composed[fg > 0] = frame[fg > 0]
            used += 1
        else:
            composed = frame.copy()
        gt = None
        if truth is not None:
            gt = gt_frames.get(idx)
            if gt is not None:
                gt = {"quad": gt["quad"], "occlusion": gt["occlusion"]}
        diag = _draw_hud(frame, rec, gt, draw_features, draw_rejected)
        out_frames.append(composed)
        diag_frames.append(diag)
        if smooth:
            content_frames.append(composed)

    base = os.path.basename(outdir.rstrip("/")) or "output"
    out_path = os.path.join(outdir, "composite.mp4")
    diag_path = os.path.join(outdir, "diagnostics.mp4")
    preview_path = os.path.join(outdir, "preview.jpg")
    encode_frames_ffmpeg(out_frames, out_path, int(track["fps"]))
    encode_frames_ffmpeg(diag_frames, diag_path, int(track["fps"]))
    if diag_frames:
        cv2.imwrite(preview_path, diag_frames[len(diag_frames) // 2])
    first_used = next((i for i in sorted(records) if quads.get(i) is not None), None)
    return {"video": os.path.basename(out_path), "diagnostics": os.path.basename(diag_path),
            "preview": os.path.basename(preview_path), "frames_with_overlay": used,
            "frames_total": len(frames)}
