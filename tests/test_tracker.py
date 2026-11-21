import json
import os

import cv2
import numpy as np
import pytest

from screentrack import tracker as T
from screentrack.geometry import apply_h, h_from_quad
from screentrack.synth import read_frames

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(ROOT, "data", "clips")


def _clip_available(name):
    return os.path.exists(os.path.join(CLIPS, name, "source.mp4"))


def _load(name):
    truth = json.load(open(os.path.join(CLIPS, name, "truth.json")))
    return truth, read_frames(os.path.join(CLIPS, name, "source.mp4"))


needs_clips = pytest.mark.skipif(not _clip_available("corner_texture"),
                                 reason="synthetic clips not generated")


def test_homography_covariance_grows_with_less_support():
    """The same geometric configuration with fewer points must be less certain."""
    rng = np.random.default_rng(0)
    tex = np.array([[0, 0], [300, 0], [300, 170], [0, 170]], np.float64)
    H = h_from_quad(tex + rng.normal(0, 2, tex.shape), 300, 170)
    many = np.column_stack([rng.uniform(0, 300, 200), rng.uniform(0, 170, 200)])
    few = many[:6]
    corners = np.array([[0, 0], [300, 0], [300, 170], [0, 170]], np.float64)
    s_many, _ = T.homography_covariance(H, many, rng.normal(0, 0.4, 200), corners)
    s_few, _ = T.homography_covariance(H, few, rng.normal(0, 0.4, 6), corners)
    assert np.mean(s_few) > np.mean(s_many)


def test_corner_concentrated_coverage_is_detected():
    """Inliers in one corner must not be reported as full-plane support."""
    pts = np.array([[10, 10], [30, 20], [50, 30], [20, 50], [70, 60]], np.float64)
    cov = T._grid_coverage(pts, (300, 170), 4)
    assert cov <= 0.25
    gx, gy = np.meshgrid(np.linspace(20, 280, 4), np.linspace(20, 150, 4))
    spread = np.column_stack([gx.ravel(), gy.ravel()])
    assert T._grid_coverage(spread, (300, 170), 4) == 1.0


def test_occlusion_estimator_perfect_frame():
    """A frame that equals the reference must report no occlusion."""
    truth, frames = _load("orbit_clean")
    cfg = T.TrackConfig()
    ref = T.build_reference(frames[0], np.array(truth["frames"][0]["quad"]), cfg, 0)
    gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    occ, good, _ = T._occlusion_map(gray, ref, ref.H_tex, cfg)
    assert occ < 0.02
    assert all(good)


@needs_clips
def test_occlusion_estimator_tracks_person():
    """While the person is on the screen the estimate must be clearly non-zero."""
    truth, frames = _load("occlusion")
    cfg = T.TrackConfig()
    ref = T.build_reference(frames[0], np.array(truth["frames"][0]["quad"]), cfg, 0)
    worst = 0.0
    for idx in range(88, 106, 5):
        g = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
        H = h_from_quad(np.array(truth["frames"][idx]["quad"]), *ref.tex_size)
        occ, _, _ = T._occlusion_map(g, ref, H, cfg)
        gt = truth["frames"][idx]["occlusion"]
        worst = max(worst, abs(occ - gt))
        assert occ > 0.3
        assert abs(occ - gt) < 0.1
    assert worst < 0.18


@needs_clips
def test_tracker_does_not_absorb_person_into_homography():
    """During the crossing the estimated quad must stay on the real screen."""
    truth, frames = _load("occlusion")
    cfg = T.TrackConfig()
    q0 = np.array(truth["frames"][0]["quad"])
    res = T.track_path(os.path.join(CLIPS, "occlusion", "source.mp4"), 0, q0, cfg,
                       max_frames=130)
    tracked = {r["frame"]: r for r in res["frames"]}
    checked = 0
    for idx in range(88, 106):
        rec = tracked.get(idx)
        if rec is None or rec.get("quad") is None:
            continue
        err = np.mean(np.linalg.norm(np.array(rec["quad"]) -
                                     np.array(truth["frames"][idx]["quad"]), axis=1))
        assert err < 10.0, f"frame {idx} quad drifted {err:.1f}px during occlusion"
        checked += 1
    assert checked >= 8


@needs_clips
def test_concentrated_texture_keeps_overlay_but_flags_weak():
    """A visible screen with one-corner texture should be tracked and labelled weak,
    not paused: missing perspective support is a confidence issue, not occlusion."""
    truth, _ = _load("corner_texture")
    cfg = T.TrackConfig()
    q0 = np.array(truth["frames"][0]["quad"])
    res = T.track_path(os.path.join(CLIPS, "corner_texture", "source.mp4"), 0, q0, cfg,
                       max_frames=90)
    states = [r["state"] for r in res["frames"]]
    assert states.count("paused") <= 10
    assert states.count("weak") >= 60
    weak = [r for r in res["frames"] if r["state"] == "weak"]
    assert np.median([r["coverage"] for r in weak]) <= 0.5
    assert all(r["confidence"] > 0 for r in weak)


@needs_clips
def test_rectified_texture_is_upright_and_textured():
    """Rectifying the anchor must recover the screen image, not a degenerate warp.

    The inverse direction is easy to get wrong and silently produces a black
    image, which then makes the corner templates flat and their match scores
    meaningless (TM_CCOEFF_NORMED returns 1.0 for a flat template anywhere).
    """
    truth, frames = _load("orbit_clean")
    cfg = T.TrackConfig()
    q = np.array(truth["frames"][0]["quad"])
    ref = T.build_reference(frames[0], q, cfg, 0)
    gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    tex = T._rectified_texture(gray, q, ref.tex_size)
    assert float(tex.std()) > 20.0
    assert (tex > 10).mean() > 0.8
    # every corner probe must have real texture to be verifiable
    stds = [float(p.std()) for p in ref.corner_patches]
    assert min(stds) > 6.0


@needs_clips
def test_corner_reverify_recovers_quad():
    """With all four probes visible the retained-corner path must find the plane."""
    truth, frames = _load("occlusion")
    cfg = T.TrackConfig()
    anchor = T.build_reference(frames[0], np.array(truth["frames"][0]["quad"]), cfg, 0)
    idx = 130
    gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
    gtq = np.array(truth["frames"][idx]["quad"])
    Hgt = h_from_quad(gtq, *anchor.tex_size)
    hint = apply_h(Hgt, anchor.corner_pts_tex)
    scores, found = T._corner_search(gray, anchor, cfg, hint, 20.0, 1.0)
    assert sum(1 for s in scores if s >= cfg.corner_score) == 4
    solved = T._solve_corner_quad(gray, anchor, cfg, np.array(found), Hgt)
    assert solved is not None
    H, _ = solved
    from screentrack.geometry import quad_from_h, corner_order
    q = corner_order(quad_from_h(H, *anchor.tex_size))
    assert np.mean(np.linalg.norm(q - gtq, axis=1)) < 4.0


@needs_clips
def test_clean_clip_accuracy_and_device_motion():
    truth, _ = _load("orbit_clean")
    cfg = T.TrackConfig()
    q0 = np.array(truth["frames"][0]["quad"])
    res = T.track_path(os.path.join(CLIPS, "orbit_clean", "source.mp4"), 0, q0, cfg,
                       max_frames=70)
    errs = []
    for r in res["frames"]:
        if r["quad"] is None:
            errs.append(0.0)
            continue
        gtq = np.array(truth["frames"][r["frame"]]["quad"])
        errs.append(np.mean(np.linalg.norm(np.array(r["quad"]) - gtq, axis=1)))
    assert np.median(errs) < 1.5
    assert res["stats"]["paused"] == 0
    # perspective really is changing in this clip, so the estimate is not pure translation
    first = np.array(res["frames"][0]["quad"])
    last = np.array(res["frames"][-1]["quad"])
    sides_first = np.linalg.norm(np.roll(first, -1, axis=0) - first, axis=1)
    sides_last = np.linalg.norm(np.roll(last, -1, axis=0) - last, axis=1)
    assert np.max(np.abs(sides_last / sides_first - 1)) > 0.15
