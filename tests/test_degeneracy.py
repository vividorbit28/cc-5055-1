"""Regression tests for the homography-degeneracy repair and track-break
relink bookkeeping.

Two synthetic screen sequences drive the end-to-end behaviour:
  * narrow_hand: a narrow-angle (near edge-on) screen with a hand covering
    half the panel; surviving features approach a collinear band and the
    tracker must open breaks and relink without pasting an invalid homography.
  * cut_move: a source cut mid-clip that swaps the displayed material and
    relocates the panel; tracking must start a fresh leg from the new shot.

The geometry-level degeneracy gate is additionally exercised on controlled
near-collinear correspondences, where the inlier residuals stay sub-pixel but
the off-line extrapolation is unconstrained.
"""
import json
import os

import cv2
import numpy as np
import pytest

from screentrack import tracker as T
from screentrack.geometry import (apply_h, h_from_quad, homography_condition,
                                  point_spread_2d)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(ROOT, "data", "clips")


def _clip_available(name):
    return os.path.exists(os.path.join(CLIPS, name, "source.mp4"))


needs_clips = pytest.mark.skipif(not _clip_available("narrow_hand"),
                                 reason="degeneracy regression clips not generated")


def _track(name):
    truth = json.load(open(os.path.join(CLIPS, name, "truth.json")))
    cfg = T.TrackConfig()
    q0 = np.array(truth["frames"][0]["quad"])
    res = T.track_path(os.path.join(CLIPS, name, "source.mp4"), 0, q0, cfg,
                       max_frames=len(truth["frames"]))
    return truth, res


def _textured_frame(rng, shape=(360, 640)):
    """A dense random texture so SIFT finds plenty of features inside the quad."""
    gray = rng.integers(0, 255, shape, dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------
# geometry-level degeneracy gate
# --------------------------------------------------------------------------

def test_point_spread_detects_collinear_band():
    rng = np.random.default_rng(5)
    healthy = np.column_stack([rng.uniform(0, 320, 40), rng.uniform(0, 180, 40)])
    band = np.column_stack([rng.uniform(0, 320, 40), rng.uniform(86, 94, 40)])
    line = np.column_stack([rng.uniform(0, 320, 12), np.full(12, 90.0)])
    assert point_spread_2d(healthy) > 1.0
    assert point_spread_2d(band) < 0.5
    assert point_spread_2d(line) < 0.05


def test_degeneracy_gate_rejects_near_collinear_but_keeps_healthy():
    cfg = T.TrackConfig()
    rng = np.random.default_rng(7)
    Ht = np.array([[1.0, 0.0, 50.0], [0.0, 0.9, 90.0],
                   [0.0004, 0.0002, 1.0]])
    band_tex = np.column_stack([rng.uniform(10, 310, 16), rng.uniform(86, 94, 16)])
    band_img = apply_h(Ht, band_tex) + rng.normal(0, 0.4, band_tex.shape)
    deg = T._homography_degenerate(band_tex, band_img, cfg)
    assert deg is not None and deg["reason"] == "near_collinear_correspondences"
    # the diagnostic carries real measurements, not zeroed placeholders
    assert deg["n_points"] == 16 and deg["dst_spread"] < cfg.min_point_spread
    healthy_tex = np.column_stack([rng.uniform(0, 320, 30), rng.uniform(0, 180, 30)])
    healthy_img = apply_h(Ht, healthy_tex) + rng.normal(0, 0.4, healthy_tex.shape)
    assert T._homography_degenerate(healthy_tex, healthy_img, cfg) is None


def test_degenerate_solve_returns_none_and_reports_diagnostic(monkeypatch):
    """A frame whose photometric inliers collapse onto a band must not yield
    an observation. The matcher is forced to return a near-collinear band whose
    along-band residuals are tiny, i.e. the case reprojection error cannot catch."""
    cfg = T.TrackConfig()
    rng = np.random.default_rng(11)
    quad = np.array([[40, 60], [600, 60], [600, 300], [40, 300]], np.float64)
    frame0 = _textured_frame(rng)
    ref = T.build_reference(frame0, quad, cfg, 0)
    shift = np.array([[1.0, 0.0, 6.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]])
    frame1 = cv2.warpPerspective(frame0, shift, (640, 360))
    # y confined to an 8px band inside the texture -> near-collinear inliers
    band_tex = np.column_stack([rng.uniform(10, ref.tex_size[0] - 10, 24),
                                rng.uniform(84, 92, 24)])
    band_img = apply_h(shift @ ref.H_tex, band_tex)

    def fake_match(*_args, **_kw):
        return band_tex, band_img, [], {"ratio": 0, "gate": 0}

    monkeypatch.setattr(T, "_match_candidates", fake_match)
    diag = {}
    obs = T._solve_observation(frame1, ref, cfg, ref.H_tex, None, None, 40.0, diag=diag)
    assert obs is None
    assert diag["degeneracy"]["reason"] == "near_collinear_correspondences"
    assert diag["n_photometric_inliers"] >= cfg.min_inliers

    # a well-spread inlier set on the same frames solves normally
    spread_tex = np.column_stack([rng.uniform(10, ref.tex_size[0] - 10, 30),
                                  rng.uniform(10, ref.tex_size[1] - 10, 30)])
    spread_img = apply_h(shift @ ref.H_tex, spread_tex)

    def healthy_match(*_args, **_kw):
        return spread_tex, spread_img, [], {"ratio": 0, "gate": 0}

    monkeypatch.setattr(T, "_match_candidates", healthy_match)
    obs = T._solve_observation(frame1, ref, cfg, ref.H_tex, None, None, 40.0)
    assert obs is not None and obs.n_inliers >= 10
    assert np.median(np.linalg.norm(apply_h(obs.H, spread_tex) - spread_img, axis=1)) < 3.0


def test_homography_condition_reports_rank_poor_design():
    rng = np.random.default_rng(3)
    Ht = np.array([[1.0, 0.0, 40.0], [0.0, 0.8, 80.0], [0.0, 0.0, 1.0]])
    line_tex = np.column_stack([rng.uniform(0, 320, 12), np.full(12, 90.0)])
    line_img = apply_h(Ht, line_tex) + rng.normal(0, 0.2, line_tex.shape)
    cond = homography_condition(line_tex, line_img)
    # the collinear set is caught by the 2-D spread; an exact affine motion
    # would also drive design_min_sv to ~0, so the spread cue must fire
    assert cond["src_spread"] < 0.05
    assert T._homography_degenerate(line_tex, line_img, T.TrackConfig()) is not None


def test_pure_affine_well_spread_set_is_not_degenerate():
    """An exactly affine homography makes the normalized DLT design matrix
    rank-deficient, yet a well-spread inlier set fully determines the affine
    part. The gate must not reject this legitimate planar motion."""
    rng = np.random.default_rng(9)
    H_aff = np.array([[1.02, 0.01, 6.0], [0.02, 0.97, 4.0], [0.0, 0.0, 1.0]])
    tex = np.column_stack([rng.uniform(10, 310, 30), rng.uniform(10, 170, 30)])
    img = apply_h(H_aff, tex)
    cond = homography_condition(tex, img)
    assert cond["design_min_sv"] < 1e-3  # rank-poor by construction ...
    assert T._homography_degenerate(tex, img, T.TrackConfig()) is None  # ... but not degenerate


# --------------------------------------------------------------------------
# end-to-end narrow-angle / half-screen occlusion
# --------------------------------------------------------------------------

@needs_clips
def test_narrow_hand_opens_and_relinks_breaks():
    truth, res = _track("narrow_hand")
    # the sequence must contain at least one explicit break -> relink cycle
    assert res["relinks"], "expected a track-break relink in narrow_hand"
    recs = {r["frame"]: r for r in res["frames"]}
    for link in res["relinks"]:
        bf, rf = link["break_frame"], link["frame"]
        assert bf < rf and link["gap"] == rf - bf
        # the break frame carries no accepted observation
        assert recs[bf]["state"] in ("predicted", "paused")
        # the relink frame is a real observation with a fresh generation id
        assert recs[rf]["state"] in ("tracking", "weak", "reacquired", "tracking_partial")
        assert recs[rf]["observation_generation"] is not None
        assert recs[rf]["relink"] is not None and recs[rf]["relink"]["relink_frame"] == rf


@needs_clips
def test_narrow_hand_never_pastes_invalid_homography():
    """Every frame that exposes an overlay must sit on the true screen; failed
    frames must withhold the overlay rather than show a stale frame."""
    truth, res = _track("narrow_hand")
    gt = {f["frame"]: f for f in truth["frames"]}
    for r in res["frames"]:
        g = gt[r["frame"]]
        if r["quad"] is None:
            assert r["state"] == "paused"
            continue
        err = float(np.mean(np.linalg.norm(np.array(r["quad"]) - np.array(g["quad"]), axis=1)))
        # a degenerate homography would throw the corners tens of px off
        assert err < 12.0, f"frame {r['frame']} corner error {err:.1f}px"


@needs_clips
def test_observation_generation_monotonic_and_strictly_observed():
    """Generation ids only appear on accepted observations and never repeat;
    a failed frame can never be announced as a new successful track."""
    _, res = _track("narrow_hand")
    gens = [r["observation_generation"] for r in res["frames"]
            if r["observation_generation"] is not None]
    assert gens == sorted(gens) and len(set(gens)) == len(gens)
    for r in res["frames"]:
        if r["state"] in ("predicted", "paused"):
            assert r["observation_generation"] is None


# --------------------------------------------------------------------------
# source cut: new material + moved panel -> fresh tracking leg
# --------------------------------------------------------------------------

@needs_clips
def test_cut_move_detects_cut_and_relinks_from_new_shot():
    truth, res = _track("cut_move")
    cut = truth["cut_frame"]
    assert cut in res["source_cuts"]
    recs = {r["frame"]: r for r in res["frames"]}
    # the cut frame itself is a break, not a successful track
    cut_rec = recs[cut]
    assert cut_rec["source_cut"] is True
    assert cut_rec["state"] in ("predicted", "paused")
    # the first frame after the cut relinks using the new material
    post_cut = [e for e in res["relinks"] if e["frame"] > cut]
    assert post_cut, "expected a relink after the source cut"
    first_relink = min(e["frame"] for e in post_cut)
    assert first_relink == cut + 1
    assert recs[first_relink]["quad"] is not None
    assert recs[first_relink]["relink"]["via_source_cut"] is True


@needs_clips
def test_cut_move_tracks_moved_panel_after_relink():
    """After relinking onto the new shot the overlay must follow the relocated
    panel for the rest of the clip, with bounded reprojection error."""
    truth, res = _track("cut_move")
    cut = truth["cut_frame"]
    gt = {f["frame"]: f for f in truth["frames"]}
    recs = {r["frame"]: r for r in res["frames"]}
    checked = 0
    for f in range(cut + 1, len(truth["frames"])):
        r = recs[f]
        if r["quad"] is None:
            continue
        err = float(np.mean(np.linalg.norm(np.array(r["quad"]) - np.array(gt[f]["quad"]), axis=1)))
        assert err < 8.0, f"post-cut frame {f} error {err:.1f}px"
        checked += 1
    assert checked >= (len(truth["frames"]) - cut) * 0.8


@needs_clips
def test_cut_move_keeps_pre_cut_track_unchanged():
    """Frames before the cut must remain normally tracked (the repair does not
    regress clean tracking), verified against reprojection and valid state."""
    truth, res = _track("cut_move")
    cut = truth["cut_frame"]
    gt = {f["frame"]: f for f in truth["frames"]}
    for r in res["frames"]:
        if r["frame"] >= cut:
            break
        assert r["quad"] is not None and r["state"] in (
            "anchor", "tracking", "weak", "tracking_partial", "reacquired")
        err = float(np.mean(np.linalg.norm(np.array(r["quad"]) -
                                           np.array(gt[r["frame"]]["quad"]), axis=1)))
        assert err < 4.0


# --------------------------------------------------------------------------
# failure-state propagation and evidence export
# --------------------------------------------------------------------------

@needs_clips
def test_failure_evidence_exported_per_frame():
    _, res = _track("narrow_hand")
    for r in res["frames"]:
        for key in ("source_cut", "track_break", "relink",
                    "observation_generation", "degeneracy"):
            assert key in r
    # every opened break either relinks later or is the open break at the end
    break_starts = {r["track_break"]["start"] for r in res["frames"] if r["track_break"]}
    relinked = {link["break_frame"] for link in res["relinks"]}
    open_start = res["open_break"]["start"] if res["open_break"] else None
    assert break_starts <= (relinked | ({open_start} if open_start is not None else set()))


@needs_clips
def test_reproj_and_valid_state_are_consistent():
    """Observed frames carry real inlier/reproj evidence; a paused overlay must
    be withheld with zero confidence rather than announced as a success. A
    rejected observation may still leave diagnostic inliers behind, but it can
    never populate the quad or the observation generation."""
    _, res = _track("narrow_hand")
    for r in res["frames"]:
        if r["state"] == "paused":
            assert r["quad"] is None
            assert r["confidence"] == 0.0
            assert r["observation_generation"] is None
            assert r["precision"] == "none"
        if r["state"] in ("tracking", "weak", "tracking_partial", "reacquired"):
            assert r["quad"] is not None
            assert r["inlier_count"] >= 4
            assert r["confidence"] > 0.0
            assert r["observation_generation"] is not None
            assert r["reproj_median"] >= 0.0
