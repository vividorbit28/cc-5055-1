"""Minimal reproduction of the homography-degeneracy defect and its repair.

Two failures are demonstrated on synthetic, fully controlled inputs:

1. Near-collinear correspondences. Inlier residuals on the band stay sub-pixel,
   so a residual-only acceptance path keeps a homography whose off-line
   extrapolation is unconstrained. The 2-D point-spread gate in
   `tracker._homography_degenerate` rejects the solve; a well-spread control on
   the same motion is accepted.

2. Short occlusion on the narrow-angle `narrow_hand` clip. While a hand covers
   half the panel the tracker opens a track break and withholds the overlay;
   when enough independent corner evidence returns it relinks. The export shows
   the break range, the first frame where re-texturing is re-permitted, and
   monotonic observation-generation ids (a failed frame is never re-announced
   as a fresh success).

Run with the project venv:  .venv/bin/python -I repro/minimal_degeneracy_repro.py
The clip-backed part skips itself when data/clips/narrow_hand has not been
generated; the geometry part always runs.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from screentrack import tracker as T
from screentrack.geometry import (apply_h, homography_condition,
                                  homography_dlt, point_spread_2d, quad_from_h)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collinear_band_demo():
    print("== 1. near-collinear correspondences ==")
    rng = np.random.default_rng(4)
    cfg = T.TrackConfig()
    # texture points span the plane; their image projections collapse onto a
    # thin strip (edge-on screen or an occluder leaving only a sliver)
    tex = np.array([[20, 20], [120, 24], [220, 19], [60, 120], [180, 116],
                    [260, 123], [90, 160], [240, 158]], np.float64)
    base = np.column_stack([40 + 0.9 * tex[:, 0], 60 + 0.12 * tex[:, 0]])
    band_img = base + rng.normal(0, 0.5, tex.shape)

    H = homography_dlt(tex, band_img)
    resid = np.linalg.norm(apply_h(H, tex) - band_img, axis=1)
    print(f"  unguarded DLT: inlier residual max {resid.max():.3f}px (looks fine)")
    print(f"  but quad extrapolation: {np.round(quad_from_h(H, 320, 180), 1).tolist()}")
    print(f"  point spread tex/img: {point_spread_2d(tex):.3f} / "
          f"{point_spread_2d(band_img):.3f}  (healthy set is order 1+)")

    gate = T._homography_degenerate(tex, band_img, cfg)
    assert gate is not None and gate["reason"] == "near_collinear_correspondences"
    print(f"  repaired gate: REJECTED -> {gate}")

    # control: the same underlying motion with a well-spread image set
    healthy_img = band_img + np.outer(rng.uniform(0, 1, len(band_img)), [0, 90])
    cond = homography_condition(tex, healthy_img)
    assert T._homography_degenerate(tex, healthy_img, cfg) is None
    print(f"  well-spread control: accepted (dst_spread {cond['dst_spread']:.3f})")

    # exact affine motion is structurally rank-poor in the DLT design matrix
    # but never degenerate when the points tile the plane
    H_aff = np.array([[1.02, 0.01, 6.0], [0.02, 0.97, 4.0], [0, 0, 1.0]])
    aff_tex = np.column_stack([rng.uniform(10, 310, 24), rng.uniform(10, 170, 24)])
    aff_img = apply_h(H_aff, aff_tex)
    assert homography_condition(aff_tex, aff_img)["design_min_sv"] < 1e-3
    assert T._homography_degenerate(aff_tex, aff_img, cfg) is None
    print("  exact-affine well-spread set: design SV ~ 0 but NOT rejected")


def short_occlusion_demo():
    clip = os.path.join(ROOT, "data", "clips", "narrow_hand")
    if not os.path.exists(os.path.join(clip, "source.mp4")):
        print("== 2. short occlusion: skipped (narrow_hand clip not generated) ==")
        return
    print("== 2. short occlusion -> break / relink on narrow_hand ==")
    truth = json.load(open(os.path.join(clip, "truth.json")))
    track = T.track_path(os.path.join(clip, "source.mp4"), 0,
                         np.array(truth["frames"][0]["quad"]), T.TrackConfig(),
                         max_frames=len(truth["frames"]))
    gt = {f["frame"]: f for f in truth["frames"]}
    recs = {r["frame"]: r for r in track["frames"]}
    assert track["relinks"], "expected at least one break -> relink cycle"
    for link in track["relinks"]:
        bf, rf = link["break_frame"], link["frame"]
        # every break frame withholds the overlay; the relink frame re-permits it
        gap_states = [recs[f]["state"] for f in range(bf, rf + 1)]
        assert all(recs[f]["quad"] is None or recs[f]["state"] == "predicted"
                   for f in range(bf, rf))
        assert recs[rf]["quad"] is not None
        err = float(np.mean(np.linalg.norm(
            np.array(recs[rf]["quad"]) - np.array(gt[rf]["quad"]), axis=1)))
        print(f"  break {bf} -> relink {rf} (gap {link['gap']}) states {gap_states}, "
              f"relink corner error {err:.2f}px")
    gens = [r["observation_generation"] for r in track["frames"]
            if r["observation_generation"] is not None]
    assert gens == sorted(gens) and len(set(gens)) == len(gens)
    paused = [r["frame"] for r in track["frames"] if r["state"] == "paused"]
    assert all(recs[f]["quad"] is None for f in paused)
    print(f"  paused frames {paused}: overlay withheld, generation ids monotonic, "
          f"open break: {track['open_break']}")


if __name__ == "__main__":
    collinear_band_demo()
    short_occlusion_demo()
    print("\nOK: degeneracy is rejected, occlusion opens an honest break and relinks.")
