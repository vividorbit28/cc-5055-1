"""Synthetic benchmark clips with known camera motion, screen texture and occluders.

The screen is a 1.6 x 0.9 m plane at world z=0.  World axes: x right, y up,
z toward the camera.  The camera looks at the screen from +z, so screen texture
pixel (0,0) is the top-left physical corner.  Ground-truth per-frame data is
written to truth.json next to the encoded clip.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import apply_h, homography_dlt

SCREEN_W, SCREEN_H = 640, 360
SCREEN_W_M, SCREEN_H_M = 1.6, 0.9
FRAME_W, FRAME_H = 640, 360
FPS = 24


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_screen_texture(kind: str = "standard", seed: int = 7) -> np.ndarray:
    """A display image with plenty of trackable structure."""
    rng = _rng(seed)
    img = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
    if kind == "corner":
        # Deliberately feature-poor: only the top-left logo panel is detailed.
        img[:] = (58, 62, 66)
        grad = np.linspace(-10, 10, SCREEN_W, dtype=np.float32)
        img = np.clip(img.astype(np.float32) + grad[None, :, None], 0, 255).astype(np.uint8)
        panel = img[18:150, 18:225].copy()
        prng = _rng(seed + 3)
        panel[:] = (30, 34, 38)
        for _ in range(40):
            x, y = int(prng.integers(2, 205)), int(prng.integers(2, 130))
            w, h = int(prng.integers(3, 26)), int(prng.integers(3, 22))
            cv2.rectangle(panel, (x, y), (x + w, y + h),
                          tuple(int(c) for c in prng.integers(90, 255, 3)), -1)
        cv2.circle(panel, (70, 66), 34, (240, 240, 240), 3)
        cv2.circle(panel, (70, 66), 18, (60, 190, 250), -1)
        cv2.putText(panel, "LOGO", (108, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        img[18:150, 18:225] = panel
        return img

    # standard display content
    img[:] = (36, 44, 60)
    cv2.rectangle(img, (0, 0), (SCREEN_W, 46), (24, 80, 150), -1)
    cv2.putText(img, "MARKET WATCH", (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(img, "14:32", (SCREEN_W - 118, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2)

    # fake chart
    base = np.array([[30, 250], [120, 200], [210, 225], [300, 150], [390, 170], [480, 95], [600, 120]],
                    np.int32)
    cv2.polylines(img, [base], False, (60, 210, 220), 3, cv2.LINE_AA)
    for x, y in base[::2]:
        cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1)
    for gy in range(90, 300, 34):
        cv2.line(img, (14, gy), (SCREEN_W - 14, gy), (52, 62, 80), 1)

    # headline text lines
    for i in range(4):
        y = 86 + i * 22
        cv2.putText(img, ["EUROPE STOCKS OPEN HIGHER", "CHIP DEMAND BEATS FORECAST",
                          "VOLATILITY FALLS FOR A THIRD SESSION", "FX: EURO STEADY AGAINST DOLLAR"][i],
                    (300, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 238, 245), 1, cv2.LINE_AA)

    # high-frequency blocks for scale/rotation cues
    chip_colors = [(230, 70, 60), (70, 205, 90), (80, 140, 245), (240, 200, 60), (200, 90, 220)]
    for i in range(18):
        x = int(rng.integers(30, SCREEN_W - 60))
        y = int(rng.integers(220, SCREEN_H - 40))
        w = int(rng.integers(14, 46))
        h = int(rng.integers(10, 26))
        cv2.rectangle(img, (x, y), (x + w, y + h), chip_colors[i % len(chip_colors)], -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (250, 250, 250), 1)
    for _ in range(26):
        x, y = int(rng.integers(5, SCREEN_W - 5)), int(rng.integers(55, SCREEN_H - 5))
        r = int(rng.integers(1, 4))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)

    # small ticker at the bottom
    cv2.rectangle(img, (0, SCREEN_H - 26), (SCREEN_W, SCREEN_H), (20, 22, 26), -1)
    cv2.putText(img, "DAX +0.84   CAC +0.61   FTSE +0.42   SMI +0.18", (12, SCREEN_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 240, 150), 1, cv2.LINE_AA)

    # subtle display texture (keeps flat areas from being perfectly flat)
    noise = rng.normal(0, 3.0, (SCREEN_H, SCREEN_W, 3))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def make_content_texture(seed: int = 11) -> np.ndarray:
    """Replacement content: high-contrast test pattern so slip is obvious."""
    img = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
    step = 40
    for y in range(0, SCREEN_H, step):
        for x in range(0, SCREEN_W, step):
            shade = 195 if ((x // step) + (y // step)) % 2 == 0 else 55
            img[y:y + step, x:x + step] = (shade, shade, shade)
    cv2.rectangle(img, (8, 8), (SCREEN_W - 8, SCREEN_H - 8), (255, 60, 60), 4)
    cv2.circle(img, (SCREEN_W // 2, SCREEN_H // 2), 92, (30, 120, 255), 5, cv2.LINE_AA)
    cv2.line(img, (SCREEN_W // 2 - 130, SCREEN_H // 2), (SCREEN_W // 2 + 130, SCREEN_H // 2),
             (30, 120, 255), 3, cv2.LINE_AA)
    cv2.line(img, (SCREEN_W // 2, SCREEN_H // 2 - 74), (SCREEN_W // 2, SCREEN_H // 2 + 74),
             (30, 120, 255), 3, cv2.LINE_AA)
    cv2.putText(img, "LIVE TEST FEED", (36, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 3)
    cv2.putText(img, "LIVE TEST FEED", (36, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 1)
    for i in range(12):
        x = 36 + i * 48
        cv2.rectangle(img, (x, SCREEN_H - 52), (x + 30, SCREEN_H - 22),
                      ((i * 47) % 255, (i * 91) % 255, (i * 63) % 255), -1)
    return img


def animate_content(base: np.ndarray, frame_idx: int) -> np.ndarray:
    """Same content plus a timecode and a sweeping marker (for slip checks)."""
    out = base.copy()
    cv2.putText(out, f"{frame_idx:04d}", (SCREEN_W - 168, SCREEN_H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    x = 24 + (frame_idx * 7) % (SCREEN_W - 48)
    cv2.circle(out, (x, 300), 9, (0, 255, 0), -1, cv2.LINE_AA)
    return out


def _rounded_rect(mask: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], radius: int,
                  value: int = 255) -> None:
    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.rectangle(mask, (x1 + r, y1), (x2 - r, y2), value, -1)
    cv2.rectangle(mask, (x1, y1 + r), (x2, y2 - r), value, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(mask, (cx, cy), r, value, -1)


def make_person_sprite(seed: int = 21) -> np.ndarray:
    """A textured person-like billboard (BGRA) that produces many strong features."""
    rng = _rng(seed)
    w, h = 240, 540
    bgra = np.zeros((h, w, 4), np.uint8)
    mask = np.zeros((h, w), np.uint8)

    # legs
    _rounded_rect(mask, (78, 330), (118, 520), 16)
    _rounded_rect(mask, (126, 330), (166, 520), 16)
    # torso
    _rounded_rect(mask, (56, 150), (186, 350), 28)
    # arms
    _rounded_rect(mask, (24, 168), (66, 340), 20)
    _rounded_rect(mask, (176, 168), (218, 340), 20)
    # neck + head + hair
    cv2.rectangle(mask, (108, 118), (134, 160), 255, -1)
    cv2.ellipse(mask, (121, 84), (40, 48), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (121, 58), (44, 32), 0, 180, 360, 255, -1)

    # clothing texture: bold stripes with color blocks
    cloth = np.zeros_like(bgra[:, :, :3])
    for y in range(h):
        c = (40 + (y * 3) % 90, 90 + (y * 5) % 120, 160 + (y * 2) % 80)
        cloth[y, :] = c
    stripes = (np.sin(np.arange(h) / 5.5) > 0).astype(np.uint8) * 110
    cloth = np.clip(cloth.astype(np.int16) + stripes[:, None, None], 0, 255).astype(np.uint8)
    for _ in range(30):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.rectangle(cloth, (x, y), (x + int(rng.integers(6, 30)), y + int(rng.integers(6, 30))),
                      tuple(int(c) for c in rng.integers(0, 255, 3)), -1)

    bgra[:, :, :3] = cloth
    # skin for head/neck
    head = np.zeros((h, w), np.uint8)
    cv2.rectangle(head, (108, 118), (134, 160), 255, -1)
    cv2.ellipse(head, (121, 84), (40, 48), 0, 0, 360, 255, -1)
    skin = np.full_like(bgra[:, :, :3], (150, 185, 220))
    skin_noise = rng.normal(0, 6, (h, w, 3))
    skin = np.clip(skin.astype(np.float32) + skin_noise, 0, 255).astype(np.uint8)
    bgra[:, :, :3][head > 0] = skin[head > 0]
    # eyes, mouth, hair
    cv2.ellipse(bgra, (106, 80), (5, 4), 0, 0, 360, (40, 40, 40, 255), -1)
    cv2.ellipse(bgra, (136, 80), (5, 4), 0, 0, 360, (40, 40, 40, 255), -1)
    cv2.ellipse(bgra, (121, 104), (14, 6), 0, 0, 180, (60, 70, 160, 255), 2)
    cv2.ellipse(bgra, (121, 52), (46, 30), 0, 180, 360, (35, 30, 30, 255), -1)
    # shoes
    cv2.ellipse(bgra, (98, 526), (26, 11), 0, 0, 360, (20, 20, 20, 255), -1)
    cv2.ellipse(bgra, (150, 526), (26, 11), 0, 0, 360, (20, 20, 20, 255), -1)

    bgra[:, :, 3] = mask
    bgra[mask == 0, :3] = 0
    return bgra


def make_hand_sprite(seed: int = 22) -> np.ndarray:
    rng = _rng(seed)
    w, h = 260, 300
    bgra = np.zeros((h, w, 4), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (128, 205), (78, 88), 8, 0, 360, 255, -1)
    for i in range(4):
        x = 62 + i * 44
        _rounded_rect(mask, (x, 46 + (i % 2) * 12), (x + 30, 160), 14)
    _rounded_rect(mask, (18, 190), (68, 292), 22)  # thumb
    cv2.rectangle(mask, (60, 230), (200, 300), 255, -1)
    skin = np.array([160, 190, 218])
    noise = rng.normal(0, 5, (h, w, 3))
    base = np.clip(skin[None, None, :].astype(np.float32) + noise, 0, 255).astype(np.uint8)
    sleeve = np.zeros_like(base)
    for y in range(h):
        sleeve[y, :] = (60 + (y * 7) % 120, 60 + (y * 3) % 80, 190)
    stripe = (np.sin(np.arange(w) / 8.0) > 0).astype(np.uint8) * 60
    sleeve = np.clip(sleeve.astype(np.int16) + stripe[None, :, None], 0, 255).astype(np.uint8)
    base[220:300, :] = sleeve[220:300, :]
    for _ in range(24):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, 220))
        cv2.circle(base, (x, y), int(rng.integers(1, 4)), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    bgra[:, :, :3] = base
    bgra[:, :, 3] = mask
    bgra[mask == 0, :3] = 0
    return bgra


def make_wall_texture(seed: int = 31) -> np.ndarray:
    rng = _rng(seed)
    w, h = 1200, 700
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (96, 104, 112)
    # wall panels and posters for parallax
    for x in range(0, w, 220):
        cv2.rectangle(img, (x, 0), (x + 4, h), (70, 76, 84), -1)
    for _ in range(14):
        x, y = int(rng.integers(0, w - 180)), int(rng.integers(40, h - 220))
        ww, hh = int(rng.integers(90, 240)), int(rng.integers(80, 260))
        color = tuple(int(c) for c in rng.integers(60, 200, 3))
        cv2.rectangle(img, (x, y), (x + ww, y + hh), color, -1)
        cv2.rectangle(img, (x, y), (x + ww, y + hh), (235, 235, 235), 3)
        for k in range(4):
            yy = y + 18 + k * 22
            if yy < y + hh - 8:
                cv2.line(img, (x + 10, yy), (x + ww - 10, yy), (30, 30, 30), 3)
    for _ in range(30):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(img, (x, y), int(rng.integers(2, 8)), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    noise = rng.normal(0, 5, (h, w, 3))
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def look_at_rotation(c: np.ndarray, target: np.ndarray, roll_deg: float) -> np.ndarray:
    f = target - c
    f /= np.linalg.norm(f)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    down = np.cross(f, right)  # camera y axis (image y points down)
    base = np.stack([right, down, f], axis=0)
    a = math.radians(roll_deg)
    rot = np.array([[math.cos(a), -math.sin(a), 0.0],
                    [math.sin(a), math.cos(a), 0.0],
                    [0.0, 0.0, 1.0]])
    return rot @ base


@dataclass
class ClipSpec:
    name: str
    texture_kind: str
    n_frames: int
    occluders: str  # "none" | "person" | "hand" | "full"
    narrow: bool = False
    cut_frame: int | None = None
    move_after_cut: bool = False
    hand_half: bool = False


def camera_at(t: float, motion: str = "orbit") -> tuple[np.ndarray, np.ndarray]:
    """Return camera centre C and rotation Rcw for parameter t in [0,1]."""
    k = 1.6
    theta = math.radians(-17.0 + 34.0 * t)
    radius = 2.45 - 0.62 * t + 0.07 * math.sin(2 * math.pi * k * t)
    c = np.array([radius * math.sin(theta),
                  0.03 * math.sin(2 * math.pi * 0.9 * t) + 0.02 * t,
                  radius * math.cos(theta)])
    target = np.array([0.04 * math.sin(2 * math.pi * t), 0.02 * math.cos(2 * math.pi * 1.3 * t), 0.0])
    roll = 6.5 * math.sin(2 * math.pi * 1.15 * t + 0.4)
    return c, look_at_rotation(c, target, roll)


def narrow_camera_at(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Orbit toward a steep off-normal angle (mostly elevation/tilt) so the
    projected screen collapses vertically into a thin band. Mid-clip the panel
    is near edge-on, which is exactly the geometry that turns surviving
    features into a near-collinear set."""
    elev = math.radians(-6.0 + 62.0 * t)
    radius = 2.45 + 0.05 * math.sin(2 * math.pi * t)
    # moving in +y and +z tilts the view so the panel's vertical axis foreshortens
    c = np.array([0.12 * math.sin(2 * math.pi * t), radius * math.sin(elev),
                  radius * math.cos(elev)])
    target = np.array([0.0, 0.0, 0.0])
    roll = 2.0 * math.sin(2 * math.pi * t)
    return c, look_at_rotation(c, target, roll)


def moved_screen_homography(K: np.ndarray, R: np.ndarray, C: np.ndarray,
                            offset: np.ndarray) -> np.ndarray:
    """Screen homography after the physical panel is translated in its plane.

    Used after a source cut where the display is also repositioned: the old
    keyframe geometry no longer applies and tracking has to start a new leg.
    """
    t = -R @ C
    A = np.column_stack([R[:, 0], R[:, 1], t + R @ offset])
    H = K @ A @ _screen_texture_to_world()
    return H / H[2, 2]


def project_points(points_world: np.ndarray, K: np.ndarray, R: np.ndarray, C: np.ndarray) -> np.ndarray:
    pc = (R @ (points_world - C).T).T
    uv = (K @ pc.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv


def screen_corners_world() -> np.ndarray:
    hw, hh = SCREEN_W_M / 2, SCREEN_H_M / 2
    return np.array([[-hw, hh, 0], [hw, hh, 0], [hw, -hh, 0], [-hw, -hh, 0]], np.float64)


def _screen_texture_to_world() -> np.ndarray:
    """Maps screen-texture pixel coords to world metres on the z=0 screen plane."""
    sx = SCREEN_W_M / SCREEN_W
    sy = SCREEN_H_M / SCREEN_H
    return np.array([[sx, 0.0, -SCREEN_W_M / 2], [0.0, -sy, SCREEN_H_M / 2], [0.0, 0.0, 1.0]])


def screen_homography(K: np.ndarray, R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Screen-texture pixel -> frame pixel."""
    t = -R @ C
    A = np.column_stack([R[:, 0], R[:, 1], t])
    H = K @ A @ _screen_texture_to_world()
    return H / H[2, 2]


def wall_homography(K: np.ndarray, R: np.ndarray, C: np.ndarray, wall_z: float = -0.95) -> np.ndarray:
    t = R[:, 2] * wall_z - R @ C
    A = np.column_stack([R[:, 0], R[:, 1], t])
    H = K @ A
    return H / H[2, 2]


def billboard_homography(K: np.ndarray, R: np.ndarray, C: np.ndarray,
                         center: np.ndarray, width: float, height: float,
                         sprite_w: int, sprite_h: int) -> np.ndarray:
    right = R[0, :]
    up = -R[1, :]
    corners = np.array([
        center - right * width / 2 + up * height / 2,
        center + right * width / 2 + up * height / 2,
        center + right * width / 2 - up * height / 2,
        center - right * width / 2 - up * height / 2,
    ])
    proj = project_points(corners, K, R, C)
    src = np.array([[0, 0], [sprite_w, 0], [sprite_w, sprite_h], [0, sprite_h]], np.float64)
    return homography_dlt(src, proj)


def warp_rgba(frame: np.ndarray, sprite: np.ndarray, H: np.ndarray) -> None:
    h, w = frame.shape[:2]
    sw, sh = sprite.shape[1], sprite.shape[0]
    warped = cv2.warpPerspective(sprite, H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    src = warped[:, :, :3].astype(np.float32)
    frame[:] = np.clip(frame.astype(np.float32) * (1 - alpha) + src * alpha, 0, 255).astype(np.uint8)


def _hand_half_window(frame_idx: int, n: int) -> tuple[int | None, int | None]:
    """Frame window in which hands cover the screen except a thin band.

    Centred on the steepest part of the narrow-angle orbit so the surviving
    features are projected into a near-collinear band.
    """
    lo, hi = int(0.62 * n), int(0.86 * n)
    if lo <= frame_idx <= hi:
        return lo, hi
    return None, None


def make_cut_texture(seed: int = 71) -> np.ndarray:
    """Visually distinct display material used after the source cut."""
    rng = _rng(seed)
    img = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
    img[:] = (24, 18, 12)
    cv2.rectangle(img, (0, 0), (SCREEN_W, 52), (150, 40, 30), -1)
    cv2.putText(img, "BREAKING DESK", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    for k in range(6):
        y = 80 + k * 42
        cv2.rectangle(img, (24, y), (190, y + 28), (40, 90, 170), -1)
        cv2.line(img, (210, y + 14), (SCREEN_W - 24, y + 14), (210, 210, 210), 2)
    for _ in range(40):
        x, y = int(rng.integers(0, SCREEN_W)), int(rng.integers(60, SCREEN_H))
        cv2.circle(img, (x, y), int(rng.integers(2, 5)),
                   tuple(int(c) for c in rng.integers(120, 255, 3)), -1)
    img = np.clip(img.astype(np.float32) + rng.normal(0, 3.0, img.shape), 0, 255).astype(np.uint8)
    return img


def _occluder_state(spec: ClipSpec, frame_idx: int, n: int) -> dict:
    """Choreography for the occlusion clips. Returns person/hand/full params or None."""
    t = frame_idx / max(1, (n - 1))
    out = {"person": None, "hand": None, "board": None}
    if spec.occluders == "person":
        # cross from right to left, peak coverage around the middle
        if 0.28 <= t <= 0.90:
            u = (t - 0.28) / 0.62
            x = 2.35 - 4.7 * u
            vis = 1.0
            if u > 0.94:
                vis = max(0.0, (1 - u) / 0.06)
            out["person"] = {"center": np.array([x, -0.12 + 0.05 * math.sin(2 * math.pi * 2.2 * t), 0.42]),
                             "height": 1.95, "width": 1.02, "opacity": vis}
        # hand sweeps in, hides the top-left corner, then leaves
        if 0.55 <= t <= 0.80:
            u = (t - 0.55) / 0.25
            out["hand"] = {"u": u, "present": 0.42 <= u <= 0.58}
        # full-board occlusion late in the clip (forces pause, not prediction)
        if spec.occluders == "person" and 0.93 <= t <= 0.995:
            u = (t - 0.93) / 0.065
            enter = min(1.0, u / 0.2, (1 - u) / 0.2)
            out["board"] = {"center": np.array([0.0, 0.0, 0.60]), "width": 2.2, "height": 1.5,
                            "opacity": max(0.0, enter)}
    elif spec.occluders == "full":
        if 0.25 <= t <= 0.80:
            u = (t - 0.25) / 0.55
            enter = min(1.0, u / 0.18, (1 - u) / 0.18)
            out["board"] = {"center": np.array([0.0, 0.0, 0.60]), "width": 2.2, "height": 1.5,
                            "opacity": max(0.0, enter)}
    return out


def _make_board_sprite(seed: int = 41) -> np.ndarray:
    rng = _rng(seed)
    w, h = 480, 340
    bgra = np.zeros((h, w, 4), np.uint8)
    base = np.zeros((h, w, 3), np.uint8)
    base[:] = (70, 78, 86)
    for y in range(0, h, 16):
        cv2.line(base, (0, y), (w, y), (110, 116, 124), 1)
    for _ in range(60):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.rectangle(base, (x, y), (x + int(rng.integers(8, 40)), y + int(rng.integers(8, 40))),
                      tuple(int(c) for c in rng.integers(40, 220, 3)), -1)
    bgra[:, :, :3] = base
    bgra[:, :, 3] = 255
    return bgra


def _rasterize_mask(shape: tuple[int, int], H: np.ndarray, sprite_alpha: np.ndarray) -> np.ndarray:
    warped = cv2.warpPerspective(sprite_alpha, H, (shape[1], shape[0]), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped > 16


def _corner_visibility(person_mask: np.ndarray, board_mask: np.ndarray,
                       quad: np.ndarray, radius: int = 7) -> list[bool]:
    h, w = person_mask.shape
    visible = []
    occ = person_mask | board_mask
    for x, y in quad:
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
        y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
        if x1 <= x0 or y1 <= y0:
            visible.append(False)
            continue
        patch = occ[y0:y1, x0:x1]
        visible.append(bool(patch.mean() < 0.35))
    return visible


def render_clip(outdir: str, spec: ClipSpec, seed: int = 5, encode: bool = True) -> dict:
    os.makedirs(outdir, exist_ok=True)
    rng = _rng(seed)
    K = np.array([[525.0, 0, FRAME_W / 2], [0, 525.0, FRAME_H / 2], [0, 0, 1.0]])
    screen_tex = make_screen_texture(spec.texture_kind, seed=seed)
    wall_tex = make_wall_texture(seed=seed + 1)
    person = make_person_sprite(seed=seed + 2)
    hand = make_hand_sprite(seed=seed + 3)
    board = _make_board_sprite(seed=seed + 4)
    content = make_content_texture(seed=seed + 5)
    cut_texture = make_cut_texture(seed=seed + 6)
    cv2.imwrite(os.path.join(outdir, "screen.png"), screen_tex)
    cv2.imwrite(os.path.join(outdir, "cut_screen.png"), cut_texture)
    cv2.imwrite(os.path.join(outdir, "content.png"), content)
    cv2.imwrite(os.path.join(outdir, "person_sprite.png"), person)

    truth_frames = []
    frames = []
    wx, wy = 4.0, 2.4
    wall_src = np.array([[0, 0], [wall_tex.shape[1], 0], [wall_tex.shape[1], wall_tex.shape[0]],
                         [0, wall_tex.shape[0]]], np.float64)
    wall_world = np.array([[-wx / 2, wy / 2, -0.95], [wx / 2, wy / 2, -0.95],
                           [wx / 2, -wy / 2, -0.95], [-wx / 2, -wy / 2, -0.95]])

    for i in range(spec.n_frames):
        t = i / max(1, (spec.n_frames - 1))
        C, R = narrow_camera_at(t) if spec.narrow else camera_at(t, spec.name)
        # Source cut: swap the displayed material at the cut frame. After the
        # cut the panel can also be repositioned, so the old keyframe geometry
        # and descriptors both stop applying.
        active_tex = screen_tex
        cut_now = spec.cut_frame is not None and i == spec.cut_frame
        post_cut = spec.cut_frame is not None and i >= spec.cut_frame
        if post_cut:
            active_tex = cut_texture
        panel_offset = np.zeros(3)
        if post_cut and spec.move_after_cut:
            u = min(1.0, (i - spec.cut_frame) / max(1, spec.n_frames - spec.cut_frame))
            panel_offset = np.array([0.42 * u, 0.22 * u, 0.0])
        frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        # wall
        wp = project_points(wall_world, K, R, C)
        H_wall = homography_dlt(wall_src, wp)
        wall_warp = cv2.warpPerspective(wall_tex, H_wall, (FRAME_W, FRAME_H))
        wmask = cv2.warpPerspective(np.full(wall_tex.shape[:2], 255, np.uint8), H_wall, (FRAME_W, FRAME_H))
        frame[wmask > 0] = wall_warp[wmask > 0]
        # screen
        if post_cut and spec.move_after_cut:
            H_screen = moved_screen_homography(K, R, C, panel_offset)
        else:
            H_screen = screen_homography(K, R, C)
        screen_warp = cv2.warpPerspective(active_tex, H_screen, (FRAME_W, FRAME_H))
        smask = cv2.warpPerspective(np.full((SCREEN_H, SCREEN_W), 255, np.uint8), H_screen,
                                    (FRAME_W, FRAME_H))
        gain = 0.96 + 0.04 * math.sin(2 * math.pi * 1.7 * t + 0.3)
        screen_warp = np.clip(screen_warp.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        frame[smask > 0] = screen_warp[smask > 0]
        # bevel around the panel
        quad = apply_h(H_screen, np.array([[0, 0], [SCREEN_W, 0], [SCREEN_W, SCREEN_H], [0, SCREEN_H]],
                                          np.float64))
        cv2.polylines(frame, [np.round(quad).astype(np.int32)], True, (22, 24, 28), 3, cv2.LINE_AA)

        # half-screen hand occlusion (degeneracy regression): a hand covers the
        # left half of the panel for a window. In the narrow-angle view the
        # surviving right-side features are compressed into a thin band, so the
        # frame is tracked only while the homography stays well conditioned;
        # when the evidence collapses the frame fails and later relinks.
        if spec.hand_half:
            hh0, hh1 = _hand_half_window(i, spec.n_frames)
            if hh0 is not None:
                hand_tex_mask = np.zeros((SCREEN_H, SCREEN_W), np.uint8)
                hand_tex_mask[:, : int(SCREEN_W * 0.55)] = 255
                Hh = H_screen
                hm = cv2.warpPerspective(hand_tex_mask, Hh, (FRAME_W, FRAME_H)) > 0
                frame[hm] = (78 + int(10 * math.sin(i)), 66, 58)
                hand_mask = hm
            else:
                hand_mask = np.zeros((FRAME_H, FRAME_W), bool)
        else:
            hand_mask = np.zeros((FRAME_H, FRAME_W), bool)

        occ = _occluder_state(spec, i, spec.n_frames)
        person_mask = np.zeros((FRAME_H, FRAME_W), bool)
        board_mask = np.zeros((FRAME_H, FRAME_W), bool)
        occ_quads = {}

        if occ["person"] is not None:
            p = occ["person"]
            Hp = billboard_homography(K, R, C, p["center"], p["width"], p["height"],
                                      person.shape[1], person.shape[0])
            pmask = _rasterize_mask((FRAME_H, FRAME_W), Hp, person[:, :, 3])
            person_mask = pmask
            occ_quads["person"] = {
                "quad": apply_h(Hp, np.array([[0, 0], [person.shape[1], 0],
                                              [person.shape[1], person.shape[0]],
                                              [0, person.shape[0]]], np.float64)).tolist(),
                "opacity": round(float(p["opacity"]), 3),
            }
            # soft shadow on the screen
            shadow = pmask.astype(np.float32)
            shadow = cv2.GaussianBlur(shadow, (0, 0), 7)
            frame = np.clip(frame.astype(np.float32) * (1 - 0.30 * shadow[:, :, None]), 0, 255).astype(np.uint8)
            if p["opacity"] > 0.01:
                warp_rgba(frame, person, Hp)

        if occ["hand"] is not None and not spec.hand_half:
            hd = occ["hand"]
            # keep a solid mark on the screen frame, blending only while moving
            u_hand = hd["u"]
            if not hd["present"]:
                hx, hy, hop = -0.80, 0.34, 0.0
            else:
                hx, hy, hop = -0.80, 0.34, 1.0
            Hh = billboard_homography(K, R, C, np.array([hx, hy, 0.52]), 1.05, 1.05,
                                      hand.shape[1], hand.shape[0])
            hand_mask = _rasterize_mask((FRAME_H, FRAME_W), Hh, hand[:, :, 3]) if hop > 0 else \
                np.zeros((FRAME_H, FRAME_W), bool)
            occ_quads["hand"] = {
                "quad": apply_h(Hh, np.array([[0, 0], [hand.shape[1], 0],
                                              [hand.shape[1], hand.shape[0]],
                                              [0, hand.shape[0]]], np.float64)).tolist(),
                "opacity": round(float(hop), 3),
            }
            if hop > 0.01:
                warp_rgba(frame, hand, Hh)

        if occ["board"] is not None:
            bd = occ["board"]
            Hb = billboard_homography(K, R, C, bd["center"], bd["width"], bd["height"],
                                      board.shape[1], board.shape[0])
            bmask = _rasterize_mask((FRAME_H, FRAME_W), Hb, board[:, :, 3])
            board_mask = bmask
            occ_quads["board"] = {
                "quad": apply_h(Hb, np.array([[0, 0], [board.shape[1], 0],
                                              [board.shape[1], board.shape[0]],
                                              [0, board.shape[0]]], np.float64)).tolist(),
                "opacity": round(float(bd["opacity"]), 3),
            }
            if bd["opacity"] > 0.5:
                warp_rgba(frame, board, Hb)
            else:
                # fade in
                alpha = np.clip(bd["opacity"], 0, 1)
                warped = cv2.warpPerspective(board, Hb, (FRAME_W, FRAME_H))
                a = (warped[:, :, 3:4].astype(np.float32) / 255.0) * alpha
                frame = np.clip(frame * (1 - a) + warped[:, :, :3].astype(np.float32) * a, 0, 255).astype(np.uint8)
                board_mask = bmask & (alpha > 0.5)

        # sensor noise + vignette
        noise = rng.normal(0, 1.6, frame.shape)
        yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
        r2 = ((xx - FRAME_W / 2) ** 2 + (yy - FRAME_H / 2) ** 2) / (FRAME_W * FRAME_H / 4)
        vig = (1.0 - 0.18 * r2)[:, :, None]
        frame = np.clip((frame.astype(np.float32) + noise) * vig, 0, 255).astype(np.uint8)

        quad_mask = np.zeros((FRAME_H, FRAME_W), np.uint8)
        cv2.fillConvexPoly(quad_mask, np.round(quad).astype(np.int32), 255)
        occ_union = (person_mask | hand_mask | board_mask) & (quad_mask > 0)
        occ_ratio = float(occ_union.sum()) / max(1, int((quad_mask > 0).sum()))
        if occ["person"] is not None and occ["person"]["opacity"] < 0.5:
            occ_ratio *= float(occ["person"]["opacity"])
        corner_vis = _corner_visibility(person_mask | hand_mask, board_mask, quad)

        frames.append(frame)
        truth_frames.append({
            "frame": i,
            "quad": quad.tolist(),
            "camera_C": C.tolist(),
            "camera_R": R.tolist(),
            "occlusion": round(float(min(1.0, occ_ratio)), 4),
            "corner_visible": corner_vis,
            "occ_person": bool(person_mask.any()),
            "occ_hand": bool(hand_mask.any()),
            "occ_board": bool(board_mask.any()),
            "occluders": occ_quads,
            "source_cut": bool(cut_now),
            "post_cut": bool(post_cut),
            "moved_panel": bool(post_cut and spec.move_after_cut),
        })

    truth = {
        "name": spec.name,
        "fps": FPS,
        "frame_size": [FRAME_W, FRAME_H],
        "texture_size": [SCREEN_W, SCREEN_H],
        "K": K.tolist(),
        "frames": truth_frames,
        "content": "content.png",
        "screen": "screen.png",
        "cut_frame": spec.cut_frame,
        "narrow": bool(spec.narrow),
        "cut_screen": "cut_screen.png" if spec.cut_frame is not None else None,
    }

    if encode:
        mp4 = os.path.join(outdir, "source.mp4")
        encode_frames_ffmpeg(frames, mp4, FPS)
    truth["source"] = "source.mp4"
    with open(os.path.join(outdir, "truth.json"), "w") as f:
        json.dump(truth, f)
    return truth


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode_frames_ffmpeg(frames: list[np.ndarray], path: str, fps: int, crf: int = 17) -> None:
    if not frames:
        raise ValueError("no frames to encode")
    h, w = frames[0].shape[:2]
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for fr in frames:
            proc.stdin.write(fr.astype(np.uint8).tobytes())
        proc.stdin.close()
    finally:
        err = proc.stderr.read().decode() if proc.stderr else ""
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg encode failed: {err[:800]}")


def read_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


CLIPS = {
    "orbit_clean": ClipSpec("orbit_clean", "standard", 150, "none"),
    "occlusion": ClipSpec("occlusion", "standard", 170, "person"),
    "full_board": ClipSpec("full_board", "standard", 140, "full"),
    "corner_texture": ClipSpec("corner_texture", "corner", 140, "none"),
    # narrow-angle view with a half-screen hand: surviving features approach a
    # collinear band and the degenerate-homography gate must hold
    "narrow_hand": ClipSpec("narrow_hand", "standard", 120, "none",
                            narrow=True, hand_half=True),
    # source cut mid-occlusion: new material and a repositioned panel, so the
    # old keyframe cannot be reused and tracking must start a fresh leg
    "cut_move": ClipSpec("cut_move", "standard", 140, "none",
                         narrow=True, cut_frame=70, move_after_cut=True),
}


def generate_all(root: str, only: list[str] | None = None) -> dict:
    out = {}
    for name, spec in CLIPS.items():
        if only and name not in only:
            continue
        d = os.path.join(root, name)
        truth = render_clip(d, spec)
        out[name] = d
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/clips")
    ap.add_argument("--only", nargs="*")
    ns = ap.parse_args()
    print(generate_all(ns.out, ns.only))
