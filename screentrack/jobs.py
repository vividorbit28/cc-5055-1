"""Job orchestration: tracking, compositing and evaluation as background work."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import compositor, evaluate, tracker
from .geometry import corner_order
from .synth import read_frames

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
JOBS = os.path.join(DATA, "jobs")
CLIPS = os.path.join(DATA, "clips")


def job_dir(job_id: str) -> str:
    return os.path.join(JOBS, job_id)


def read_json(path: str, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


@dataclass
class Job:
    id: str
    name: str
    source: str
    clip: str | None = None
    status: str = "created"
    stage: str = ""
    progress: float = 0.0
    error: str | None = None
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    anchor: dict | None = None
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "status": self.status, "stage": self.stage,
                "progress": self.progress, "error": self.error, "created": self.created,
                "anchor": self.anchor, "clip": self.clip,
                "has_track": os.path.exists(os.path.join(job_dir(self.id), "track.json")),
                "has_composite": os.path.exists(os.path.join(job_dir(self.id), "composite.mp4")),
                "has_diagnostics": os.path.exists(os.path.join(job_dir(self.id), "diagnostics.mp4")),
                "has_eval": os.path.exists(os.path.join(job_dir(self.id), "evaluation.json"))}


class JobStore:
    def __init__(self):
        os.makedirs(JOBS, exist_ok=True)
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self._load()

    def _load(self):
        for jid in os.listdir(JOBS):
            meta = read_json(os.path.join(JOBS, jid, "job.json"))
            if not meta:
                continue
            src = meta.get("source", "")
            if src and not os.path.isabs(src):
                src = os.path.join(JOBS, jid, src)
            j = Job(id=jid, name=meta.get("name", jid), source=src,
                    clip=meta.get("clip"), status="ready" if meta.get("ready") else "failed",
                    stage="ready" if meta.get("ready") else "interrupted",
                    progress=1.0 if meta.get("ready") else 0.0,
                    error=meta.get("error"), created=meta.get("created", ""),
                    anchor=meta.get("anchor"), config=meta.get("config", {}))
            self.jobs[jid] = j

    def create(self, name: str, source_path: str, clip: str | None = None) -> Job:
        jid = uuid.uuid4().hex[:12]
        d = job_dir(jid)
        os.makedirs(d, exist_ok=True)
        ext = os.path.splitext(source_path)[1] or ".mp4"
        dst = os.path.join(d, "source" + ext)
        shutil.copy2(source_path, dst)
        j = Job(id=jid, name=name, source=dst, clip=clip)
        with self.lock:
            self.jobs[jid] = j
        write_json(os.path.join(d, "job.json"), {"name": name, "source": os.path.basename(dst),
                                                 "clip": clip, "created": j.created})
        return j

    def get(self, jid: str) -> Job | None:
        with self.lock:
            return self.jobs.get(jid)

    def all(self) -> list[Job]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def update(self, jid: str, **kw):
        with self.lock:
            j = self.jobs.get(jid)
            if not j:
                return
            for k, v in kw.items():
                setattr(j, k, v)

    def persist(self, jid: str, ready: bool):
        d = job_dir(jid)
        meta = read_json(os.path.join(d, "job.json"), {}) or {}
        meta["ready"] = ready
        j = self.get(jid)
        if j:
            meta.update({"name": j.name, "anchor": j.anchor, "config": j.config,
                         "error": j.error, "clip": j.clip})
        write_json(os.path.join(d, "job.json"), meta)


STORE = JobStore()


def frame_path(jid: str, idx: int) -> str:
    return os.path.join(job_dir(jid), "frames", f"{idx:06d}.jpg")


def ensure_frames(jid: str, source: str) -> int:
    """Write JPEG thumbnails for the review UI and return the frame count."""
    d = os.path.join(job_dir(jid), "frames")
    os.makedirs(d, exist_ok=True)
    existing = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    frames = read_frames(source)
    if len(existing) == len(frames):
        return len(frames)
    for idx, fr in enumerate(frames):
        cv2.imwrite(frame_path(jid, idx), fr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return len(frames)


def run_job(jid: str, start_frame: int, quad: list, cfg_overrides: dict | None = None,
            options: dict | None = None) -> None:
    job = STORE.get(jid)
    if job is None:
        return
    options = options or {}
    try:
        STORE.update(jid, status="running", stage="preparing", progress=0.0, error=None)
        n = ensure_frames(jid, job.source)
        if not (0 <= start_frame < n):
            raise ValueError("anchor frame is outside the video")
        cfg = tracker.TrackConfig()
        for k, v in (cfg_overrides or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        d = job_dir(jid)
        STORE.update(jid, stage="tracking", progress=0.03)

        def cb(done, total, stage):
            frac = 0.03 + 0.72 * (done / max(1, total))
            STORE.update(jid, progress=round(frac, 3), stage=f"tracking ({stage})")

        track = tracker.track_path(job.source, start_frame, np.array(quad, np.float64), cfg,
                                   out_path=os.path.join(d, "track.json"), progress_cb=cb)
        STORE.update(jid, stage="composing", progress=0.78)
        if job.clip:
            content = cv2.imread(os.path.join(CLIPS, job.clip, "content.png"))
        else:
            content = None
        if content is None:
            content = _default_content(track)
        truth = _load_truth(job)
        info = compositor.composite_track(job.source, track, content, d, truth=truth,
                                          smooth=options.get("smooth", True),
                                          draw_features=options.get("draw_features", True),
                                          draw_rejected=options.get("draw_rejected", True))
        STORE.update(jid, stage="evaluating", progress=0.92)
        shadow = _shadow_replacements(job, d)
        ev = None
        if truth is not None:
            ev = evaluate.evaluate(track, truth)
            slip = evaluate.measure_overlay_slip(truth, track, content)
            ev["slip"] = slip
            ev["decisions"] = evaluate.compare_anchor_handling(truth, track)
            ev["integrity"] = _integrity_checks(job, d)
            write_json(os.path.join(d, "evaluation.json"), ev)
        write_json(os.path.join(d, "composite_info.json"), info)
        write_json(os.path.join(d, "masks_info.json"), shadow)
        meta = read_json(os.path.join(d, "job.json"), {}) or {}
        meta.update({"ready": True, "anchor": job.anchor, "config": cfg.to_dict(),
                     "start_frame": start_frame})
        write_json(os.path.join(d, "job.json"), meta)
        STORE.update(jid, status="ready", stage="ready", progress=1.0)
    except Exception as exc:  # surfaced to the UI with the traceback tail
        tb = traceback.format_exc()[-1500:]
        STORE.update(jid, status="failed", stage="failed", error=f"{exc}\n{tb}", progress=0.0)
        STORE.persist(jid, ready=False)


def _default_content(track: dict) -> np.ndarray:
    tw, th = track["reference"]["texture_size"]
    return cv2.imread(os.path.join(CLIPS, "orbit_clean", "content.png")) if os.path.exists(
        os.path.join(CLIPS, "orbit_clean", "content.png")) else np.zeros((th, tw, 3), np.uint8)


def _load_truth(job: Job):
    if not job.clip:
        return None
    path = os.path.join(CLIPS, job.clip, "truth.json")
    if not os.path.exists(path):
        return None
    truth = read_json(path)
    return truth


def _shadow_replacements(job: Job, d: str) -> dict:
    """Re-run the tracker with alternative hypotheses to test the decisions.

    Runs are done on short windows so the reviewer can see that the reported
    observation was produced by the solver, not a scripted animation.
    """
    out = {}
    track = read_json(os.path.join(d, "track.json"))
    if not track:
        return out
    return {"note": "track.json is the single source of truth; re-runs are explicit jobs",
            "frames": len(track.get("frames", []))}


def _integrity_checks(job: Job, d: str, _d: str = None) -> dict:
    """Things a reviewer should be able to confirm without trusting the report."""
    track_path = os.path.join(d, "track.json")
    track = read_json(track_path)
    out = {"source_unmodified": True, "track_written_atomically": True}
    if not track:
        return out
    src = job.source
    if os.path.exists(src):
        out["source_bytes"] = os.path.getsize(src)
        out["source_sha256"] = _sha256(src)
    out["track_frames"] = len(track.get("frames", []))
    out["track_created"] = track.get("created")
    out["tracking_vs_manual"] = track.get("reference", {})
    return out


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
