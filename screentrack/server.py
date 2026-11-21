"""FastAPI service for the screen replacement tracker."""
from __future__ import annotations

import io
import json
import os
import shutil
import threading

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import jobs, synth
from .geometry import corner_order

app = FastAPI(title="Screen Replacement Tracker")
STATIC = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(jobs.CLIPS, exist_ok=True)
os.makedirs(jobs.JOBS, exist_ok=True)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/clips")
def list_clips():
    out = []
    if os.path.isdir(jobs.CLIPS):
        for name in sorted(os.listdir(jobs.CLIPS)):
            d = os.path.join(jobs.CLIPS, name)
            truth = jobs.read_json(os.path.join(d, "truth.json"))
            if truth is None:
                continue
            out.append({
                "name": name,
                "frames": len(truth["frames"]),
                "fps": truth["fps"],
                "frame_size": truth["frame_size"],
                "description": {
                    "orbit_clean": "camera orbit with scale, rotation and perspective change (no occluders)",
                    "occlusion": "person crosses the screen (>50% coverage) and a hand hides one corner",
                    "full_board": "screen fully covered for 60+ frames, then revealed",
                    "corner_texture": "texture concentrated in one corner; far corners unsupported",
                }.get(name, ""),
            })
    return {"clips": out}


@app.get("/api/clips/{name}/frame/{idx}")
def clip_frame(name: str, idx: int):
    d = os.path.join(jobs.CLIPS, name)
    src = os.path.join(d, "source.mp4")
    if not os.path.exists(src):
        raise HTTPException(404, "clip not found")
    truth = jobs.read_json(os.path.join(d, "truth.json"))
    total = len(truth["frames"]) if truth else 0
    if not (0 <= idx < total):
        raise HTTPException(404, "frame out of range")
    return _frame_response(src, idx)


@app.get("/api/clips/{name}/truth")
def clip_truth(name: str):
    d = os.path.join(jobs.CLIPS, name)
    truth = jobs.read_json(os.path.join(d, "truth.json"))
    if truth is None:
        raise HTTPException(404, "truth not found")
    return truth


def _frame_response(src: str, idx: int):
    fr = _read_single_frame(src, idx)
    if fr is None:
        raise HTTPException(404, "frame out of range")
    ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return JSONResponse({"frame": idx, "jpeg_b64": _b64(buf.tobytes())})


def _read_single_frame(src: str, idx: int):
    """Decode frame idx. The OpenCV build has no H.264 decoder in this image,
    so fall back to ffmpeg via the same reader used for the benchmark clips."""
    from .synth import read_frames
    import tempfile
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if ok:
        return fr
    frames = read_frames(src)
    if 0 <= idx < len(frames):
        return frames[idx]
    return None


def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode()


@app.post("/api/jobs")
async def create_job(name: str = Form(...), clip: str | None = Form(None),
                     file: UploadFile | None = File(None)):
    if clip:
        src = os.path.join(jobs.CLIPS, clip, "source.mp4")
        if not os.path.exists(src):
            raise HTTPException(400, "unknown clip")
        job = jobs.STORE.create(name or clip, src, clip=clip)
    elif file is not None:
        tmp = os.path.join(jobs.JOBS, f"upload_{hash(name)}.tmp")
        with open(tmp, "wb") as f:
            shutil.copyfileobj(file.file, f)
        job = jobs.STORE.create(name or "upload", tmp, clip=None)
        os.remove(tmp)
    else:
        raise HTTPException(400, "provide a clip name or an uploaded file")
    jobs.ensure_frames(job.id, job.source)
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": [j.to_dict() for j in jobs.STORE.all()]}


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    j = jobs.STORE.get(jid)
    if not j:
        raise HTTPException(404, "job not found")
    d = jobs.job_dir(jid)
    ev = jobs.read_json(os.path.join(d, "evaluation.json"))
    info = jobs.read_json(os.path.join(d, "composite_info.json"))
    data = j.to_dict()
    data["evaluation"] = ev
    data["composite_info"] = info
    return data


@app.get("/api/jobs/{jid}/frame/{idx}")
def job_frame(jid: str, idx: int):
    j = jobs.STORE.get(jid)
    if not j:
        raise HTTPException(404, "job not found")
    return _frame_response(j.source, idx)


@app.post("/api/jobs/{jid}/track")
async def start_track(jid: str, request: Request):
    j = jobs.STORE.get(jid)
    if not j:
        raise HTTPException(404, "job not found")
    body = await request.json()
    start = int(body.get("start_frame", 0))
    quad = body.get("quad")
    if not quad or len(quad) != 4:
        raise HTTPException(400, "quad must have four points")
    quad = corner_order(np.array(quad, np.float64))
    j.anchor = {"frame": start, "quad": quad.tolist(),
                "corners_frame": body.get("corners_frame")}
    jobs.STORE.persist(jid, ready=False)
    cfg = body.get("config") or {}
    opts = body.get("options") or {}
    th = threading.Thread(target=jobs.run_job, args=(jid, start, quad.tolist(), cfg, opts),
                          daemon=True)
    th.start()
    return {"started": True, "job": j.to_dict()}


@app.get("/api/jobs/{jid}/track")
def get_track(jid: str):
    p = os.path.join(jobs.job_dir(jid), "track.json")
    if not os.path.exists(p):
        raise HTTPException(404, "track not available")
    return FileResponse(p, media_type="application/json")


@app.get("/api/jobs/{jid}/evaluation")
def get_evaluation(jid: str):
    p = os.path.join(jobs.job_dir(jid), "evaluation.json")
    if not os.path.exists(p):
        raise HTTPException(404, "evaluation not available")
    return FileResponse(p, media_type="application/json")


@app.get("/api/jobs/{jid}/video/{kind}")
def get_video(jid: str, kind: str):
    names = {"composite": "composite.mp4", "diagnostics": "diagnostics.mp4",
             "source": None}
    if kind not in names:
        raise HTTPException(404, "unknown video")
    if kind == "source":
        j = jobs.STORE.get(jid)
        if not j:
            raise HTTPException(404, "job not found")
        return FileResponse(j.source, media_type="video/mp4")
    p = os.path.join(jobs.job_dir(jid), names[kind])
    if not os.path.exists(p):
        raise HTTPException(404, "video not available")
    return FileResponse(p, media_type="video/mp4")


@app.get("/api/jobs/{jid}/truth")
def get_truth(jid: str):
    j = jobs.STORE.get(jid)
    if not j or not j.clip:
        return {"truth": None}
    truth = jobs.read_json(os.path.join(jobs.CLIPS, j.clip, "truth.json"))
    return {"truth": truth}


@app.get("/api/clips/{name}/asset/{kind}")
def clip_asset(name: str, kind: str):
    if kind not in ("content", "screen", "source"):
        raise HTTPException(404, "unknown asset")
    d = os.path.join(jobs.CLIPS, name)
    fname = {"content": "content.png", "screen": "screen.png", "source": "source.mp4"}[kind]
    p = os.path.join(d, fname)
    if not os.path.exists(p):
        raise HTTPException(404, "missing asset")
    media = "image/png" if fname.endswith(".png") else "video/mp4"
    return FileResponse(p, media_type=media)


@app.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    only = body.get("only")
    def work():
        synth.generate_all(jobs.CLIPS, only)
    threading.Thread(target=work, daemon=True).start()
    return {"started": True}


app.mount("/static", StaticFiles(directory=STATIC), name="static")
