"""
خادم FastAPI لتبديل الوجوه في الفيديو
التشغيل:  uvicorn main:app --host 0.0.0.0 --port 8000
"""
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.engine import FaceSwapEngine, SwapOptions

BASE = Path(__file__).resolve().parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE / "models"))
WORK_DIR = Path(os.getenv("WORK_DIR", BASE / "workdir"))
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", BASE.parent / "frontend"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI Video Face Swap")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

JOBS: dict[str, dict] = {}
JOB_QUEUE: "queue.Queue[str]" = queue.Queue()
_engine: FaceSwapEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> FaceSwapEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = FaceSwapEngine(MODELS_DIR)
        return _engine


# ------------------------------------------------------------------ API
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "engine_loaded": _engine is not None,
        "gpu": bool(_engine and _engine.use_gpu),
        "queue": JOB_QUEUE.qsize(),
    }


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    face: UploadFile = File(...),
    enhance: bool = Form(True),
    enhance_strength: float = Form(0.8),
    blend: float = Form(1.0),
    color_match: float = Form(0.5),
    all_faces: bool = Form(False),
    watermark: bool = Form(True),
    consent: bool = Form(False),
):
    if not consent:
        raise HTTPException(400, "يجب تأكيد امتلاك موافقة صاحب الوجه قبل المعالجة")
    if not (video.content_type or "").startswith("video/"):
        raise HTTPException(400, "ملف الفيديو غير صالح")
    if (face.content_type or "") not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(400, "صورة الوجه يجب أن تكون JPG أو PNG")

    job_id = uuid.uuid4().hex[:12]
    jdir = WORK_DIR / job_id
    jdir.mkdir(parents=True)
    vpath = jdir / ("input" + (Path(video.filename or "v.mp4").suffix or ".mp4"))
    fpath = jdir / ("face" + (Path(face.filename or "f.jpg").suffix or ".jpg"))

    for up, dst in ((video, vpath), (face, fpath)):
        size = 0
        with open(dst, "wb") as f:
            while chunk := await up.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1024 * 1024:
                    shutil.rmtree(jdir, ignore_errors=True)
                    raise HTTPException(413, f"الحد الأقصى للملف {MAX_UPLOAD_MB}MB")
                f.write(chunk)

    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0.0,
        "frames_done": 0,
        "frames_total": 0,
        "fps_proc": 0.0,
        "eta_sec": None,
        "message": "في قائمة الانتظار",
        "dir": str(jdir),
        "video": str(vpath),
        "face": str(fpath),
        "options": SwapOptions(
            enhance=enhance,
            enhance_strength=enhance_strength,
            blend=blend,
            color_match=color_match,
            all_faces=all_faces,
            watermark=watermark,
        ),
        "created": time.time(),
    }
    JOB_QUEUE.put(job_id)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "المهمة غير موجودة")
    return {k: v for k, v in j.items() if k not in ("dir", "video", "face", "options")}


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str, download: bool = False):
    j = JOBS.get(job_id)
    if not j or j["status"] != "done":
        raise HTTPException(404, "النتيجة غير جاهزة")
    return FileResponse(
        Path(j["dir"]) / "result.mp4",
        media_type="video/mp4",
        filename=f"faceswap_{job_id}.mp4" if download else None,
    )


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    j = JOBS.pop(job_id, None)
    if j:
        shutil.rmtree(j["dir"], ignore_errors=True)
    return {"deleted": bool(j)}


# ------------------------------------------------------------------ worker
def _update(j, **kw):
    j.update(kw)


def process_job(job_id: str):
    j = JOBS[job_id]
    jdir = Path(j["dir"])
    _update(j, status="loading", message="تحميل النماذج...")
    engine = get_engine()

    face_img = cv2.imread(j["face"])
    if face_img is None:
        raise ValueError("تعذّر قراءة صورة الوجه")
    source_face = engine.get_source_face(face_img)

    cap = cv2.VideoCapture(j["video"])
    if not cap.isOpened():
        raise ValueError("تعذّر فتح الفيديو")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    raw = jdir / "raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    _update(j, status="processing", frames_total=total, message="معالجة الإطارات...")

    t0, done = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = engine.process_frame(frame, source_face, j["options"])
        writer.write(out)
        done += 1
        if done % 2 == 0 or done == total:
            el = time.time() - t0
            rate = done / el if el > 0 else 0
            _update(
                j,
                frames_done=done,
                progress=round(done / total * 95, 2) if total else 0,
                fps_proc=round(rate, 2),
                eta_sec=int((total - done) / rate) if rate and total else None,
            )
    cap.release()
    writer.release()

    _update(j, status="encoding", progress=96, message="ترميز الفيديو ودمج الصوت...")
    result = jdir / "result.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(raw), "-i", j["video"],
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
        str(result),
    ]
    try:
        subprocess.run(cmd, check=True)
        raw.unlink(missing_ok=True)
    except Exception as e:
        print("[ffmpeg] fallback to raw:", e)
        shutil.move(raw, result)

    _update(j, status="done", progress=100, eta_sec=0, message="اكتملت المعالجة ✓")


def worker_loop():
    while True:
        job_id = JOB_QUEUE.get()
        j = JOBS.get(job_id)
        if not j:
            continue
        try:
            process_job(job_id)
        except Exception as e:
            import traceback
            traceback.print_exc()
            _update(j, status="error", message=f"خطأ: {e}")
        finally:
            JOB_QUEUE.task_done()


@app.on_event("startup")
def _startup():
    threading.Thread(target=worker_loop, daemon=True).start()
    # تحميل مسبق للنماذج في الخلفية لتسريع أول مهمة
    threading.Thread(target=lambda: _safe_preload(), daemon=True).start()


def _safe_preload():
    try:
        get_engine()
        print("[engine] models loaded ✓  GPU:", _engine.use_gpu)
    except Exception as e:
        print("[engine] preload failed:", e)


# ------------------------------------------------------------------ frontend
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
