import asyncio
import copy
import json
import logging
import os
import shutil
import time
import urllib.parse
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------------- Config (env-first) ----------------
API_KEY = os.getenv("VYBERS_API_KEY", "12345")

COMFYUI_API_URL = os.getenv("COMFYUI_API_URL", "http://0.0.0.0:8188")
WORKFLOW_FILE = os.getenv(
    "WORKFLOW_FILE",
    "/workspace/fdy/comfyui_vybers-ai/wan2.1-t2v-1_3B_api.json",
)

POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "2"))
MAX_WAIT_TIME = int(os.getenv("MAX_WAIT_TIME", "300"))

STATIC_DIR = os.getenv("STATIC_DIR", "videos")  # served output dir
STATIC_DIR_PATH = Path(STATIC_DIR)
STATIC_DIR_PATH.mkdir(parents=True, exist_ok=True)

# Optional: if you know your public base URL, set it (e.g. https://gpu-api.vybers.ai)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

# TTL cleanup (optional)
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", str(24 * 3600)))  # 24h default
ENABLE_CLEANUP = os.getenv("ENABLE_CLEANUP", "1") == "1"

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("vybers_comfy_api")

# ---------------- Load workflow ----------------
try:
    with open(WORKFLOW_FILE, "r", encoding="utf-8") as f:
        base_workflow = json.load(f)
    logger.info(f"Loaded workflow: {WORKFLOW_FILE}")
except Exception as e:
    logger.error(f"Failed to load workflow {WORKFLOW_FILE}: {e}")
    raise

# ---------------- Request models ----------------
class GenerateVideoRequest(BaseModel):
    positive: str
    negative: str = ""
    height: int = 512
    width: int = 512
    length: int = 81  # num frames
    fps: int = 16

# ---------------- App ----------------
app = FastAPI(title="Vybers ComfyUI Video Generation API")

# If your webapp calls GPU API directly from browser, you need CORS.
# If you proxy through Next.js, you can disable/lock this down.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- In-memory job store ----------------
# For single instance this is fine. If you need persistence/restarts, swap for sqlite/redis.
Jobs = Dict[str, dict]
jobs: Jobs = {}  # job_id == prompt_id

def _auth(x_api_key: Optional[str]):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

def _public_base(request: Request) -> str:
    # Prefer env var for correctness behind proxies.
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    # Fallback: derive from incoming request.
    return str(request.base_url).rstrip("/")

def _now() -> float:
    return time.time()

def _safe_copy(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, str(dst))

def _extract_first_media_file(outputs: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to find an output file path from ComfyUI history outputs.

    Returns: (fullpath, filename)
    """
    if not outputs:
        return None, None

    # Common keys seen in ComfyUI/VHS outputs: gifs, videos, video, images
    candidate_keys = ["gifs", "videos", "video", "images"]

    for _, out in outputs.items():
        for key in candidate_keys:
            if key in out and out[key]:
                item = out[key][0]
                # Most robust: fullpath if present
                fullpath = item.get("fullpath")
                filename = item.get("filename")
                if fullpath and os.path.exists(fullpath):
                    return fullpath, filename or os.path.basename(fullpath)

    return None, None

def _download_from_comfyui_view(file_info: dict, save_as: Path) -> None:
    """
    Fallback if ComfyUI doesn't provide fullpath.
    Uses /view?filename=...&subfolder=...&type=...
    """
    filename = file_info["filename"]
    subfolder = file_info.get("subfolder", "")
    filetype = file_info.get("type", "output")

    file_url = f"{COMFYUI_API_URL}/view?filename={urllib.parse.quote(filename)}"
    if subfolder:
        file_url += f"&subfolder={urllib.parse.quote(subfolder)}"
    if filetype:
        file_url += f"&type={urllib.parse.quote(filetype)}"

    resp = requests.get(file_url, timeout=60)
    resp.raise_for_status()
    save_as.parent.mkdir(parents=True, exist_ok=True)
    save_as.write_bytes(resp.content)

def _submit_prompt(req: GenerateVideoRequest) -> str:
    workflow = copy.deepcopy(base_workflow)

    # Update workflow nodes (your existing mapping)
    workflow["16"]["inputs"]["positive_prompt"] = req.positive
    workflow["16"]["inputs"]["negative_prompt"] = req.negative
    workflow["37"]["inputs"]["width"] = req.width
    workflow["37"]["inputs"]["height"] = req.height
    workflow["37"]["inputs"]["num_frames"] = req.length
    workflow["58"]["inputs"]["frame_rate"] = req.fps

    prompt_resp = requests.post(f"{COMFYUI_API_URL}/prompt", json={"prompt": workflow}, timeout=30)
    prompt_resp.raise_for_status()
    return prompt_resp.json()["prompt_id"]

def _fetch_history(job_id: str) -> dict:
    resp = requests.get(f"{COMFYUI_API_URL}/history/{job_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()

def _ensure_job_record(job_id: str) -> dict:
    if job_id not in jobs:
        jobs[job_id] = {
            "job_id": job_id,
            "created_at": _now(),
            "status": "queued",   # queued|running|succeeded|failed|expired
            "error": None,
            "served_path": None,
            "completed_at": None,
        }
    return jobs[job_id]

def _finalize_success(job_id: str, history_item: dict) -> dict:
    outputs = history_item.get("outputs", {})
    fullpath, filename = _extract_first_media_file(outputs)

    served_path = STATIC_DIR_PATH / f"{job_id}.mp4"

    if fullpath:
        _safe_copy(fullpath, served_path)
    else:
        # Try derive file_info from outputs for /view fallback
        file_info = None
        for _, out in outputs.items():
            for k in ("gifs", "videos", "video", "images"):
                if k in out and out[k]:
                    file_info = out[k][0]
                    break
            if file_info:
                break
        if not file_info:
            raise HTTPException(status_code=500, detail="No output file info found")
        _download_from_comfyui_view(file_info, served_path)

    rec = _ensure_job_record(job_id)
    rec["status"] = "succeeded"
    rec["served_path"] = str(served_path)
    rec["completed_at"] = _now()
    return rec

def _status_from_history(job_id: str) -> dict:
    rec = _ensure_job_record(job_id)

    # If file exists already, return succeeded quickly
    if rec.get("status") == "succeeded" and rec.get("served_path"):
        if Path(rec["served_path"]).exists():
            return rec
        # file missing => expired or cleaned
        rec["status"] = "expired"
        rec["served_path"] = None
        return rec

    history = _fetch_history(job_id)

    if job_id not in history:
        # ComfyUI might not have it ready in history yet
        rec["status"] = "running"
        return rec

    item = history[job_id]
    status_obj = item.get("status", {})
    status_str = status_obj.get("status_str")

    if status_str == "error":
        rec["status"] = "failed"
        rec["error"] = status_obj
        return rec

    if status_str != "success":
        rec["status"] = "running"
        return rec

    # success -> ensure we have a served file
    try:
        return _finalize_success(job_id, item)
    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = str(e)
        return rec

async def _cleanup_loop():
    while True:
        try:
            if ENABLE_CLEANUP and JOB_TTL_SECONDS > 0:
                cutoff = _now() - JOB_TTL_SECONDS
                # clean job records + files
                for job_id, rec in list(jobs.items()):
                    created_at = rec.get("created_at", 0)
                    if created_at and created_at < cutoff:
                        sp = rec.get("served_path")
                        if sp and Path(sp).exists():
                            try:
                                Path(sp).unlink()
                            except Exception:
                                pass
                        rec["status"] = "expired"
                        rec["served_path"] = None
        except Exception as e:
            logger.warning(f"cleanup_loop error: {e}")

        await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    if ENABLE_CLEANUP:
        asyncio.create_task(_cleanup_loop())

# ---------------- New API: jobs ----------------

@app.post("/jobs", response_model=dict)
def create_job(req: GenerateVideoRequest, request: Request, x_api_key: str = Header(None)):
    """
    Create an async job. Returns job_id immediately.
    """
    _auth(x_api_key)

    try:
        job_id = _submit_prompt(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit task: {e}")

    rec = _ensure_job_record(job_id)
    rec["status"] = "queued"
    rec["created_at"] = _now()

    return {
        "job_id": job_id,
        "status": rec["status"],
        "status_url": f"{_public_base(request)}/jobs/{job_id}",
        "download_url": None,
    }

@app.get("/jobs/{job_id}", response_model=dict)
def get_job(job_id: str, request: Request, x_api_key: str = Header(None)):
    """
    Poll job status. When succeeded, provides download_url.
    """
    _auth(x_api_key)

    try:
        rec = _status_from_history(job_id)
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"ComfyUI not reachable: {e}")

    download_url = None
    if rec["status"] == "succeeded":
        download_url = f"{_public_base(request)}/jobs/{job_id}/download"

    return {
        "job_id": job_id,
        "status": rec["status"],
        "created_at": rec.get("created_at"),
        "completed_at": rec.get("completed_at"),
        "error": rec.get("error"),
        "download_url": download_url,
    }

# ---------------- Download with Range support ----------------

def _iter_file_range(path: Path, start: int, end: int, chunk_size: int = 1024 * 1024):
    with path.open("rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = f.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk

@app.get("/jobs/{job_id}/download")
def download_job_video(job_id: str, request: Request):
    """
    Stream the MP4. Supports HTTP Range for browser playback/seeking.
    """

    rec = _ensure_job_record(job_id)
    if rec.get("status") != "succeeded" or not rec.get("served_path"):
        # Attempt to refresh status (in case user goes straight to download)
        rec = _status_from_history(job_id)
        if rec.get("status") != "succeeded" or not rec.get("served_path"):
            raise HTTPException(status_code=404, detail="Video not ready")

    path = Path(rec["served_path"])
    if not path.exists():
        rec["status"] = "expired"
        rec["served_path"] = None
        raise HTTPException(status_code=404, detail="Video expired or missing")

    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": "video/mp4",
        "Content-Disposition": f'inline; filename="{path.name}"',
    }

    if not range_header:
        return StreamingResponse(path.open("rb"), headers=headers)

    # Parse: Range: bytes=start-end
    try:
        _, rng = range_header.split("=")
        start_s, end_s = rng.split("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
        end = min(end, file_size - 1)
        if start > end:
            raise ValueError("invalid range")
    except Exception:
        raise HTTPException(status_code=416, detail="Invalid Range header")

    headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    return StreamingResponse(
        _iter_file_range(path, start, end),
        status_code=206,
        headers=headers,
    )

# ---------------- Backward compatible endpoint ----------------

@app.post("/generate_video", response_model=dict)
def generate_video(req: GenerateVideoRequest, request: Request, x_api_key: str = Header(None), wait: int = 1):
    """
    Backward compatible: by default waits for completion (wait=1).
    For webapp async flow, call POST /jobs and poll GET /jobs/{id}.
    """
    _auth(x_api_key)

    # create job
    create = create_job(req, request, x_api_key)

    if not wait:
        return {
            "prompt_id": create["job_id"],
            "status": create["status"],
            "files": [],
            "status_url": create["status_url"],
        }

    # wait (old behavior)
    job_id = create["job_id"]
    start = _now()
    while _now() - start < MAX_WAIT_TIME:
        rec = _status_from_history(job_id)
        if rec["status"] == "succeeded":
            return {
                "prompt_id": job_id,
                "status": "success",
                "files": [
                    {
                        "node_id": "unknown",
                        "type": "video",
                        "url": f"{_public_base(request)}/jobs/{job_id}/download",
                        "filename": rec["served_path"],
                    }
                ],
            }
        if rec["status"] == "failed":
            raise HTTPException(status_code=500, detail=f"ComfyUI task failed: {rec.get('error')}")
        time.sleep(POLL_INTERVAL)

    raise HTTPException(status_code=504, detail="Task timeout")

# ---------------- Health ----------------

@app.get("/health")
def health_check():
    try:
        resp = requests.get(f"{COMFYUI_API_URL}/system_stats", timeout=5)
        resp.raise_for_status()
        return {
            "status": "healthy",
            "comfyui_status": "connected",
            "static_dir": STATIC_DIR,
            "public_base_url": PUBLIC_BASE_URL or None,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ComfyUI not available: {e}")

# ---------------- Optional: keep static hosting too ----------------
# You can keep this, but the /jobs/{id}/download endpoint is the one browsers like best.
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "9003"))
    logger.info(f"Starting API server on 0.0.0.0:{port}")
    logger.info(f"ComfyUI: {COMFYUI_API_URL}")
    logger.info(f"Workflow: {WORKFLOW_FILE}")
    uvicorn.run(app, host="0.0.0.0", port=port)
