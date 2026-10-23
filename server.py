"""YOLO inference over HTTP, one worker thread per model, with monitoring and
model management.

Every model has its own thread(s) holding its own YOLO instance, so requests
for different models run at the same time instead of waiting for each other.
The FastAPI event loop only receives uploads and hands the work over; image
decoding, inference and serialization all run on the model's thread.

Run:
    uvicorn server:app --host 0.0.0.0 --port 6767
    python server.py                              # same, PORT env var or 6767

Models are listed in models.json next to this file, e.g.
    {"model1": {"file": "model1.engine", "workers": 1}}
with the files in model/. It is created on the first start (from DEFAULT_MODELS, empty by default)
and rewritten whenever a model is loaded, reloaded or unloaded through the API.

Monitoring figures come from Ultralytics only: each result's Results.speed and
box count, the model's AutoBackend and export metadata, and ultralytics.utils
for versions and hardware. The server only counts and averages them.

Endpoints:
    POST   /predict               detect objects (used by the C++ SDK)
    GET    /models                models, their status and class names
    GET    /monitor               server info, per-model status and statistics
    GET    /monitor/{name}        one model with all its recent requests, per stage
    POST   /models                load a model: {"name", "file", "workers"}
    POST   /models/{name}/reload  reload, optionally {"file", "workers"}
    DELETE /models/{name}         unload a model
    GET    /files                 model files in model/
    POST   /files                 upload a model file (.engine, .pt, .onnx)
    DELETE /files/{filename}      delete a model file no model uses
    POST   /exports               convert a .pt file to a TensorRT .engine
    GET    /exports               conversions: status and log
    DELETE /exports/{id}          cancel a conversion
    PATCH  /models/{name}         {"save_images": true | false}
    GET    /settings              image server address and saving counters
    PUT    /settings              {"image_server": "127.0.0.1:6770"}

Saving images: each model has a "save_images" switch (models.json, off by
default, set from the dashboard). While it is on, every image sent to that
model is passed with its detections to the image server
(imageserver/image_server.py, local on the Jetson by default) on a background
thread, so detection is not slowed down; the answer carries its "image_id".
Clients need no change.

Environment:
    WORKERS_PER_MODEL  threads (= YOLO instances / TensorRT contexts) for the
                       default models, default 1. With 2+, requests for the
                       SAME model also run in parallel, at the cost of GPU memory.
    PORT               port for `python server.py`, default 6767
    IMAGE_SERVER       image server address until one is set through /settings,
                       default http://127.0.0.1:6770
"""

import asyncio
import gc
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from itertools import count
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Union

import requests
import ultralytics
import ultralytics.utils as ul_utils
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_cpu_info, get_gpu_info

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
CONFIG_FILE = BASE_DIR / "models.json"
DEFAULT_MODELS = {}  # models for a new install with no models.json; add them from the dashboard
WORKERS_PER_MODEL = max(1, int(os.getenv("WORKERS_PER_MODEL", "1")))
MAX_WORKERS = 4
MODEL_SUFFIXES = {".engine", ".pt", ".onnx"}
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
WARMUP_SIZE = (640, 640)
HISTORY = 600       # recent requests kept per model for /monitor
HISTORY_SENT = 60   # of those, how many /monitor returns point by point
EXPORT_DIR = MODEL_DIR / ".export"  # scratch space of running conversions
EXPORT_TIMEOUT = 3600  # s; a conversion still running then is stopped
EXPORTS_KEPT = 20    # finished conversions listed by /exports
IMAGE_SERVER = os.getenv("IMAGE_SERVER", "http://127.0.0.1:6770").rstrip("/")
SETTINGS_FILE = BASE_DIR / "settings.json"
SAVE_QUEUE = 200     # images waiting for the image server; beyond this new ones are not kept


class _GpuGate:
    """Inference on different models runs in parallel (shared), but loading
    or freeing a TensorRT engine needs the GPU to itself (exclusive).

    On the Jetson, an engine being deserialized or getting its execution
    context while another engine runs fails with CUDA runtime errors, both
    for several loads at once and for one load during inference. A load or
    unload therefore waits for running inferences to finish and holds new
    ones back for its 1-2 s; queued requests just wait a little longer.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._running = 0       # inferences in progress
        self._exclusive = False
        self._waiting = 0       # loads/unloads waiting: they go before new inferences

    @contextmanager
    def shared(self):
        with self._cond:
            while self._exclusive or self._waiting:
                self._cond.wait()
            self._running += 1
        try:
            yield
        finally:
            with self._cond:
                self._running -= 1
                if self._running == 0:
                    self._cond.notify_all()

    @contextmanager
    def exclusive(self):
        with self._cond:
            self._waiting += 1
            while self._exclusive or self._running:
                self._cond.wait()
            self._waiting -= 1
            self._exclusive = True
        try:
            yield
        finally:
            with self._cond:
                self._exclusive = False
                self._cond.notify_all()


_GPU = _GpuGate()


class ModelUnavailable(Exception):
    """The model is loading, failed to load or is being unloaded."""


def _plain(value):
    """Turns what Ultralytics reports (tuples, devices, ...) into JSON values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return str(value)


def _describe(model, path):
    """What Ultralytics reports about a loaded model: its AutoBackend and the
    metadata written into the file when it was exported."""
    predictor = model.predictor
    backend = predictor.model
    meta = getattr(backend, "metadata", None) or {}
    # The engine's precision is in its export arguments (Ultralytics 8.4:
    # quantize=16 / 8; older: half=True). AutoBackend's fp16 only describes the
    # input tensor, which stays FP32 for an FP16 engine.
    export_args = meta.get("args") or {}
    precision = {16: "FP16", 8: "INT8"}.get(export_args.get("quantize")) or (
        "FP16" if export_args.get("half") or getattr(backend, "fp16", False) else "FP32"
    )
    return _plain({
        "task": model.task,
        "format": getattr(backend, "format", None) or path.suffix.lstrip("."),
        "imgsz": predictor.args.imgsz,
        "precision": precision,
        "batch": getattr(backend, "batch", None),
        "device": predictor.device,
        "classes": len(model.names),
        "description": meta.get("description"),
        "exported": meta.get("date"),
        "exported_with": meta.get("version"),
    })


def _server_info():
    """Versions and hardware as ultralytics.utils reports them."""
    try:
        gpu = get_gpu_info(0)
    except Exception:
        gpu = None
    return _plain({
        "ultralytics": ultralytics.__version__,
        "python": getattr(ul_utils, "PYTHON_VERSION", None),
        "torch": getattr(ul_utils, "TORCH_VERSION", None),
        "cpu": get_cpu_info(),
        "gpu": gpu,
    })


class ModelStats:
    """Request statistics of one model, built from each result's Results.speed
    (preprocess / inference / postprocess ms) and box count."""

    def __init__(self):
        self._lock = threading.Lock()
        self.ok = 0
        self.failed = 0
        self.detections = 0
        self.last_error = None
        self.speed_sum = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
        self.recent = deque(maxlen=HISTORY)  # (time, preprocess, inference, postprocess ms, detections)

    def add(self, speed, detections):
        stages = [speed.get(key) or 0.0 for key in self.speed_sum]
        with self._lock:
            self.ok += 1
            self.detections += detections
            for key, value in zip(self.speed_sum, stages):
                self.speed_sum[key] += value
            self.recent.append((time.time(), *stages, detections))

    def fail(self, exc):
        with self._lock:
            self.failed += 1
            self.last_error = f"{type(exc).__name__}: {exc}"

    def snapshot(self, full=False):
        """Totals and recent percentiles; `history` = the last HISTORY_SENT
        requests, or with full=True all HISTORY of them with their stages."""
        with self._lock:
            ok, failed, detections = self.ok, self.failed, self.detections
            last_error, sums = self.last_error, dict(self.speed_sum)
            recent = list(self.recent)
        now = time.time()
        totals = sorted(pre + inf + post for _, pre, inf, post, _ in recent)

        def percentile(p):
            return round(totals[min(len(totals) - 1, int(p * len(totals)))], 2) if totals else None

        return {
            "requests": ok,
            "failed": failed,
            "last_error": last_error,
            "per_minute": sum(1 for r in recent if now - r[0] <= 60),
            "detections_per_image": round(detections / ok, 2) if ok else None,
            "avg_ms": {key: round(value / ok, 2) for key, value in sums.items()} if ok else None,
            "recent_ms": {
                "p50": percentile(0.5),
                "p95": percentile(0.95),
                "max": round(totals[-1], 2) if totals else None,
                "window": len(totals),
            },
            "history": [
                {"t": round(t, 3), "ms": round(pre + inf + post, 2), "detections": d}
                for t, pre, inf, post, d in recent[-HISTORY_SENT:]
            ] if not full else [
                {
                    "t": round(t, 3), "ms": round(pre + inf + post, 2), "preprocess": round(pre, 2),
                    "inference": round(inf, 2), "postprocess": round(post, 2), "detections": d,
                }
                for t, pre, inf, post, d in recent
            ],
        }


class ModelWorker:
    """Thread(s) that own one model and run its jobs in arrival order.

    A YOLO instance (and its TensorRT execution context) is not safe to call
    from several threads at once, so each thread loads its own instance and
    is the only one that touches it.

    status: "loading" -> "ready" (or "error"); "stopping" once unloaded.
    During a reload, `replacement` is the new worker loading in the background.
    """

    def __init__(self, name, file, workers, save_images=False):
        self.name = name
        self.file = file
        self.save_images = save_images  # pass every image to the image server
        self.path = MODEL_DIR / file
        self.names = {}
        self.info = {}
        self.status = "loading"
        self.error = None
        self.loaded_at = None
        self.replacement = None
        self.stats = ModelStats()
        self.jobs = queue.Queue()
        self._lock = threading.Lock()  # status, error, info, _loaded; ordering of submit vs stop
        self._loaded = 0
        self._done = threading.Event()  # every thread finished loading, well or not
        self.threads = [
            threading.Thread(target=self._run, name=f"{name}-{i}", daemon=True)
            for i in range(workers)
        ]

    def start(self):
        if not self.path.is_file():
            self._set_error(f"Model file not found: {self.file}")
            self._done.set()
            return
        for thread in self.threads:
            thread.start()

    def wait_loaded(self):
        self._done.wait()

    def stop(self):
        # Under the lock, so no job can be queued behind the stop markers and
        # wait forever; jobs already queued are still answered first.
        with self._lock:
            self.status = "stopping"
            for _ in self.threads:
                self.jobs.put(None)

    def submit(self, image_bytes, conf, iou):
        future = Future()
        with self._lock:
            if self.status == "error":
                raise ModelUnavailable(f"Model '{self.name}' failed to load: {self.error}")
            if self.status != "ready":
                raise ModelUnavailable(f"Model '{self.name}' is {self.status}")
            self.jobs.put((future, time.perf_counter(), image_bytes, conf, iou))
        return future

    def _set_error(self, message):
        with self._lock:
            if self.status in ("loading", "ready"):
                self.status = "error"
            self.error = self.error or message

    def _loaded_one(self, model):
        with self._lock:
            self._loaded += 1
            if model is not None and not self.info:
                self.names = model.names
                self.info = _describe(model, self.path)
            if self._loaded == len(self.threads):
                if self.status == "loading":
                    self.status = "ready"
                    self.loaded_at = time.time()
                self._done.set()

    def _run(self):
        try:
            with _GPU.exclusive():
                model = YOLO(str(self.path), task="detect")
                # The first predict() builds the predictor and the TensorRT
                # context; do it here so the first real request is not slow.
                model.predict(Image.new("RGB", WARMUP_SIZE), verbose=False)
        except Exception as exc:
            self._set_error(f"{type(exc).__name__}: {exc}")
            self._loaded_one(None)
            return
        self._loaded_one(model)

        while True:
            job = self.jobs.get()
            if job is None:
                break
            future, enqueued, image_bytes, conf, iou = job
            if not future.set_running_or_notify_cancel():
                continue
            try:
                out = self._infer(model, enqueued, image_bytes, conf, iou)
            except Exception as exc:
                self.stats.fail(exc)
                future.set_exception(exc)
                continue
            self.stats.add(out["model_stages"], len(out["detections"]))
            future.set_result(out)

        # Unloaded: free the engine now, with the GPU to itself (_GpuGate).
        with _GPU.exclusive():
            del model
            gc.collect()

    @staticmethod
    def _infer(model, enqueued, image_bytes, conf, iou):
        start = time.perf_counter()
        queue_wait_ms = (start - enqueued) * 1000

        # Decoded in memory (no temp file); YOLO takes the PIL image as is.
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Uploaded file is not a decodable image") from exc
        decode_ms = (time.perf_counter() - start) * 1000

        with _GPU.shared():
            inference_start = time.perf_counter()
            result = model.predict(image, conf=conf, iou=iou, verbose=False)[0]
            inference_call_ms = (time.perf_counter() - inference_start) * 1000

        serialization_start = time.perf_counter()
        detections = (
            result.boxes.data.cpu().tolist() if result.boxes is not None else []
        )
        serialization_ms = (time.perf_counter() - serialization_start) * 1000

        return {
            "detections": detections,
            "names": result.names,
            "model_stages": result.speed,
            "queue_wait": queue_wait_ms,
            "decode": decode_ms,
            "inference_call": inference_call_ms,
            "serialization": serialization_ms,
        }


# Run in a separate Python process: argv[1] = the .pt file, argv[2] = export() arguments as JSON.
_EXPORT_SCRIPT = (
    "import json, sys\n"
    "from ultralytics import YOLO\n"
    "print('ENGINE', YOLO(sys.argv[1]).export(**json.loads(sys.argv[2])), flush=True)\n"
)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class ExportJob:
    """Converts a .pt file into a TensorRT .engine with Ultralytics' export().

    It runs in its own process: building an engine takes minutes and a lot of
    memory, and a separate CUDA context keeps it away from the engines this
    server is running (the _GpuGate conflict is within one process). The .pt
    is copied to a scratch folder so export() never writes over files in
    model/; only the finished engine is moved there.

    status: "queued" -> "running" -> "done" | "failed" | "cancelled".
    """

    def __init__(self, job_id, file, output, imgsz, half):
        self.id = job_id
        self.file = file
        self.output = output
        self.imgsz = imgsz
        self.half = half
        self.status = "queued"
        self.error = None
        self.created = time.time()
        self.started = None
        self.finished = None
        self.log = deque(maxlen=200)  # Ultralytics' and TensorRT's output lines
        self._proc = None
        self._timed_out = False
        self._lock = threading.Lock()

    def view(self):
        return {
            "id": self.id,
            "file": self.file,
            "output": self.output,
            "imgsz": self.imgsz,
            "half": self.half,
            "status": self.status,
            "error": self.error,
            "created": round(self.created, 3),
            "started": round(self.started, 3) if self.started else None,
            "finished": round(self.finished, 3) if self.finished else None,
            "log": list(self.log)[-40:],
        }

    def cancel(self):
        with self._lock:
            if self.status == "queued":
                self.status = "cancelled"
                self.finished = time.time()
                return True
            if self.status == "running" and self._proc:
                self.status = "cancelling"
                self._proc.kill()
                return True
        return False

    def _kill_on_timeout(self):
        self._timed_out = True
        self._proc.kill()

    def run(self):
        with self._lock:
            if self.status != "queued":
                return
            self.status = "running"
            self.started = time.time()
        work = EXPORT_DIR / self.id
        try:
            work.mkdir(parents=True, exist_ok=True)
            source = work / self.file
            shutil.copy2(MODEL_DIR / self.file, source)
            # simplify=False: the ONNX simplifier (onnxslim) is not installed on the
            # Jetson and pip refuses to auto-install it; TensorRT optimizes anyway.
            args = {"format": "engine", "imgsz": self.imgsz, "half": self.half, "device": 0, "simplify": False}
            with self._lock:
                self._proc = subprocess.Popen(
                    [sys.executable, "-u", "-c", _EXPORT_SCRIPT, str(source), json.dumps(args)],
                    cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, errors="replace",
                )
            timer = threading.Timer(EXPORT_TIMEOUT, self._kill_on_timeout)
            timer.start()
            for line in self._proc.stdout:
                line = _ANSI.sub("", line).rstrip()  # Ultralytics colours its log
                if line:
                    self.log.append(line)
            code = self._proc.wait()
            timer.cancel()

            engine = source.with_suffix(".engine")
            with self._lock:
                if self.status == "cancelling":
                    self.status = "cancelled"
                elif code == 0 and engine.is_file():
                    if _users_of(self.output):
                        raise RuntimeError(f"'{self.output}' was loaded by a model meanwhile; not replaced")
                    engine.replace(MODEL_DIR / self.output)
                    self.status = "done"
                else:
                    self.status = "failed"
                    reason = f"stopped after {EXPORT_TIMEOUT} s" if self._timed_out else f"exit code {code}"
                    self.error = f"Export failed ({reason})" + (f": {self.log[-1]}" if self.log else "")
        except Exception as exc:
            self.status = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.finished = time.time()
            shutil.rmtree(work, ignore_errors=True)


def _new_image_id():
    """Sortable, unique: 20261022-114805-3fa2c9d1 (the image server's format)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]


class ImageSaver:
    """Hands images to the image server (image_server.py) on a background
    thread, so keeping an image never slows /predict down. If the image
    server is down or too slow, the image is not kept and `failed` counts it;
    /predict still answers normally.

    `server` is the image server's address, set from the dashboard.
    """

    def __init__(self):
        self.server = IMAGE_SERVER
        self.saved = 0
        self.failed = 0
        self.last_error = None
        self._jobs = queue.Queue(maxsize=SAVE_QUEUE)

    def start(self):
        threading.Thread(target=self._run, name="image-saver", daemon=True).start()

    def submit(self, image_id, image_bytes, filename, model, out, meta):
        try:
            self._jobs.put_nowait((image_id, image_bytes, filename, model, out["detections"], out["names"], meta))
            return True
        except queue.Full:
            self.failed += 1
            self.last_error = "Image server too slow: save queue full, image not kept"
            return False

    def _run(self):
        session = requests.Session()
        while True:
            image_id, image_bytes, filename, model, detections, names, meta = self._jobs.get()
            try:
                reply = session.post(
                    f"{self.server}/images",
                    files={"file": (filename, image_bytes)},
                    data={
                        "id": image_id,
                        "model": model,
                        "detections": json.dumps(detections),
                        "names": json.dumps(names),
                        "meta": json.dumps(meta),
                    },
                    timeout=15,
                )
                if reply.status_code >= 300:
                    raise RuntimeError(f"HTTP {reply.status_code}: {reply.text[:200]}")
                self.saved += 1
            except Exception as exc:
                self.failed += 1
                self.last_error = f"{type(exc).__name__}: {exc}"

    def view(self):
        return {
            "image_server": self.server,
            "saved": self.saved,
            "failed": self.failed,
            "queued": self._jobs.qsize(),
            "last_error": self.last_error,
        }


_saver = ImageSaver()


def _load_settings():
    try:
        data = json.loads(SETTINGS_FILE.read_text())
    except (OSError, ValueError):
        data = {}
    _saver.server = _normalize_address(data.get("image_server") or IMAGE_SERVER)


def _save_settings():
    SETTINGS_FILE.write_text(json.dumps({"image_server": _saver.server}, indent=2) + "\n")


def _normalize_address(address):
    """'140.129.7.180:6770', 'http://host:6770/' -> 'http://host:6770'."""
    address = address.strip().rstrip("/")
    return address if "://" in address else "http://" + address


_exports = {}  # id -> ExportJob, oldest first
_export_queue = queue.Queue()
_export_ids = count(1)


def _export_worker():
    # One conversion at a time: each one needs much of the Jetson's memory.
    while True:
        _export_queue.get().run()


app = FastAPI()
workers = {}
_registry_lock = threading.Lock()  # changes to `workers` and models.json
_started = time.time()
_server = {}


def _load_config():
    if CONFIG_FILE.is_file():
        return json.loads(CONFIG_FILE.read_text())
    return {
        name: {"file": file, "workers": WORKERS_PER_MODEL}
        for name, file in DEFAULT_MODELS.items()
    }


def _save_config():
    data = {
        name: {"file": w.file, "workers": len(w.threads), "save_images": w.save_images}
        for name, w in workers.items()
    }
    tmp = CONFIG_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(CONFIG_FILE)


def _check_file_name(filename):
    """A plain file name in model/ with a model suffix; anything else is refused."""
    if (
        not filename
        or Path(filename).name != filename
        or Path(filename).suffix.lower() not in MODEL_SUFFIXES
    ):
        raise HTTPException(
            status_code=400,
            detail=f"'{filename}' is not a model file name ({', '.join(sorted(MODEL_SUFFIXES))})",
        )
    return filename


def _check_workers(n):
    if not 1 <= n <= MAX_WORKERS:
        raise HTTPException(status_code=400, detail=f"workers must be 1 to {MAX_WORKERS}")
    return n


def _users_of(filename):
    return [name for name, w in list(workers.items()) if w.file == filename]


@app.on_event("startup")
def load_models():
    _server.update(_server_info())
    MODEL_DIR.mkdir(exist_ok=True)
    shutil.rmtree(EXPORT_DIR, ignore_errors=True)  # left over from an interrupted conversion
    threading.Thread(target=_export_worker, name="export", daemon=True).start()
    _load_settings()
    _saver.start()
    for name, entry in _load_config().items():
        workers[name] = ModelWorker(
            name, entry["file"], entry.get("workers", WORKERS_PER_MODEL), bool(entry.get("save_images"))
        )

    # Each worker loads its model on its own thread, one at a time (_GpuGate).
    for worker in workers.values():
        worker.start()
    for worker in workers.values():
        worker.wait_loaded()
    if not CONFIG_FILE.is_file():
        _save_config()
    print("Models: " + ", ".join(f"{n} ({w.status})" for n, w in workers.items()))


@app.on_event("shutdown")
def stop_workers():
    for worker in workers.values():
        worker.stop()


@app.get("/models")
def list_models():
    return {
        "models": {
            name: {
                "names": worker.names,
                "workers": len(worker.threads),
                "status": worker.status,
                "file": worker.file,
                "save_images": worker.save_images,
            }
            for name, worker in list(workers.items())
        }
    }


def _model_view(w, full=False):
    return {
        "status": w.status,
        "reloading": w.replacement is not None,
        "error": w.error,
        "file": w.file,
        "workers": len(w.threads),
        "save_images": w.save_images,
        "queue": w.jobs.qsize(),
        "loaded_at": round(w.loaded_at, 3) if w.loaded_at else None,
        "info": w.info,
        "stats": w.stats.snapshot(full),
    }


@app.get("/monitor")
def monitor():
    return {
        "server": {**_server, "started": round(_started, 3), "uptime_s": round(time.time() - _started)},
        "models": {name: _model_view(w) for name, w in list(workers.items())},
    }


@app.get("/monitor/{name}")
def monitor_model(name: str):
    worker = workers.get(name)
    if worker is None:
        raise HTTPException(status_code=404, detail=f"Unknown model '{name}'")
    return {"name": name, **_model_view(worker, full=True)}


class ModelSpec(BaseModel):
    name: str
    file: str
    workers: int = 1
    save_images: bool = False


class ModelPatch(BaseModel):
    save_images: bool


class ReloadSpec(BaseModel):
    file: Optional[str] = None
    workers: Optional[int] = None


@app.post("/models", status_code=202)
def add_model(spec: ModelSpec):
    if not NAME_PATTERN.match(spec.name):
        raise HTTPException(status_code=400, detail="name: 1-64 letters, digits, '_' or '-'")
    _check_file_name(spec.file)
    _check_workers(spec.workers)
    if not (MODEL_DIR / spec.file).is_file():
        raise HTTPException(status_code=404, detail=f"No file '{spec.file}' in model/")
    with _registry_lock:
        if spec.name in workers:
            raise HTTPException(status_code=409, detail=f"Model '{spec.name}' already exists")
        worker = workers[spec.name] = ModelWorker(spec.name, spec.file, spec.workers, spec.save_images)
        _save_config()
    worker.start()
    return {"model": spec.name, "status": worker.status}


@app.post("/models/{name}/reload", status_code=202)
def reload_model(name: str, spec: Optional[ReloadSpec] = None):
    """Loads a new copy in the background; the old one keeps answering until
    the new one is ready, so requests never see the model missing."""
    spec = spec or ReloadSpec()
    with _registry_lock:
        old = workers.get(name)
        if old is None:
            raise HTTPException(status_code=404, detail=f"Unknown model '{name}'")
        if old.replacement is not None:
            raise HTTPException(status_code=409, detail=f"Model '{name}' is already reloading")
        file = _check_file_name(spec.file) if spec.file else old.file
        count = _check_workers(spec.workers) if spec.workers else len(old.threads)
        if not (MODEL_DIR / file).is_file():
            raise HTTPException(status_code=404, detail=f"No file '{file}' in model/")
        new = old.replacement = ModelWorker(name, file, count, old.save_images)
    new.start()
    threading.Thread(target=_finish_reload, args=(old, new), daemon=True).start()
    return {"model": name, "status": "reloading"}


def _finish_reload(old, new):
    new.wait_loaded()
    with _registry_lock:
        old.replacement = None
        swap = workers.get(old.name) is old and new.status == "ready"
        if swap:
            new.save_images = old.save_images  # switched during the reload: keep it
            workers[old.name] = new
            _save_config()
        elif workers.get(old.name) is old:
            # The new copy failed: keep serving with the old one, say why.
            old.error = f"Reload failed: {new.error}"
    (old if swap else new).stop()


@app.patch("/models/{name}")
def patch_model(name: str, spec: ModelPatch):
    with _registry_lock:
        worker = workers.get(name)
        if worker is None:
            raise HTTPException(status_code=404, detail=f"Unknown model '{name}'")
        worker.save_images = spec.save_images
        _save_config()
    return {"model": name, "save_images": worker.save_images}


@app.delete("/models/{name}")
def unload_model(name: str):
    with _registry_lock:
        worker = workers.pop(name, None)
        if worker is None:
            raise HTTPException(status_code=404, detail=f"Unknown model '{name}'")
        _save_config()
    worker.stop()
    return {"model": name, "status": "unloaded"}


@app.get("/files")
def list_files():
    files = sorted(p for p in MODEL_DIR.iterdir() if p.is_file() and p.suffix.lower() in MODEL_SUFFIXES)
    return {
        "files": [
            {
                "name": p.name,
                "size": p.stat().st_size,
                "modified": round(p.stat().st_mtime, 3),
                "used_by": _users_of(p.name),
            }
            for p in files
        ]
    }


@app.post("/files", status_code=201)
def upload_file(file: UploadFile = File(...), overwrite: bool = Form(False)):
    name = _check_file_name(file.filename)
    dest = MODEL_DIR / name
    if dest.exists():
        if not overwrite:
            raise HTTPException(status_code=409, detail=f"'{name}' already exists")
        if _users_of(name):
            raise HTTPException(
                status_code=409,
                detail=f"'{name}' is used by {', '.join(_users_of(name))}; unload it first",
            )
    # Written next to the target, then renamed: a half-uploaded file never
    # replaces a good one.
    part = MODEL_DIR / (name + ".part")
    with part.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    part.replace(dest)
    return {"file": name, "size": dest.stat().st_size}


@app.delete("/files/{filename}")
def delete_file(filename: str):
    _check_file_name(filename)
    path = MODEL_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No file '{filename}' in model/")
    if _users_of(filename):
        raise HTTPException(
            status_code=409,
            detail=f"'{filename}' is used by {', '.join(_users_of(filename))}; unload it first",
        )
    path.unlink()
    return {"file": filename, "deleted": True}


class ExportSpec(BaseModel):
    file: str
    output: Optional[str] = None
    imgsz: Union[int, List[int]] = 640  # 640, or [height, width] such as [736, 1280]
    half: bool = True
    overwrite: bool = False


@app.post("/exports", status_code=202)
def start_export(spec: ExportSpec):
    _check_file_name(spec.file)
    if Path(spec.file).suffix.lower() != ".pt":
        raise HTTPException(status_code=400, detail="Only .pt files can be converted: Ultralytics exports from the PyTorch model")
    if not (MODEL_DIR / spec.file).is_file():
        raise HTTPException(status_code=404, detail=f"No file '{spec.file}' in model/")
    output = _check_file_name(spec.output or Path(spec.file).stem + ".engine")
    if Path(output).suffix.lower() != ".engine":
        raise HTTPException(status_code=400, detail="The output must be an .engine file")
    sizes = [spec.imgsz] if isinstance(spec.imgsz, int) else list(spec.imgsz)
    if len(sizes) not in (1, 2) or not all(32 <= v <= 4096 for v in sizes):
        raise HTTPException(status_code=400, detail="imgsz: one size or [height, width], each 32 to 4096")
    if (MODEL_DIR / output).exists():
        if not spec.overwrite:
            raise HTTPException(status_code=409, detail=f"'{output}' already exists")
        if _users_of(output):
            raise HTTPException(status_code=409, detail=f"'{output}' is used by {', '.join(_users_of(output))}; unload it first")
    with _registry_lock:
        if any(j.output == output and j.status in ("queued", "running") for j in _exports.values()):
            raise HTTPException(status_code=409, detail=f"A conversion to '{output}' is already queued or running")
        job = ExportJob(str(next(_export_ids)), spec.file, output, sizes[0] if len(sizes) == 1 else sizes, spec.half)
        _exports[job.id] = job
        finished = [j for j in _exports.values() if j.finished]
        for old in finished[:-EXPORTS_KEPT]:
            del _exports[old.id]
    _export_queue.put(job)
    return job.view()


@app.get("/exports")
def list_exports():
    return {"exports": [job.view() for job in reversed(list(_exports.values()))]}


@app.delete("/exports/{job_id}")
def cancel_export(job_id: str):
    job = _exports.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown conversion '{job_id}'")
    if not job.cancel():
        raise HTTPException(status_code=409, detail=f"Conversion {job_id} is already {job.status}")
    return job.view()


class SettingsSpec(BaseModel):
    image_server: str


@app.get("/settings")
def get_settings():
    return _saver.view()


@app.put("/settings")
def put_settings(spec: SettingsSpec):
    if not re.fullmatch(r"(https?://)?[A-Za-z0-9._-]+(:\d{1,5})?/?", spec.image_server.strip()):
        raise HTTPException(status_code=400, detail="image_server: an address like 127.0.0.1:6770")
    _saver.server = _normalize_address(spec.image_server)
    _saver.last_error = None  # it was about the previous address
    _save_settings()
    return _saver.view()


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form(...),
    conf: float = Form(0.25),
    iou: float = Form(0.7),
):
    request_start = time.perf_counter()

    worker = workers.get(model_name)
    if worker is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{model_name}'. Choose from: {list(workers)}",
        )

    upload_start = time.perf_counter()
    image_bytes = await file.read()
    upload_read_ms = (time.perf_counter() - upload_start) * 1000

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Awaiting frees the event loop: other requests are accepted and sent to
    # their own model's thread while this one runs.
    try:
        out = await asyncio.wrap_future(worker.submit(image_bytes, conf, iou))
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    total_server_ms = (time.perf_counter() - request_start) * 1000
    timing = {
        "upload_read": round(upload_read_ms, 2),
        "queue_wait": round(out["queue_wait"], 2),
        "decode": round(out["decode"], 2),
        "inference_call": round(out["inference_call"], 2),
        "model_stages": {
            key: round(value, 2) for key, value in out["model_stages"].items()
        },
        "serialization": round(out["serialization"], 2),
        "total_server": round(total_server_ms, 2),
    }

    # Kept in the background by the image server; the id is known right away.
    image_id = None
    if worker.save_images:
        image_id = _new_image_id()
        meta = {
            "timing_ms": timing,
            "conf": conf,
            "iou": iou,
            "source": request.client.host if request.client else None,
        }
        if not _saver.submit(image_id, image_bytes, file.filename or "image.jpg", model_name, out, meta):
            image_id = None

    return {
        "model": model_name,
        "timing_ms": timing,
        "names": out["names"],
        "detections": out["detections"],
        "image_id": image_id,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "6767")))
