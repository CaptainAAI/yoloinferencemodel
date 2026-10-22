# yoloinferencemodel

YOLO object detection over HTTP for an NVIDIA Jetson: a FastAPI server (`server.py`) that runs Ultralytics YOLO
models (TensorRT `.engine`, `.pt`, `.onnx`), each on its own worker thread, so requests for different models run at
the same time. It also manages models and model files, converts `.pt` files to TensorRT, reports monitoring figures
taken from Ultralytics, and can keep every image a model sees in an image server.

It is one of four repositories that work together:

| Repository | What it is | Runs on | Port |
| --- | --- | --- | --- |
| **yoloinferencemodel** (this one) | Inference server `server.py` | Jetson | 6767 |
| [yoloinferencewebserver](https://github.com/CaptainAAI/yoloinferencewebserver) | Web dashboard: monitoring, model management, saved images | Jetson | 8080 |
| [yoloinferenceimageserver](https://github.com/CaptainAAI/yoloinferenceimageserver) | Image store for saved images (filter, delete, export) | Jetson, local only | 6770 |
| [yoloinferencesdkcpp](https://github.com/CaptainAAI/yoloinferencesdkcpp) | C++ SDK `YoloHttp.dll` for Windows programs | Windows PC | none |

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Running it by hand](#running-it-by-hand)
- [Files and folders](#files-and-folders)
- [Configuration](#configuration)
- [Detecting objects](#detecting-objects)
- [Managing models](#managing-models)
- [Model files and TensorRT conversion](#model-files-and-tensorrt-conversion)
- [Monitoring](#monitoring)
- [Saving images](#saving-images)
- [API reference](#api-reference)
- [deploy.sh](#deploysh)
- [How it works](#how-it-works)
- [Troubleshooting](#troubleshooting)

## Features

- **Parallel models.** Each model has its own thread(s) and YOLO instance. Three models answer one image in about
  57 ms together instead of about 110 ms one after another (Jetson Orin, Ethernet).
- **Load, reload and unload models while running.** A reload loads the new copy behind the old one, so requests never
  see the model missing; the model list is kept in `models.json`.
- **Upload, delete and convert model files.** `.pt` files are converted to TensorRT engines in a separate process
  while the server keeps answering.
- **Monitoring from Ultralytics only.** `Results.speed` per request, the model's export metadata and
  `ultralytics.utils`; the server only counts and averages them.
- **Image saving.** A per-model switch sends every image and its boxes to the image server in the background.

## Requirements

- An NVIDIA Jetson (tested: Orin, JetPack with TensorRT 10.16) or any Linux machine with an NVIDIA GPU.
- Python 3.10 or newer (tested 3.12) with Ultralytics, PyTorch and TensorRT installed as for any Ultralytics
  TensorRT setup.
- The web packages in `requirements.txt`:

  ```bash
  pip3 install --user --break-system-packages -r requirements.txt
  ```

  (`--break-system-packages` is needed on Ubuntu 24.04's system Python; leave it out in a virtual environment.)
- Model files, e.g. `model1.engine`, in `~/yoloinference/model/`.

## Quick start

On the Jetson:

```bash
git clone https://github.com/CaptainAAI/yoloinferencemodel.git
cd yoloinferencemodel
bash deploy.sh
```

`deploy.sh` checks the requirements, copies `server.py` to `~/yoloinference`, starts it in the background on port
6767, waits until the models are loaded and prints them:

```
==> Status
LISTEN 0      2048         0.0.0.0:6767       0.0.0.0:*
  model1           ready    model1.engine  save_images=False
  particle         ready    ParticlesDetection_0112.engine  save_images=False
```

Check it from any machine:

```bash
curl http://140.129.7.180:6767/models
```

To update later: `git pull && bash deploy.sh`. To make it start after a reboot: `bash deploy.sh --service`.

## Running it by hand

```bash
cd ~/yoloinference
python3 server.py                                  # port 6767 (or $PORT)
uvicorn server:app --host 0.0.0.0 --port 6767      # the same
```

`fastapi run server.py` works too, but only with `--port 6767`: without it, it uses 8000, which is Triton's port on
the same Jetson.

On the first start, `server.py` creates `models.json` with `model1`, `model2` and `model3` on `model1.engine`,
`model2.engine` and `model3.engine`. From then on the models in `models.json` are loaded.

## Files and folders

Everything the server uses lives next to `server.py` (`~/yoloinference` with `deploy.sh`):

| Path | Contents |
| --- | --- |
| `server.py` | The server |
| `model/` | Model files: `.engine`, `.pt`, `.onnx`. `model/.export/` holds conversions in progress |
| `models.json` | The loaded models: name, file, threads, save switch. Written by the server |
| `settings.json` | The image server address. Written by the server |
| `server.log` | Output when started by `deploy.sh` |
| `server.py.bak-<date>` | The previous version, kept by `deploy.sh` |

## Configuration

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `PORT` | 6767 | Port for `python3 server.py` |
| `WORKERS_PER_MODEL` | 1 | Threads of the default models, only when `models.json` does not exist yet |
| `IMAGE_SERVER` | `http://127.0.0.1:6770` | Image server address until one is set with `PUT /settings` |

`models.json`, one entry per model; the key is the name clients use:

```json
{
  "model1": {"file": "model1.engine", "workers": 1, "save_images": false},
  "particle": {"file": "ParticlesDetection_0112.engine", "workers": 2, "save_images": true}
}
```

Edit it only while the server is stopped; while it runs, use the API or the dashboard, which rewrite it.

**Model name vs. file.** The name (`particle`) is what clients send; the file is only the engine behind it. Several
names may use the same file, and a name can be moved to a new file with a reload without changing any client.

## Detecting objects

`POST /predict` with a multipart form:

```bash
curl -F model_name=model1 -F file=@test.jpg http://140.129.7.180:6767/predict
curl -F model_name=model1 -F conf=0.5 -F iou=0.45 -F file=@test.jpg http://140.129.7.180:6767/predict
```

| Field | Default | Meaning |
| --- | --- | --- |
| `file` | required | Image bytes (JPEG, PNG, BMP, ...) of any size |
| `model_name` | required | A loaded model's name |
| `conf` | 0.25 | Minimum confidence |
| `iou` | 0.7 | IoU threshold of non-maximum suppression |

Answer:

```json
{
  "model": "model1",
  "timing_ms": {
    "upload_read": 0.02, "queue_wait": 0.09, "decode": 7.2, "inference_call": 12.1,
    "model_stages": {"preprocess": 2.4, "inference": 6.4, "postprocess": 3.3},
    "serialization": 0.05, "total_server": 20.4
  },
  "names": {"0": "person", "63": "laptop"},
  "detections": [[100.5, 98.2, 902.0, 603.7, 0.55, 63]],
  "image_id": null
}
```

Each detection is `[x1, y1, x2, y2, score, class_id]` in pixels of the image sent. Ultralytics resizes and pads the
image to the model's input size (from the engine's metadata, e.g. 640 × 640) and maps the boxes back. `image_id` is
set when the model's save switch is on.

Errors: 404 unknown model (the message lists the valid names), 400 empty or not an image, 503 model loading or
failed to load, 500 inference error. All errors are `{"detail": "..."}`.

From C++ use the SDK ([yoloinferencesdkcpp](https://github.com/CaptainAAI/yoloinferencesdkcpp)):
`client.Detect("model1", frame)`.

## Managing models

```bash
S=http://140.129.7.180:6767
# load a model (in the background; /predict answers 503 until it is ready)
curl -X POST -H "Content-Type: application/json" -d '{"name": "model4", "file": "best.engine", "workers": 1}' $S/models
# reload: other thread count and/or another file; the old copy answers until the new one is ready
curl -X POST -H "Content-Type: application/json" -d '{"workers": 2}' $S/models/model4/reload
curl -X POST -H "Content-Type: application/json" -d '{"file": "best_v2.engine"}' $S/models/model4/reload
# unload (requests already waiting are still answered)
curl -X DELETE $S/models/model4
```

`workers` is 1 to 4. With 2 or more, requests for the **same** model also run in parallel, each thread with its own
TensorRT context (more GPU memory). If a reload fails, the old copy stays and its `error` says `Reload failed: ...`.

The dashboard ([yoloinferencewebserver](https://github.com/CaptainAAI/yoloinferencewebserver)) does all of this with
buttons.

## Model files and TensorRT conversion

```bash
curl $S/files                                                   # list model/
curl -F file=@best.pt $S/files                                  # upload (.engine, .pt, .onnx)
curl -F file=@best.engine -F overwrite=true $S/files            # replace (not while a model uses it)
curl -X DELETE $S/files/best.pt                                 # delete (not while a model uses it)
```

Convert a `.pt` file to a TensorRT engine:

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"file": "best.pt", "output": "best.engine", "imgsz": 640, "half": true}' $S/exports
curl $S/exports                                                 # status and log
curl -X DELETE $S/exports/1                                     # cancel
```

- It runs `YOLO("best.pt").export(format="engine", imgsz=..., half=..., device=0, simplify=False)` in a separate
  Python process, one conversion at a time; requests keep being answered. `yolo26n.pt` at 640, FP16 took 385 s.
- `imgsz` is one size or `[height, width]`; each side is rounded up to a multiple of 32 (for 1280 × 720 images use
  `[736, 1280]`). The engine's input size is fixed.
- `half: true` builds FP16, which is about twice as fast as FP32 on the Jetson.
- Only `.pt` files can be converted (Ultralytics exports from the PyTorch model). Build engines on the Jetson that
  runs them: TensorRT warns about engines built on another device.

## Monitoring

`GET /monitor` returns the server's versions and hardware and, per model, its status and statistics. Every figure
comes from Ultralytics:

| Source | Figures |
| --- | --- |
| `Results.speed` and box count of each request | Requests, failures, requests in the last minute, boxes per image, average preprocess / inference / postprocess ms, p50 / p95 / max of the last 600 requests, the last 60 points |
| The model's AutoBackend and export metadata | Task, format, input size, precision (FP32 / FP16 / INT8), batch, device, classes, training dataset, export date and Ultralytics version |
| `ultralytics.utils` | Ultralytics, PyTorch and Python versions, CPU, GPU |

`GET /monitor/<name>` returns one model with all 600 stored requests and their stages. Statistics restart when a
model is reloaded.

## Saving images

Turn on saving for a model, and every image sent to it is kept with its boxes by the image server
([yoloinferenceimageserver](https://github.com/CaptainAAI/yoloinferenceimageserver)):

```bash
curl -X PATCH -H "Content-Type: application/json" -d '{"save_images": true}' $S/models/particle
curl -X PUT -H "Content-Type: application/json" -d '{"image_server": "127.0.0.1:6770"}' $S/settings
curl $S/settings          # {"image_server": ..., "saved": 6, "failed": 0, "queued": 0, "last_error": null}
```

The images go to the image server on a background thread (queue of 200), so detection is not slowed down. If the
image server is down, images are not kept, `failed` counts them, and detection carries on. Clients change nothing;
`/predict` answers with the stored image's `image_id`.

## API reference

| Endpoint | Request | Answer |
| --- | --- | --- |
| `POST /predict` | multipart `file`, `model_name`, `conf`, `iou` | Detections, timing, class names, `image_id` |
| `GET /models` | | `{models: {name: {names, workers, status, file, save_images}}}` |
| `GET /monitor` | | `{server: {...}, models: {name: {status, reloading, error, file, workers, save_images, queue, loaded_at, info, stats}}}` |
| `GET /monitor/{name}` | | One model with its full history |
| `POST /models` | JSON `name`, `file`, `workers`=1, `save_images`=false | 202 `{model, status}` |
| `POST /models/{name}/reload` | JSON `file`?, `workers`? | 202 `{model, status: "reloading"}` |
| `PATCH /models/{name}` | JSON `save_images` | `{model, save_images}` |
| `DELETE /models/{name}` | | `{model, status: "unloaded"}` |
| `GET /files` | | `{files: [{name, size, modified, used_by}]}` |
| `POST /files` | multipart `file`, `overwrite`=false | 201 `{file, size}` |
| `DELETE /files/{filename}` | | `{file, deleted}` |
| `POST /exports` | JSON `file`, `output`?, `imgsz`=640, `half`=true, `overwrite`=false | 202, the conversion |
| `GET /exports` | | `{exports: [{id, file, output, imgsz, half, status, error, created, started, finished, log}]}` |
| `DELETE /exports/{id}` | | The cancelled conversion |
| `GET /settings` | | `{image_server, saved, failed, queued, last_error}` |
| `PUT /settings` | JSON `image_server` | The settings |

FastAPI's interactive docs are at `http://<jetson>:6767/docs`.

## deploy.sh

| Command | What it does |
| --- | --- |
| `bash deploy.sh` | Checks Python and packages, copies `server.py` to `~/yoloinference` (keeps the old one as `server.py.bak-<date>`), restarts it with `nohup`, waits up to 5 minutes for the models, prints the status. Survives logging out, not a reboot |
| `bash deploy.sh --service` | The same as a systemd service `yolo-inference` (starts after a reboot, restarts after a crash, log in `server.log`). Asks for your sudo password once; later plain runs restart the service |
| `bash deploy.sh --status` | Only shows what is running and the models |

Settings: `INFER_DIR` (default `~/yoloinference`), `INFER_PORT` (6767), e.g. `INFER_PORT=6800 bash deploy.sh`.
`model/`, `models.json` and `settings.json` are never touched.

## How it works

- **Threads.** FastAPI's event loop only receives uploads and puts each job in the queue of its model's worker
  thread. The worker decodes the image with Pillow (in memory; the server does not use OpenCV), runs
  `model.predict()`, and returns the boxes. A YOLO / TensorRT instance is never called from two threads.
- **GPU gate.** On the Jetson, a TensorRT engine that is loaded or freed while another engine runs inference fails
  with CUDA errors. Inference takes a shared lock, loading and freeing an exclusive one: loads wait for running
  inferences, new inferences wait 1 to 2 s for the load. Measured: 150 of 150 requests succeeded while models were
  loaded, reloaded and unloaded.
- **Start-up.** Models load one at a time and each runs one warm-up inference, so the first real request is fast. A
  missing or broken model file shows `error` instead of stopping the server.
- **Memory.** Unloading frees the model for the next load; the system's free memory only rises after a restart.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Connection refused` on 6767 | Not running: `bash deploy.sh`, then `tail ~/yoloinference/server.log` |
| `503 Model 'x' is loading` | Wait a few seconds; `GET /models` shows `ready` when done |
| `503 Model 'x' failed to load: ...` | Bad or missing file; check `GET /monitor` → `error`, fix the file, reload |
| `404 Unknown model` | Use a name from `GET /models` (the name, not the file name) |
| A request takes about 1 s | Network path: use the Jetson's Ethernet address, not a Tailscale / Wi-Fi name |
| `[TRT] [W] Using an engine plan file across different models of devices` | The engine was built on another device; convert the `.pt` on this Jetson |
| PyTorch warns about compute capability 8.7 | Inference works; if "no kernel image" errors appear, install NVIDIA's Jetson build of PyTorch |
| Conversion fails with `externally-managed-environment` | Already handled: the server exports with `simplify=False` |

**Security:** there is no login. Anyone who can reach port 6767 can load, unload and delete models. Keep the Jetson
on a trusted network.
