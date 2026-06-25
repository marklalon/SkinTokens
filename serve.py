"""
SkinTokens / TokenRig inference server — FastAPI persistent service.

The model pipeline is loaded once at startup and kept resident in VRAM.
Clients submit a 3D file (OBJ/FBX/GLB) and receive a rigged GLB over HTTP or
WebSocket. Generations are serialized through a single-GPU work queue so the
server can accept many concurrent connections while running one job at a time.

.. note::

    Results are **non-deterministic** — the same input file may produce a
    different skeleton (joint count / structure) across runs, even with identical
    parameters (``do_sample=False``, same ``num_beams``).  This is inherent to
    flash_attention_2 + bf16: the tiling algorithm introduces tiny floating-point
    differences per forward pass, which accumulate over 28 transformer layers and
    cause beam‑search tie‑breaking to diverge.  TF32 remains enabled for speed;
    disabling it would reduce — but not eliminate — this effect.

Run:
    python serve.py --host 0.0.0.0 --port 8087

Environment variables:
    SKINTOKENS_MODEL_CKPT     Path to the model checkpoint
    SKINTOKENS_HF_PATH        Optional HuggingFace model path
    SKINTOKENS_DEVICE         Device override (default: cuda if available)
"""

import os

os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import asyncio
import atexit
import base64
import logging
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import List, Optional

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logger = logging.getLogger("skintokens.serve")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

logger.info("Startup progress: importing runtime dependencies")
_runtime_import_started = time.monotonic()
import torch
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from tqdm import tqdm

logger.info("Startup progress: runtime dependencies imported elapsed=%.2fs",
            time.monotonic() - _runtime_import_started)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_CKPT = os.environ.get(
    "SKINTOKENS_MODEL_CKPT",
    "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
)
HF_PATH = os.environ.get("SKINTOKENS_HF_PATH") or None
DEVICE = os.environ.get("SKINTOKENS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
STARTUP_HEARTBEAT_SEC = max(
    1.0, float(os.environ.get("SKINTOKENS_STARTUP_HEARTBEAT_SEC", "15"))
)
SUPPORTED_EXT = {".obj", ".fbx", ".glb"}


def _run_startup_stage(label: str, operation):
    """Run a startup operation with start/end logs and an elapsed heartbeat."""
    started_at = time.monotonic()
    finished = threading.Event()
    logger.info("Startup stage started: %s", label)

    def heartbeat() -> None:
        while not finished.wait(STARTUP_HEARTBEAT_SEC):
            logger.info("Startup stage in progress: %s elapsed=%.2fs",
                        label, time.monotonic() - started_at)

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        result = operation()
    except Exception:
        logger.exception("Startup stage failed: %s elapsed=%.2fs",
                         label, time.monotonic() - started_at)
        raise
    finally:
        finished.set()
        heartbeat_thread.join()
    logger.info("Startup stage completed: %s elapsed=%.2fs",
                label, time.monotonic() - started_at)
    return result


# --------------------------------------------------------------------------- #
# Global state
# --------------------------------------------------------------------------- #
class ServerState:
    model = None
    tokenizer = None
    transform = None
    bpy_proc: Optional[subprocess.Popen] = None
    ready: bool = False
    loaded_at: float = 0.0
    busy: bool = False
    generation_lock: asyncio.Lock = asyncio.Lock()


state = ServerState()


class GenerationCancelled(Exception):
    """Raised when a generation request has been cancelled."""


class CancellationToken:
    """Cancellation state shared safely by the event loop and worker threads."""

    def __init__(self) -> None:
        self._thread_event = threading.Event()
        self._async_event = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        self._reason = "generation cancelled"
        self._reason_lock = threading.Lock()

    @property
    def reason(self) -> str:
        with self._reason_lock:
            return self._reason

    @property
    def cancelled(self) -> bool:
        return self._thread_event.is_set()

    def cancel(self, reason: str) -> None:
        with self._reason_lock:
            if self._thread_event.is_set():
                return
            self._reason = reason
            self._thread_event.set()
        try:
            self._loop.call_soon_threadsafe(self._async_event.set)
        except RuntimeError:
            pass

    async def wait(self) -> None:
        await self._async_event.wait()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise GenerationCancelled(self.reason)


@asynccontextmanager
async def _acquire_or_cancel(lock: asyncio.Lock, cancellation: CancellationToken):
    """Acquire a lock, but immediately remove cancelled queued work."""
    cancellation.raise_if_cancelled()
    acquire_task = asyncio.create_task(lock.acquire())
    cancel_task = asyncio.create_task(cancellation.wait())
    lock_held = False
    try:
        await asyncio.wait(
            (acquire_task, cancel_task), return_when=asyncio.FIRST_COMPLETED
        )
        if cancellation.cancelled:
            cancellation.raise_if_cancelled()
        await acquire_task
        lock_held = True
        cancel_task.cancel()
        with suppress(asyncio.CancelledError):
            await cancel_task
        yield
    finally:
        if not acquire_task.done():
            acquire_task.cancel()
        try:
            acquired = await acquire_task
        except asyncio.CancelledError:
            acquired = False
        if acquired and not lock_held:
            lock_held = True
        if not cancel_task.done():
            cancel_task.cancel()
        with suppress(asyncio.CancelledError):
            await cancel_task
        if lock_held:
            lock.release()


async def _to_thread_cancellable(operation, *args, cancellation: CancellationToken):
    """Keep a lock held until cancelled worker code has really stopped."""
    worker = asyncio.create_task(asyncio.to_thread(operation, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancellation.cancel("server request task cancelled")
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except GenerationCancelled:
                break
        if worker.done() and not worker.cancelled():
            with suppress(Exception):
                worker.result()
        raise


# --------------------------------------------------------------------------- #
# Generation parameters
# --------------------------------------------------------------------------- #
class GenParams(BaseModel):
    top_k: int = 5
    top_p: float = 0.95
    temperature: float = 1.0
    repetition_penalty: float = 2.0
    num_beams: int = 3
    do_sample: bool = False
    use_skeleton: bool = False
    use_transfer: bool = False
    use_postprocess: bool = False


# --------------------------------------------------------------------------- #
# Pipeline loading
# --------------------------------------------------------------------------- #
def _start_bpy_server():
    """Start the Blender Python server subprocess."""
    popen_kwargs = dict(
        args=[sys.executable, "bpy_server.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(**popen_kwargs)
    logger.info("bpy_server.py started (pid=%d)", proc.pid)

    def cleanup():
        logger.info("Terminating bpy_server.py (pid=%d)", proc.pid)
        try:
            if proc.poll() is not None:
                return
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

    atexit.register(cleanup)
    return proc


def _wait_for_bpy_server(timeout: float = 30):
    """Wait until the bpy_server is ready to accept requests."""
    import requests
    from src.server.spec import BPY_SERVER

    t0 = time.time()
    while True:
        try:
            requests.get(f"{BPY_SERVER}/ping", timeout=1)
            logger.info("bpy_server is ready")
            return
        except Exception:
            if time.time() - t0 > timeout:
                raise RuntimeError("bpy_server failed to start within timeout")
            time.sleep(0.5)


def _load_pipeline():
    """Load the model pipeline and start the bpy_server.

    The bpy_server and the model are loaded in parallel.  The bpy_server
    is started first (fast subprocess fork), but we do not wait for it to
    become healthy until after the model has also been loaded.  This
    overlaps the ~4 s Blender startup with the ~10 s model load.
    """
    from src.data.transform import Transform
    from src.tokenizer.parse import get_tokenizer
    from src.server.spec import get_model

    # 1. Start bpy_server subprocess (fast — just a fork)
    state.bpy_proc = _run_startup_stage("starting bpy_server", _start_bpy_server)

    # 2. Kick off bpy_server readiness check in a background thread so it
    #    runs concurrently with model loading.
    bpy_error: list[Exception | None] = [None]

    def _wait_for_bpy_bg():
        try:
            _wait_for_bpy_server()
        except Exception as exc:
            bpy_error[0] = exc
        finally:
            pass  # thread exits, join() handles synchronization

    bpy_thread = threading.Thread(target=_wait_for_bpy_bg, daemon=True)
    bpy_thread.start()

    # 3. Load model (this dominates startup time)
    def load_model_fn():
        model = get_model(MODEL_CKPT, hf_path=HF_PATH, device=DEVICE)
        tokenizer = get_tokenizer(**model.tokenizer_config)
        transform = Transform.parse(**model.transform_config["predict_transform"])
        return model, tokenizer, transform

    model, tokenizer, transform = _run_startup_stage(
        f"loading model from {MODEL_CKPT}", load_model_fn
    )

    # 4. Wait for bpy_server readiness (should already be done by now)
    bpy_thread.join()
    if bpy_error[0] is not None:
        raise bpy_error[0]

    state.model = model
    state.tokenizer = tokenizer
    state.transform = transform
    state.ready = True
    state.loaded_at = time.time()
    logger.info("Pipeline ready (fully resident in VRAM)")


# --------------------------------------------------------------------------- #
# Core generation logic
# --------------------------------------------------------------------------- #
def _run_generation(
    file_data: bytes,
    filename: str,
    params: GenParams,
    request_id: str,
    cancellation: CancellationToken,
    progress_callback=None,
) -> bytes:
    """Run the full rigging pipeline and return GLB bytes."""
    from pathlib import Path
    from torch import Tensor
    from src.data.dataset import DatasetConfig, RigDatasetModule
    from src.model.tokenrig import TokenRigResult
    from src.server.spec import BPY_SERVER, object_to_bytes, bytes_to_object
    from src.data.vertex_group import voxel_skin
    import requests as req_lib

    cancellation.raise_if_cancelled()

    # Write input file to temp location
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: {SUPPORTED_EXT}")

    tmp_input_dir = Path(tempfile.mkdtemp(prefix="skintokens_input_"))
    tmp_output_dir = Path(tempfile.mkdtemp(prefix="skintokens_output_"))
    input_path = tmp_input_dir / filename
    out_path = tmp_output_dir / f"{Path(filename).stem}.glb"

    try:
        input_path.write_bytes(file_data)
        cancellation.raise_if_cancelled()

        if progress_callback:
            progress_callback(5, "building dataset")

        datapath = {
            "data_name": None,
            "loader": "bpy_server",
            "filepaths": {"articulation": [str(input_path)]},
        }

        dataset_config = DatasetConfig.parse(
            shuffle=False,
            batch_size=1,
            num_workers=1,
            pin_memory=DEVICE.startswith("cuda"),
            persistent_workers=False,
            datapath=datapath,
        ).split_by_cls()

        module = RigDatasetModule(
            predict_dataset_config=dataset_config,
            predict_transform=state.transform,
            tokenizer=state.tokenizer,
            process_fn=state.model._process_fn,
        )

        dataloader = module.predict_dataloader()["articulation"]
        cancellation.raise_if_cancelled()

        if progress_callback:
            progress_callback(10, "running inference")

        for batch in dataloader:
            batch = {
                k: v.to(state.model.device, non_blocking=True) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }

            if not params.use_skeleton:
                batch.pop("skeleton_tokens", None)
                batch.pop("skeleton_mask", None)

            batch["generate_kwargs"] = dict(
                max_new_tokens=2048,
                top_k=params.top_k,
                top_p=params.top_p,
                temperature=params.temperature,
                repetition_penalty=params.repetition_penalty,
                num_return_sequences=1,
                num_beams=params.num_beams,
                do_sample=params.do_sample,
            )

            if "skeleton_tokens" in batch and "skeleton_mask" in batch:
                mask = batch["skeleton_mask"][0] == 1
                skeleton_tokens = batch["skeleton_tokens"][0][mask].cpu().numpy()
            else:
                skeleton_tokens = None

            cancellation.raise_if_cancelled()
            if progress_callback:
                progress_callback(30, "model sampling")

            _ps_t0 = time.monotonic()
            with torch.inference_mode():
                preds: List[TokenRigResult] = state.model.predict_step(
                    batch,
                    skeleton_tokens=[skeleton_tokens] if skeleton_tokens is not None else None,
                    make_asset=True,
                )["results"]
            _ps_t1 = time.monotonic()
            logger.info("[%s] predict_step total: %.3fs", request_id, _ps_t1 - _ps_t0)

            cancellation.raise_if_cancelled()
            if progress_callback:
                progress_callback(60, "post-processing")

            asset = preds[0].asset
            assert asset is not None

            if params.use_postprocess:
                voxel = asset.voxel(resolution=196)
                asset.skin *= voxel_skin(
                    grid=0,
                    grid_coords=voxel.coords,
                    joints=asset.joints,
                    vertices=asset.vertices,
                    faces=asset.faces,
                    mode="square",
                    voxel_size=voxel.voxel_size,
                )
                asset.normalize_skin()

            if progress_callback:
                progress_callback(75, "exporting GLB via bpy")

            out_path.parent.mkdir(parents=True, exist_ok=True)

            if params.use_transfer:
                payload = dict(
                    source_asset=asset,
                    target_path=asset.path,
                    export_path=str(out_path),
                    group_per_vertex=4,
                )
                res = _post_bpy_payload("transfer", payload)
            else:
                payload = dict(
                    asset=asset,
                    filepath=str(out_path),
                    group_per_vertex=4,
                )
                res = _post_bpy_payload("export", payload)

            cancellation.raise_if_cancelled()

            if res != "ok":
                raise RuntimeError(f"bpy export failed: {res}")

            if progress_callback:
                progress_callback(95, "reading output")

            glb_data = out_path.read_bytes()

            if progress_callback:
                progress_callback(100, "complete")

            return glb_data

        raise RuntimeError("No data in dataloader")
    finally:
        # Cleanup temp files
        import shutil
        with suppress(OSError):
            shutil.rmtree(tmp_input_dir, ignore_errors=True)
        with suppress(OSError):
            shutil.rmtree(tmp_output_dir, ignore_errors=True)


def _post_bpy_payload(endpoint: str, payload):
    """Send a payload to the bpy_server and return the result."""
    import requests as req_lib
    from src.server.spec import BPY_SERVER, object_to_bytes, bytes_to_object

    payload_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"skintokens_{endpoint}_", suffix=".pt", delete=False
        ) as f:
            serialized = object_to_bytes(payload)
            f.write(serialized)
            payload_path = f.name
        request_payload = {"payload_path": payload_path}
        response = req_lib.post(
            f"{BPY_SERVER}/{endpoint}",
            data=object_to_bytes(request_payload),
        )
        response.raise_for_status()
        result = bytes_to_object(response.content)
        if isinstance(result, dict) and result.get("error") is not None:
            raise RuntimeError(result.get("traceback") or result["error"])
        return result
    finally:
        if payload_path is not None:
            try:
                os.remove(payload_path)
            except OSError:
                pass


class ProgressReporter:
    """Per-request progress/timing."""

    def __init__(self, request_id: str, cancellation: CancellationToken,
                 progress_callback=None):
        self.request_id = request_id
        self.cancellation = cancellation
        self.progress_callback = progress_callback
        self.started_at = time.monotonic()
        self.last_report_at = self.started_at

    def raise_if_cancelled(self) -> None:
        self.cancellation.raise_if_cancelled()

    def report(self, percent: int, stage: str) -> None:
        self.raise_if_cancelled()
        now = time.monotonic()
        delta = now - self.last_report_at
        self.last_report_at = now
        elapsed = round(now - self.started_at, 2)
        logger.info("[%s] progress=%d%% stage=%s elapsed=%.2fs delta=%.2fs",
                    self.request_id, percent, stage, elapsed, delta)
        if self.progress_callback is not None:
            self.progress_callback(percent, stage, elapsed)
        self.raise_if_cancelled()


async def _generate(
    file_data: bytes,
    filename: str,
    params: GenParams,
    request_id: str,
    progress_callback=None,
    cancellation: Optional[CancellationToken] = None,
) -> bytes:
    """Run generation with lock serialization."""
    cancellation = cancellation or CancellationToken()
    reporter = ProgressReporter(request_id, cancellation, progress_callback)

    queued = state.generation_lock.locked()
    logger.info("[%s] queued=%s filename=%r", request_id, queued, filename)
    wait_started = time.monotonic()

    async with _acquire_or_cancel(state.generation_lock, cancellation):
        state.busy = True
        logger.info("[%s] generation lock acquired after %.2fs", request_id,
                    time.monotonic() - wait_started)
        try:
            glb = await _to_thread_cancellable(
                _run_generation, file_data, filename, params, request_id,
                cancellation, reporter.report,
                cancellation=cancellation,
            )
        finally:
            state.busy = False

    return glb


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
async def _watch_http_disconnect(
    request: Request, cancellation: CancellationToken
) -> None:
    while not cancellation.cancelled:
        if await request.is_disconnected():
            cancellation.cancel("HTTP client disconnected")
            return
        await asyncio.sleep(0.25)


async def _watch_ws_cancellation(
    ws: WebSocket, cancellation: CancellationToken, request_id: str
) -> bool:
    """Return True for an explicit cancel message, False for a disconnect."""
    try:
        while True:
            message = await ws.receive_json()
            message_type = message.get("type", message.get("action", ""))
            if str(message_type).lower() in {"cancel", "interrupt"}:
                logger.info("[%s] explicit cancellation requested", request_id)
                cancellation.cancel("client requested cancellation")
                return True
            logger.warning("[%s] ignoring WebSocket message type=%r",
                           request_id, message_type)
    except WebSocketDisconnect:
        logger.warning("[%s] WebSocket disconnected; cancelling generation",
                       request_id)
        cancellation.cancel("WebSocket client disconnected")
        return False
    except RuntimeError:
        cancellation.cancel("WebSocket client disconnected")
        return False


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(_load_pipeline)
    try:
        yield
    finally:
        state.model = None
        state.tokenizer = None
        state.transform = None
        state.ready = False
        if state.bpy_proc is not None and state.bpy_proc.poll() is None:
            logger.info("Shutting down bpy_server")
            if os.name == "nt":
                state.bpy_proc.terminate()
            else:
                os.killpg(os.getpgid(state.bpy_proc.pid), signal.SIGTERM)
        torch.cuda.empty_cache()


app = FastAPI(title="SkinTokens / TokenRig Inference Server", version="1.0",
              lifespan=lifespan)


@app.get("/health")
async def health():
    if not state.ready:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "busy": state.busy}


@app.get("/info")
async def info():
    return {
        "model_ckpt": MODEL_CKPT,
        "hf_path": HF_PATH,
        "device": DEVICE,
        "ready": state.ready,
        "busy": state.busy,
        "loaded_at": state.loaded_at,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/generate")
async def generate(
    request: Request,
    file: UploadFile = File(..., description="3D model file (OBJ/FBX/GLB)"),
    top_k: int = Form(5),
    top_p: float = Form(0.95),
    temperature: float = Form(1.0),
    repetition_penalty: float = Form(2.0),
    num_beams: int = Form(3),
    do_sample: bool = Form(False),
    use_skeleton: bool = Form(False),
    use_transfer: bool = Form(False),
    use_postprocess: bool = Form(False),
):
    """Multipart upload a 3D file -> binary GLB response with rigging."""
    request_id = uuid.uuid4().hex[:8]
    received_at = time.monotonic()
    logger.info("[%s] HTTP request received filename=%r content_type=%r",
                request_id, file.filename, file.content_type)

    if not state.ready:
        logger.warning("[%s] rejected: model still loading", request_id)
        return JSONResponse({"error": "model still loading"}, status_code=503)

    params = GenParams(
        top_k=top_k, top_p=top_p, temperature=temperature,
        repetition_penalty=repetition_penalty, num_beams=num_beams,
        do_sample=do_sample,
        use_skeleton=use_skeleton, use_transfer=use_transfer,
        use_postprocess=use_postprocess,
    )

    try:
        file_data = await file.read()
        logger.info("[%s] file read bytes=%d filename=%s params=%s",
                    request_id, len(file_data), file.filename, params.model_dump())
    except Exception as e:
        logger.warning("[%s] invalid file: %s", request_id, e)
        return JSONResponse({"error": f"invalid file: {e}"}, status_code=400)

    cancellation = CancellationToken()
    disconnect_watcher = asyncio.create_task(
        _watch_http_disconnect(request, cancellation)
    )
    try:
        glb = await _generate(
            file_data, file.filename or "input.obj", params, request_id,
            cancellation=cancellation,
        )
    except GenerationCancelled as e:
        logger.info("[%s] HTTP generation cancelled: %s", request_id, e)
        return JSONResponse({"error": str(e), "request_id": request_id}, status_code=499)
    except Exception as e:
        logger.exception("[%s] generation failed", request_id)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        disconnect_watcher.cancel()
        with suppress(asyncio.CancelledError):
            await disconnect_watcher

    logger.info("[%s] HTTP request completed status=200 bytes=%d elapsed=%.2fs",
                request_id, len(glb), time.monotonic() - received_at)
    return Response(
        content=glb,
        media_type="model/gltf-binary",
        headers={
            "Content-Disposition": f'attachment; filename="{Path(file.filename or "input").stem}.glb"'
        },
    )


@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    """
    WebSocket protocol:
      client -> {"file_base64": "...", "filename": "model.obj", ...params}
      client -> {"type": "cancel"}
      server -> {"stage": "queued"|"processing"|"done"|"cancelled"|"error", ...}
      final  -> {"stage": "done", "glb_base64": "..."}
    """
    request_id = uuid.uuid4().hex[:8]
    cancellation = None
    generation_task = None
    receiver_task = None

    await ws.accept()
    logger.info("[%s] WebSocket connected client=%s", request_id, ws.client)

    try:
        req = await ws.receive_json()
        logger.info("[%s] WebSocket generation request received", request_id)

        if not state.ready:
            logger.warning("[%s] rejected: model still loading", request_id)
            await ws.send_json({"stage": "error", "message": "model still loading"})
            await ws.close()
            return

        try:
            file_data = base64.b64decode(req.pop("file_base64"), validate=True)
            filename = req.pop("filename", "input.obj")
        except Exception as e:
            logger.warning("[%s] invalid file: %s", request_id, e)
            await ws.send_json({"stage": "error", "message": f"invalid file: {e}"})
            await ws.close()
            return

        params = GenParams(
            **{k: v for k, v in req.items() if k in GenParams.model_fields}
        )
        logger.info("[%s] file decoded bytes=%d filename=%s params=%s",
                    request_id, len(file_data), filename, params.model_dump())

        loop = asyncio.get_running_loop()
        cancellation = CancellationToken()

        def send_progress(percent, stage, elapsed):
            message = {
                "stage": "processing", "step": stage,
                "progress": percent, "elapsed_sec": elapsed,
                "request_id": request_id,
            }
            future = asyncio.run_coroutine_threadsafe(ws.send_json(message), loop)
            try:
                future.result(timeout=5)
            except Exception:
                future.cancel()
                cancellation.cancel("WebSocket client disconnected")

        queued = state.generation_lock.locked()
        logger.info("[%s] queued=%s", request_id, queued)
        await ws.send_json({
            "stage": "queued", "queued": queued, "request_id": request_id
        })
        await ws.send_json({
            "stage": "processing", "progress": 0, "request_id": request_id
        })

        t0 = time.time()
        generation_task = asyncio.create_task(
            _generate(
                file_data, filename, params, request_id, send_progress,
                cancellation=cancellation,
            )
        )
        receiver_task = asyncio.create_task(
            _watch_ws_cancellation(ws, cancellation, request_id)
        )
        done, _ = await asyncio.wait(
            (generation_task, receiver_task), return_when=asyncio.FIRST_COMPLETED
        )

        explicit_cancel = False
        if receiver_task in done:
            explicit_cancel = receiver_task.result()

        if cancellation.cancelled:
            try:
                await generation_task
            except GenerationCancelled:
                pass
            if explicit_cancel:
                await ws.send_json({
                    "stage": "cancelled",
                    "message": cancellation.reason,
                    "request_id": request_id,
                })
                await ws.close()
            logger.info("[%s] WebSocket generation cancelled: %s",
                        request_id, cancellation.reason)
            return

        receiver_task.cancel()
        with suppress(asyncio.CancelledError):
            await receiver_task

        glb = await generation_task
        await ws.send_json({
            "stage": "done",
            "elapsed_sec": round(time.time() - t0, 2),
            "progress": 100,
            "request_id": request_id,
            "glb_base64": base64.b64encode(glb).decode("ascii"),
        })
        logger.info("[%s] WebSocket request completed bytes=%d elapsed=%.2fs",
                    request_id, len(glb), time.time() - t0)
        await ws.close()

    except WebSocketDisconnect:
        logger.warning("[%s] WebSocket disconnected", request_id)
    except GenerationCancelled as e:
        logger.info("[%s] WebSocket generation cancelled: %s", request_id, e)
    except Exception as e:
        logger.exception("[%s] WebSocket generation failed", request_id)
        try:
            await ws.send_json({"stage": "error", "message": str(e)})
            await ws.close()
        except Exception:
            pass
    finally:
        if receiver_task is not None and not receiver_task.done():
            receiver_task.cancel()
            with suppress(asyncio.CancelledError):
                await receiver_task
        if generation_task is not None and not generation_task.done():
            if cancellation is not None:
                cancellation.cancel("WebSocket handler stopped")
            with suppress(asyncio.CancelledError, GenerationCancelled, Exception):
                await asyncio.shield(generation_task)


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def main():
    global MODEL_CKPT, HF_PATH

    parser = argparse.ArgumentParser(description="SkinTokens / TokenRig inference server")
    parser.add_argument("--host", default=os.environ.get("SKINTOKENS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SKINTOKENS_PORT", "8087")))
    parser.add_argument("--model-ckpt", default=MODEL_CKPT)
    parser.add_argument("--hf-path", default=HF_PATH)
    args = parser.parse_args()

    MODEL_CKPT = args.model_ckpt
    HF_PATH = args.hf_path

    import uvicorn

    class HealthCheckFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return '"GET /health' not in msg and '"GET /health ' not in msg

    logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

    log_config = uvicorn.config.LOGGING_CONFIG.copy()
    log_config["formatters"] = {
        name: {**formatter, "fmt": f"%(asctime)s.%(msecs)03d {formatter['fmt']}",
               "datefmt": LOG_DATE_FORMAT}
        for name, formatter in uvicorn.config.LOGGING_CONFIG["formatters"].items()
    }
    uvicorn.run(app, host=args.host, port=args.port, workers=1,
                ws_max_size=64 * 1024 * 1024, log_config=log_config)


if __name__ == "__main__":
    main()
