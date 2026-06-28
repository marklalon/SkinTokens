"""
SkinTokens / TokenRig inference server — FastAPI persistent service.

The model pipeline is loaded once at startup and kept resident in VRAM.
Clients submit a 3D file (OBJ/FBX/GLB) and receive a rigged GLB over WebSocket. Generations are serialized through a single-GPU work queue so the
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
import json
import logging
import numpy as np
import random
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import (
    asynccontextmanager,
    contextmanager,
    redirect_stderr,
    redirect_stdout,
    suppress,
)
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
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
SKELETON_RENAMER_URL = os.environ.get(
    "SKINTOKENS_SKELETON_RENAMER_URL",
    "http://skeleton-renamer:8088",
)


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


@contextmanager
def _suppress_bpy_output():
    """Silence Blender/bpy stdout, stderr, and addon logging during bpy calls."""
    previous_disable = logging.root.manager.disable
    redirected_fds: list[tuple[int, int | None]] = []

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        try:
            for fd, stream in ((1, sys.stdout), (2, sys.stderr)):
                with suppress(Exception):
                    stream.flush()
                try:
                    saved_fd = os.dup(fd)
                except OSError:
                    saved_fd = None
                else:
                    os.dup2(devnull.fileno(), fd)
                redirected_fds.append((fd, saved_fd))

            with redirect_stdout(devnull), redirect_stderr(devnull):
                logging.disable(logging.CRITICAL)
                yield
        finally:
            logging.disable(previous_disable)
            for fd, saved_fd in reversed(redirected_fds):
                if saved_fd is None:
                    continue
                os.dup2(saved_fd, fd)
                os.close(saved_fd)


# --------------------------------------------------------------------------- #
# Global state
# --------------------------------------------------------------------------- #
class ServerState:
    model = None
    tokenizer = None
    transform = None
    generation_executor: Optional[ThreadPoolExecutor] = None
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
    loop = asyncio.get_running_loop()
    worker = loop.run_in_executor(state.generation_executor, operation, *args)
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
    top_k: int = 1
    top_p: float = 1.0
    temperature: float = 0.1
    repetition_penalty: float = 1.0
    num_beams: int = Field(default=10, ge=1, le=16)
    num_samples: int = Field(default=1, ge=1, le=8)
    seed: int | None = None
    use_skeleton: bool = False
    use_postprocess: bool = False
    skip_renamer: bool = False


# --------------------------------------------------------------------------- #
# Pipeline loading
# --------------------------------------------------------------------------- #
def _load_bpy_inproc():
    """Import bpy-backed parser on the dedicated generation thread."""
    with _suppress_bpy_output():
        from src.rig_package.parser.bpy import BpyParser  # noqa: F401


def _load_pipeline():
    """Load the model pipeline and initialize the bpy runtime."""
    from src.data.transform import Transform
    from src.tokenizer.parse import get_tokenizer
    from src.server.spec import get_model

    if state.generation_executor is None:
        state.generation_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="skintokens-gen",
        )

    def load_bpy_fn():
        assert state.generation_executor is not None
        state.generation_executor.submit(_load_bpy_inproc).result()

    _run_startup_stage("initializing geometry runtime", load_bpy_fn)

    def load_model_fn():
        model = get_model(MODEL_CKPT, hf_path=HF_PATH, device=DEVICE)
        tokenizer = get_tokenizer(**model.tokenizer_config)
        transform = Transform.parse(**model.transform_config["predict_transform"])
        return model, tokenizer, transform

    model, tokenizer, transform = _run_startup_stage(
        f"loading model from {MODEL_CKPT}", load_model_fn
    )

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
    image_data: bytes | None = None,
    image_name: str | None = None,
) -> list[tuple[bytes, dict]]:
    """Run the full rigging pipeline and return a list of (glb_bytes, renamer_meta) tuples,
    one per sample (controlled by num_samples in GenParams)."""
    from pathlib import Path
    from torch import Tensor
    from src.data.dataset import DatasetConfig, RigDatasetModule
    from src.model.tokenrig import TokenRigResult
    from src.data.vertex_group import voxel_skin

    cancellation.raise_if_cancelled()

    # Write input file to temp location
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: {SUPPORTED_EXT}")

    tmp_input_dir = Path(tempfile.mkdtemp(prefix="skintokens_input_"))
    tmp_output_dir = Path(tempfile.mkdtemp(prefix="skintokens_output_"))
    input_path = tmp_input_dir / filename

    try:
        input_path.write_bytes(file_data)
        cancellation.raise_if_cancelled()

        if progress_callback:
            progress_callback(5, "building dataset")

        datapath = {
            "data_name": None,
            "loader": "bpy",
            "filepaths": {"articulation": [str(input_path)]},
        }

        dataset_config = DatasetConfig.parse(
            shuffle=False,
            batch_size=1,
            num_workers=0,
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

        with _suppress_bpy_output():
            dataloader = module.predict_dataloader()["articulation"]
        cancellation.raise_if_cancelled()

        if progress_callback:
            progress_callback(10, "running inference")

        batch_iterator = iter(dataloader)
        while True:
            with _suppress_bpy_output():
                try:
                    batch = next(batch_iterator)
                except StopIteration:
                    batch = None

            if batch is None:
                break

            batch = {
                k: v.to(state.model.device, non_blocking=True) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }

            if not params.use_skeleton:
                batch.pop("skeleton_tokens", None)
                batch.pop("skeleton_mask", None)

            if params.seed is not None:
                torch.manual_seed(params.seed)
                torch.cuda.manual_seed_all(params.seed)
                np.random.seed(params.seed)
                random.seed(params.seed)
            batch["generate_kwargs"] = dict(
                max_new_tokens=2048,
                top_k=params.top_k,
                top_p=params.top_p,
                temperature=params.temperature,
                repetition_penalty=params.repetition_penalty,
                num_beams=params.num_beams,
                num_samples=params.num_samples,
                do_sample=True,
            )

            if "skeleton_tokens" in batch and "skeleton_mask" in batch:
                mask = batch["skeleton_mask"][0] == 1
                skeleton_tokens = batch["skeleton_tokens"][0][mask].cpu().numpy()
            else:
                skeleton_tokens = None

            cancellation.raise_if_cancelled()
            if progress_callback:
                progress_callback(10, "model sampling")

            with torch.inference_mode():
                preds: List[TokenRigResult] = state.model.predict_step(
                    batch,
                    skeleton_tokens=[skeleton_tokens] if skeleton_tokens is not None else None,
                    make_asset=True,
                    progress_callback=progress_callback if progress_callback else None,
                )["results"]

            cancellation.raise_if_cancelled()
            # --- Export each sample as a separate GLB ---
            all_glbs: list[tuple[bytes, dict]] = []
            from src.rig_package.parser.bpy import transfer_rigging
            sample_count = max(len(preds), 1)
            sample_progress_start = 75
            sample_progress_span = 24

            def report_sample_progress(sample_idx: int, phase: str) -> None:
                if not progress_callback:
                    return
                base = sample_progress_start + (sample_progress_span * sample_idx) // sample_count
                done = sample_progress_start + (
                    sample_progress_span * (sample_idx + 1)
                ) // sample_count
                mid = base + max(1, (done - base) // 2)
                sample_label = f"sample {sample_idx + 1}/{sample_count}"
                if phase == "start":
                    if sample_idx > 0:
                        return
                    step = f"exporting {sample_label}"
                    percent = base
                elif phase == "exported":
                    step = f"exported {sample_label}"
                    percent = min(mid, done - 1)
                elif phase == "complete":
                    step = f"finished {sample_label}"
                    percent = done
                else:
                    raise ValueError(f"unknown sample progress phase: {phase}")
                progress_callback(percent, step)

            for sample_idx, pred in enumerate(preds):
                cancellation.raise_if_cancelled()
                report_sample_progress(sample_idx, "start")

                asset = pred.asset
                assert asset is not None
                collapsed_joints = asset.collapse_near_parent_joints()
                if collapsed_joints:
                    logger.info(
                        "[%s] sample=%d collapsed near-parent skeleton joints: %s",
                        request_id,
                        sample_idx,
                        ", ".join(collapsed_joints),
                    )

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

                sample_out_path = tmp_output_dir / f"sample_{sample_idx}.glb"
                sample_out_path.parent.mkdir(parents=True, exist_ok=True)

                with _suppress_bpy_output():
                    transfer_rigging(
                        source_asset=asset,
                        target_path=asset.path,
                        export_path=str(sample_out_path),
                        group_per_vertex=4,
                        auto_ground=True,
                    )

                glb_data = sample_out_path.read_bytes()
                cancellation.raise_if_cancelled()
                report_sample_progress(sample_idx, "exported")

                if params.skip_renamer:
                    logger.info("[%s] sample=%d skipping skeleton renamer", request_id, sample_idx)
                    all_glbs.append((glb_data, {}))
                else:
                    renamed_data, renamer_meta = _run_skeleton_rename(
                        glb_data,
                        file_name=filename,
                        conf_thresh=0.8,
                        request_id=request_id,
                        image_data=image_data,
                        image_name=image_name,
                    )
                    all_glbs.append((renamed_data, renamer_meta))
                report_sample_progress(sample_idx, "complete")

            if progress_callback:
                progress_callback(100, "complete")

            return all_glbs

        raise RuntimeError("No data in dataloader")
    finally:
        # Cleanup temp files
        import shutil
        with suppress(OSError):
            shutil.rmtree(tmp_input_dir, ignore_errors=True)
        with suppress(OSError):
            shutil.rmtree(tmp_output_dir, ignore_errors=True)


def _run_skeleton_rename(
    glb_data: bytes,
    file_name: str,
    conf_thresh: float,
    request_id: str,
    image_data: bytes | None = None,
    image_name: str | None = None,
) -> bytes:
    """Send GLB data to the remote skeleton renamer service via WebSocket and
    return the renamed GLB bytes.

    If ``image_data`` / ``image_name`` are provided they are forwarded to the
    renamer for LLM-based species/skeleton verification.
    """
    import asyncio as _asyncio
    import json as _json

    ws_url = SKELETON_RENAMER_URL.rstrip("/")
    if ws_url.startswith("https://"):
        ws_url = "wss://" + ws_url[len("https://"):]
    elif ws_url.startswith("http://"):
        ws_url = "ws://" + ws_url[len("http://"):]
    elif not ws_url.startswith(("ws://", "wss://")):
        ws_url = "ws://" + ws_url
    ws_url += "/ws/skeleton-renamer"

    async def _rename():
        import websockets as _ws
        image_size = len(image_data) if image_data else 0
        payload_dict: dict = {
            "file_name": file_name,
            "conf_thresh": conf_thresh,
            "image_size": image_size,
        }
        if image_data and image_name:
            payload_dict["image_name"] = image_name
        payload = _json.dumps(payload_dict)
        async with _ws.connect(
            ws_url,
            max_size=64 * 1024 * 1024,
            open_timeout=30,
        ) as ws:
            await ws.send(payload)
            await ws.send(glb_data)
            if image_data:
                await ws.send(image_data)
            async for raw_message in ws:
                message = _json.loads(raw_message)
                stage = message.get("stage", "unknown")
                if stage == "done":
                    glb_size = message.get("glb_size", 0)
                    if not glb_size:
                        raise RuntimeError("renamer returned done without glb_size")
                    renamed_bytes = await ws.recv()
                    if isinstance(renamed_bytes, str):
                        renamed_bytes = renamed_bytes.encode()
                    renamer_meta = {k: v for k, v in message.items()
                                    if k not in ("stage", "glb_size")}
                    return renamed_bytes, renamer_meta
                elif stage == "error":
                    raise RuntimeError(message.get("message", "unknown renamer error"))
                elif stage == "cancelled":
                    raise RuntimeError(f"renamer cancelled: {message.get('message', '')}")
            raise RuntimeError("WebSocket closed before renamer result")

    logger.info("[%s] calling skeleton renamer at %s (file=%s, conf_thresh=%.2f, image=%s)",
                request_id, ws_url, file_name, conf_thresh,
                image_name if image_name else "none")
    try:
        renamed, renamer_meta = _asyncio.run(_rename())
        logger.info("[%s] skeleton rename done", request_id)
        return renamed, renamer_meta
    except Exception:
        logger.error("[%s] skeleton renamer failed", request_id)
        raise


class ProgressReporter:
    """Per-request progress/timing."""

    CLIENT_ONLY_STAGE_PREFIXES = ("model sampling",)

    def __init__(self, request_id: str, cancellation: CancellationToken,
                 progress_callback=None):
        self.request_id = request_id
        self.cancellation = cancellation
        self.progress_callback = progress_callback
        self.started_at = time.monotonic()
        self.last_logged_at = self.started_at

    def raise_if_cancelled(self) -> None:
        self.cancellation.raise_if_cancelled()

    def _should_log(self, stage: str) -> bool:
        return not any(stage.startswith(prefix)
                       for prefix in self.CLIENT_ONLY_STAGE_PREFIXES)

    def report(self, percent: int, stage: str) -> None:
        self.raise_if_cancelled()
        now = time.monotonic()
        elapsed = round(now - self.started_at, 2)
        if self._should_log(stage):
            delta = now - self.last_logged_at
            self.last_logged_at = now
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
    image_data: bytes | None = None,
    image_name: str | None = None,
):
    """Run generation with lock serialization.

    Returns:
        List of (glb_bytes, renamer_meta) tuples, one per sample.
        renamer_meta is a dict of all extra fields from the skeleton-renamer
        "done" message (e.g. ``species``, ``species_tags``, ``joint_count``).
    """
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
            results = await _to_thread_cancellable(
                _run_generation, file_data, filename, params, request_id,
                cancellation, reporter.report, image_data, image_name,
                cancellation=cancellation,
            )
        finally:
            state.busy = False

    return results


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
        if state.generation_executor is not None:
            state.generation_executor.shutdown(wait=False, cancel_futures=True)
            state.generation_executor = None
        torch.cuda.empty_cache()


app = FastAPI(title="SkinTokens / TokenRig Inference Server", version="1.0",
              lifespan=lifespan)


@app.get("/health")
async def health():
    if not state.ready:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "busy": state.busy}


@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    """
    WebSocket protocol:
      client -> {"filename": "model.obj", "image_name": "...", "image_size": N, ...params}
      client -> <binary: file data>
      client -> <binary: image data>   (only if image_size > 0)
      client -> {"type": "cancel"}
      server -> {"stage": "queued"|"processing"|"done"|"cancelled"|"error", ...}
      server -> {"stage": "done", "glb_size": N, ...}
      server -> <binary: GLB data>
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

        filename = req.pop("filename", "input.obj")
        image_name: str | None = req.pop("image_name", None)
        image_size: int = req.pop("image_size", 0)

        # Receive binary file data
        try:
            file_data = await ws.receive_bytes()
        except Exception as e:
            logger.warning("[%s] invalid file: %s", request_id, e)
            await ws.send_json({"stage": "error", "message": f"invalid file: {e}"})
            await ws.close()
            return

        # Receive optional image for skeleton-renamer
        image_data: bytes | None = None
        if image_size > 0:
            image_data = await ws.receive_bytes()

        params = GenParams(
            **{k: v for k, v in req.items() if k in GenParams.model_fields}
        )
        logger.info("[%s] file decoded bytes=%d filename=%s params=%s image=%s",
                    request_id, len(file_data), filename, params.model_dump(),
                    image_name if image_name else "none")

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
                image_data=image_data, image_name=image_name,
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

        results = await generation_task  # list of (glb, renamer_meta)

        for sample_idx, (glb, renamer_meta) in enumerate(results):
            done_msg: dict = {
                "stage": "done",
                "sample_index": sample_idx,
                "num_samples": len(results),
                "elapsed_sec": round(time.time() - t0, 2),
                "progress": 100,
                "request_id": request_id,
                "glb_size": len(glb),
            }
            done_msg.update(renamer_meta)
            await ws.send_json(done_msg)
            await ws.send_bytes(glb)

        logger.info("[%s] WebSocket request completed samples=%d elapsed=%.2fs",
                    request_id, len(results), time.time() - t0)
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
