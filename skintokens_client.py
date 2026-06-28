"""SkinTokens / TokenRig WebSocket client — stream progress and save a rigged GLB.

Example:
    python skintokens_client.py --file assets/character.obj --output out.glb
    python skintokens_client.py --file model.fbx --server http://HOST:8087 \
        --use-skeleton --use-postprocess
    python skintokens_client.py --file model.fbx --image char.png \
        --server http://HOST:8087

When ``--image`` is provided, a character reference image is sent through to the
skeleton-renamer service (``SKINTOKENS_SKELETON_RENAMER_URL``) for LLM-based
species/skeleton verification.

Requires the ``websockets`` package.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import suppress
from pathlib import Path


def _websocket_url(server: str) -> str:
    base = server.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    elif not base.startswith(("ws://", "wss://")):
        base = "ws://" + base
    return base + "/ws/generate"


class ProgressDisplay:
    """Render progress in place on a terminal, or as lines when redirected."""

    def __init__(self, width: int = 30):
        self.width = width
        self.last_length = 0
        self.active = False

    def update(self, percent: int, step: str, elapsed: float | None = None) -> None:
        percent = max(0, min(100, int(percent)))
        filled = round(self.width * percent / 100)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed_text = f"  {elapsed:.1f}s" if elapsed is not None else ""
        line = f"[client] [{bar}] {percent:3d}%  {step}{elapsed_text}"

        if sys.stdout.isatty():
            sys.stdout.write("\r" + line.ljust(self.last_length))
            sys.stdout.flush()
            self.last_length = max(self.last_length, len(line))
            self.active = True
        else:
            print(line)

    def finish(self) -> None:
        if self.active:
            print()
            self.active = False


def _sample_output_path(output_path: str, sample_index: int, total_samples: int) -> str:
    if total_samples <= 1:
        return output_path
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}_{sample_index}{path.suffix or '.glb'}"))


async def _run(args) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError(
            "missing dependency 'websockets'; install it with: pip install websockets"
        ) from exc

    ws_url = _websocket_url(args.server)
    with open(args.file, "rb") as f:
        file_data = f.read()

    filename = os.path.basename(args.file)

    image_data = None
    image_name = None
    if args.image:
        with open(args.image, "rb") as f:
            image_data = f.read()
        image_name = os.path.basename(args.image)
        print(f"[client] image attached: {args.image} ({len(image_data)} bytes)")

    payload = {
        "filename": filename,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "num_beams": args.num_beams,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "do_sample": True,
        "use_skeleton": args.use_skeleton,
        "use_postprocess": args.use_postprocess,
        "skip_renamer": args.skip_renamer,
        "image_size": len(image_data) if image_data else 0,
    }
    if image_name:
        payload["image_name"] = image_name

    progress = ProgressDisplay()
    started_at = time.monotonic()
    print(f"[client] connecting {ws_url}")
    try:
        async with websockets.connect(
            ws_url,
            max_size=64 * 1024 * 1024,
            open_timeout=args.timeout,
        ) as ws:
            await ws.send(json.dumps(payload))
            await ws.send(file_data)
            if image_data:
                await ws.send(image_data)

            glb_count = 0
            expected_samples = args.num_samples
            pending_sample_index = 0
            try:
                async for raw_message in ws:
                    if isinstance(raw_message, bytes):
                        glb = raw_message
                        sample_output = _sample_output_path(
                            args.output, pending_sample_index, expected_samples
                        )
                        glb_count += 1
                        output_dir = os.path.dirname(os.path.abspath(sample_output))
                        os.makedirs(output_dir, exist_ok=True)
                        with open(sample_output, "wb") as f:
                            f.write(glb)
                        print(f"[client] saved {len(glb)} bytes -> {sample_output}")
                        if glb_count >= expected_samples:
                            return
                        continue

                    message = json.loads(raw_message)
                    stage = message.get("stage", "unknown")

                    if stage == "queued":
                        if message.get("queued"):
                            print("[client] queued; waiting for the GPU")
                        else:
                            print("[client] GPU is available; starting generation")
                    elif stage == "processing":
                        progress.update(
                            message.get("progress", 0),
                            message.get("step", "processing"),
                            message.get("elapsed_sec"),
                        )
                    elif stage == "done":
                        expected_samples = int(message.get("num_samples", expected_samples))
                        pending_sample_index = int(message.get("sample_index", glb_count))
                        progress.update(100, "complete", message.get("elapsed_sec"))
                        progress.finish()
                        # Binary GLB frame follows the done message
                        renamer_meta = {k: v for k, v in message.items()
                                         if k not in ("stage", "glb_size",
                                                      "elapsed_sec", "progress",
                                                      "request_id", "sample_index",
                                                      "num_samples")}
                        if renamer_meta:
                            for k, v in renamer_meta.items():
                                print(f"[client] {k}: {v}")
                        # Continue loop to receive the binary frame
                        continue
                    elif stage == "cancelled":
                        raise asyncio.CancelledError(message.get("message"))
                    elif stage == "error":
                        err_msg = message.get("message") or "unknown server error"
                        raise RuntimeError(err_msg)
                    else:
                        print(f"[client] server stage: {stage}")
            except asyncio.CancelledError:
                with suppress(Exception):
                    await asyncio.shield(ws.send(json.dumps({"type": "cancel"})))
                raise

            raise RuntimeError("WebSocket closed before the result was received")
    finally:
        progress.finish()
        print(f"[client] request elapsed: {time.monotonic() - started_at:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SkinTokens / TokenRig WebSocket client with live progress"
    )
    parser.add_argument("--file", required=True, help="Input 3D file path (OBJ/FBX/GLB)")
    parser.add_argument("--image", default=None,
                        help="Character reference image (PNG/JPEG) forwarded to skeleton-renamer "
                             "for LLM-based species/skeleton verification")
    parser.add_argument(
        "--output",
        default=None,
        help="Output GLB path (default: outputs/<input_name>_skined.glb)",
    )
    parser.add_argument("--server", default="http://localhost:8087", help="Server base URL")

    parser.add_argument("--top-k", type=int, default=1, help="Top-k sampling (default: 1)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling (default: 1.0)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature (default: 0.1)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty (default: 1.0)")
    parser.add_argument("--num-beams", type=int, default=10, help="Number of beams (1-16, default: 10)")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of parallel samples to generate (1-8, default: 1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible generation")

    parser.add_argument("--use-skeleton", action="store_true", help="Use skeleton for skin generation")
    parser.add_argument("--use-postprocess", action="store_true", help="Use postprocess (voxel skin)")
    parser.add_argument("--skip-renamer", action="store_true", help="Skip the skeleton renamer step")

    parser.add_argument("--timeout", type=int, default=30, help="Connection timeout seconds")
    args = parser.parse_args()

    if args.output is None:
        input_name = os.path.splitext(os.path.basename(args.file))[0]
        args.output = f"outputs/{input_name}_skined.glb"

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n[client] cancelled", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        err_msg = str(exc).strip()
        if not err_msg:
            err_msg = f"{type(exc).__name__} (no detail – check server logs)"
        print(f"[client] error: {err_msg}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
