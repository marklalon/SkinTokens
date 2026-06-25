import argparse
import atexit
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Iterable, Optional, Tuple

import requests
from torch import Tensor
from tqdm import tqdm

os.environ["XFORMERS_IGNORE_FLASH_VERSION_CHECK"] = "1"

from src.data.dataset import DatasetConfig, RigDatasetModule
from src.data.transform import Transform
from src.model.tokenrig import TokenRigResult
from src.tokenizer.parse import get_tokenizer
from src.server.spec import (
    BPY_SERVER,
    get_model,
    object_to_bytes,
    bytes_to_object,
)
from src.data.vertex_group import voxel_skin
from src.paths import EXPERIMENTS_DIR

MODEL_CKPTS = [
    str(EXPERIMENTS_DIR / "articulation_xl_quantization_256_token_4/grpo_1400.ckpt"),
]

HF_PATHS = [
    "None",
]


def start_bpy_server():
    popen_kwargs = dict(
        args=[sys.executable, "bpy_server.py"],
        stdout=None,
        stderr=None,
    )
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(**popen_kwargs)
    print(f"[Main] bpy_server.py started (pid={proc.pid})")

    def cleanup():
        print(f"[Main] Terminating bpy_server.py (pid={proc.pid})")
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


model = None
tokenizer = None
transform = None
CURRENT_MODEL_CKPT: Optional[str] = None
CURRENT_HF_PATH: Optional[str] = None


def load_model(model_ckpt: str, hf_path: Optional[str], device: Optional[str] = None) -> Tuple[str, str]:
    global model, tokenizer, transform, CURRENT_MODEL_CKPT, CURRENT_HF_PATH
    if hf_path == "None":
        hf_path = None
    if model is not None and model_ckpt == CURRENT_MODEL_CKPT and hf_path == CURRENT_HF_PATH:
        return ("Model already loaded.", model_ckpt)

    if not model_ckpt:
        raise RuntimeError("model_ckpt is empty. Please select a checkpoint.")

    print(f"Loading model: {model_ckpt}, hf_path={hf_path}")
    model = get_model(model_ckpt, hf_path=hf_path, device=device)
    assert model.tokenizer_config is not None
    tokenizer = get_tokenizer(**model.tokenizer_config)
    transform = Transform.parse(**model.transform_config["predict_transform"])
    CURRENT_MODEL_CKPT = model_ckpt
    CURRENT_HF_PATH = hf_path
    return ("Model loaded.", model_ckpt)


SUPPORTED_EXT = {".obj", ".fbx", ".glb"}


def collect_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    files = []
    for p in input_path.rglob("*"):
        if p.suffix.lower() in SUPPORTED_EXT:
            files.append(p)
    return files


def map_output_path(
    in_path: Path,
    input_root: Path,
    output_root: Path,
) -> Path:
    rel = in_path.relative_to(input_root)
    return (output_root / rel).with_suffix(".glb")


def post_bpy_payload(endpoint: str, payload):
    payload_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f"skintokens_{endpoint}_", suffix=".pt", delete=False) as f:
            f.write(object_to_bytes(payload))
            payload_path = f.name
        request_payload = {"payload_path": payload_path}
        response = requests.post(
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


def run_rig(
    filepaths: List[Path],
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    num_beams: int,
    use_skeleton: bool,
    use_transfer: bool,
    use_postprocess: bool,
    output_paths: List[Path],
    model_ckpt: str,
    hf_path: Optional[str],
):
    assert len(filepaths) == len(output_paths)

    load_model(model_ckpt, hf_path)

    datapath = {
        "data_name": None,
        "loader": "bpy_server",
        "filepaths": {"articulation": [str(p) for p in filepaths]},
    }

    dataset_config = DatasetConfig.parse(
        shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory=str(model.device).startswith("cuda"),
        persistent_workers=False,
        datapath=datapath,
    ).split_by_cls()

    module = RigDatasetModule(
        predict_dataset_config=dataset_config,
        predict_transform=transform,
        tokenizer=tokenizer,
        process_fn=model._process_fn,
    )

    dataloader = module.predict_dataloader()["articulation"]

    results_out = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = {
            k: v.to(model.device, non_blocking=True) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

        if not use_skeleton:
            batch.pop("skeleton_tokens", None)
            batch.pop("skeleton_mask", None)

        batch["generate_kwargs"] = dict(
            max_new_tokens=2048,
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            num_return_sequences=1,
            num_beams=int(num_beams),
            do_sample=True,
        )

        if "skeleton_tokens" in batch and "skeleton_mask" in batch:
            mask = batch["skeleton_mask"][0] == 1
            skeleton_tokens = batch["skeleton_tokens"][0][mask].cpu().numpy()
        else:
            skeleton_tokens = None

        preds: List[TokenRigResult] = model.predict_step(
            batch,
            skeleton_tokens=[skeleton_tokens] if skeleton_tokens is not None else None,
            make_asset=True,
        )["results"]

        asset = preds[0].asset
        assert asset is not None

        if use_postprocess:
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

        out_path = output_paths[i]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if use_transfer:
            payload = dict(
                source_asset=asset,
                target_path=asset.path,
                export_path=str(out_path),
                group_per_vertex=4,
            )
            res = post_bpy_payload("transfer", payload)
        else:
            payload = dict(
                asset=asset,
                filepath=str(out_path),
                group_per_vertex=4,
            )
            res = post_bpy_payload("export", payload)

        if res != "ok":
            print(f"[Error] {res}")
        else:
            print(f"[OK] Exported: {out_path}")

        results_out.append(out_path)

    return results_out


def run_cli(args):
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    files = collect_files(input_path)
    if not files:
        raise RuntimeError("No valid 3D files found.")

    if len(files) == 1 and output_path.suffix:
        outputs = [output_path]
    else:
        outputs = [
            map_output_path(f, input_path, output_path)
            for f in files
        ]

    run_rig(
        files,
        args.top_k,
        args.top_p,
        args.temperature,
        args.repetition_penalty,
        args.num_beams,
        args.use_skeleton,
        args.use_transfer,
        args.use_postprocess,
        outputs,
        args.model_ckpt,
        args.hf_path,
    )


def wait_for_bpy_server(timeout=30):
    t0 = time.time()
    while True:
        try:
            requests.get(f"{BPY_SERVER}/ping", timeout=1)
            print("[Main] bpy_server is ready")
            return
        except Exception:
            if time.time() - t0 > timeout:
                raise RuntimeError("bpy_server failed to start")
            time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("TokenRig Demo")
    parser.add_argument("--input", help="Input file or directory")
    parser.add_argument("--output", help="Output file or directory")

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=2.0)
    parser.add_argument("--num_beams", type=int, default=10)

    parser.add_argument("--use_skeleton", action="store_true")
    parser.add_argument("--use_transfer", action="store_true")
    parser.add_argument("--use_postprocess", action="store_true")

    parser.add_argument("--model_ckpt", default=MODEL_CKPTS[0] if MODEL_CKPTS else "")
    parser.add_argument("--hf_path", default=None)

    args = parser.parse_args()

    if not args.input:
        parser.error("--input is required for CLI mode. Use serve.py for the HTTP API server.")

    server_proc = start_bpy_server()
    wait_for_bpy_server()

    run_cli(args)
