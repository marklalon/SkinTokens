import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

from torch import Tensor
from tqdm import tqdm

os.environ["XFORMERS_IGNORE_FLASH_VERSION_CHECK"] = "1"

from src.data.dataset import DatasetConfig, RigDatasetModule
from src.data.transform import Transform
from src.model.tokenrig import TokenRigResult
from src.tokenizer.parse import get_tokenizer
from src.server.spec import get_model
from src.data.vertex_group import voxel_skin
from src.paths import EXPERIMENTS_DIR
from src.rig_package.parser.bpy import transfer_rigging

MODEL_CKPTS = [
    str(EXPERIMENTS_DIR / "articulation_xl_quantization_256_token_4/grpo_1400.ckpt"),
]

model = None
tokenizer = None
transform = None
CURRENT_MODEL_CKPT: Optional[str] = None
CURRENT_HF_PATH: Optional[str] = None


def load_model(model_ckpt: str, hf_path: Optional[str], device: Optional[str] = None) -> Tuple[str, str]:
    global model, tokenizer, transform, CURRENT_MODEL_CKPT, CURRENT_HF_PATH
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


def run_rig(
    filepaths: List[Path],
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    num_beams: int,
    use_skeleton: bool,
    use_postprocess: bool,
    output_paths: List[Path],
    model_ckpt: str,
    hf_path: Optional[str],
):
    assert len(filepaths) == len(output_paths)

    load_model(model_ckpt, hf_path)

    datapath = {
        "data_name": None,
        "loader": "bpy",
        "filepaths": {"articulation": [str(p) for p in filepaths]},
    }

    dataset_config = DatasetConfig.parse(
        shuffle=False,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
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
        collapsed_joints = asset.collapse_near_parent_joints()
        if collapsed_joints:
            print(f"[postprocess] Collapsed near-parent joints: {', '.join(collapsed_joints)}")

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

        transfer_rigging(
            source_asset=asset,
            target_path=asset.path,
            export_path=str(out_path),
            group_per_vertex=4,
        )

        print(f"[OK] Exported: {out_path}")

        results_out.append(out_path)

    return results_out


def run_cli(args):
    input_path = Path(args.input).resolve()
    
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = (Path("outputs") / f"{input_path.stem}_bind.glb").resolve()

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
        args.use_postprocess,
        outputs,
        args.model_ckpt,
        args.hf_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TokenRig Demo")
    parser.add_argument("--input", help="Input file or directory")
    parser.add_argument("--output", default=None, help="Output file or directory (default: outputs/{input_name}_bind.glb)")

    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=5)

    parser.add_argument("--use_skeleton", action="store_true")
    parser.add_argument("--use_postprocess", action="store_true")

    parser.add_argument("--model_ckpt", default=MODEL_CKPTS[0] if MODEL_CKPTS else "")
    parser.add_argument("--hf_path", default=None)

    args = parser.parse_args()

    if not args.input:
        parser.error("--input is required for CLI mode. Use serve.py for the HTTP API server.")

    run_cli(args)
